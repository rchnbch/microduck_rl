"""Behaviour-descriptor axes — the candidates, the accumulator, and the choice.

Walking-v2 binned its archive on per-foot ground-contact duty factor, and
measured that this robot cannot move in it: six teacher gaits spanning a 6x
speed range land within 0.03 of each other, and 256 byte-identical replicas of
one genome spread only 0.014. A QD archive is only as interesting as the axes
it is binned on, so v3 replaces the axes rather than the search.

This module is the machinery for doing that **by measurement**:

* :class:`GaitStats` folds one batched rollout into every candidate axis at
  once — contact statistics, posture statistics, motion statistics and an
  actuator-power proxy — so the whole candidate table costs one rollout per
  genome rather than one per axis.
* :data:`AXES` is the catalogue: name, label, which extra readouts the axis
  needs from the harness, and a plausible range.
* :class:`DescriptorCfg` names the two axes an archive is binned on plus their
  grid ranges. Its default is v2's duty-factor pair, so every v1/v2 code path
  keeps its old behaviour until told otherwise.

Two properties every axis here is designed for, because they are what v2's
descriptor failed:

**Repeatable.** MuJoCo-Warp's contact solve is order-sensitive, and a walking
biped amplifies that into a 0.605 m spread in displacement across identical
worlds (``qd.check_repeatability``). Anything computed as a *time average over
the whole episode* — cadence, crouch depth, actuator power — averages that
chaos out; anything proportional to the final displacement inherits it whole.
The candidates include both kinds on purpose, and
:mod:`qd.select_descriptor` measures which is which rather than assuming.

**Only meaningful while upright.** Every accumulation here is masked by the
same ``counted`` flag :class:`qd.common.RolloutMetrics` uses for fitness, so a
fallen robot contributes nothing to its own descriptor. Under v3's unanimous
8-replica insertion gate every archived elite is a full-episode survivor, so
its descriptor is always averaged over the full 7 s.

Nothing in here imports mjlab or touches a GPU beyond the tensors it is handed,
so it is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_CONTROL_DT: float = 0.02
"""50 Hz — the repo's control rate, and what every QD harness runs at."""

ROBOT_MASS_KG: float = 0.8
"""Microduck's ~800 g (AGENTS.md), used only for the cost-of-transport proxy."""

# Extra per-step readouts an axis may need from the harness, beyond the base
# position / projected gravity / foot contact every rollout already reports.
# Named so a harness can advertise exactly what it supplies and an axis that
# is not supplied comes back NaN instead of silently wrong.
EXTRA_CHANNELS: tuple[str, ...] = (
    "lin_vel_w",  # (N, 3) trunk linear velocity, world frame
    "ang_vel_b",  # (N, 3) trunk angular velocity, base frame
    "joint_vel",  # (N, nv) joint velocities
    "qfrc_actuator",  # (N, nv) actuator force in joint space
)


@dataclass(frozen=True)
class GaitAxis:
    """One candidate behaviour-descriptor axis."""

    name: str
    label: str
    """Human-readable, with units — used for plot axes and the viewer."""

    needs: tuple[str, ...] = ()
    """Extra channels required; empty means base readouts are enough."""

    default_range: tuple[float, float] = (0.0, 1.0)
    """A plausible grid range, superseded by whatever the measurement set says."""


AXES: dict[str, GaitAxis] = {
    a.name: a
    for a in (
        # --- contact ------------------------------------------------------- #
        GaitAxis("duty_left", "left-foot duty factor"),
        GaitAxis("duty_right", "right-foot duty factor"),
        GaitAxis("duty_mean", "mean duty factor"),
        GaitAxis("double_support", "double-support fraction"),
        GaitAxis("flight_fraction", "flight fraction (both feet off)"),
        GaitAxis("step_frequency", "step frequency [Hz]", (), (0.0, 6.0)),
        GaitAxis("stride_length", "stride length [m/step]", (), (0.0, 0.15)),
        # --- posture ------------------------------------------------------- #
        GaitAxis("torso_height_mean", "mean trunk height [m]", (), (0.07, 0.13)),
        GaitAxis("torso_height_osc", "trunk-height oscillation [m]", (), (0.0, 0.02)),
        GaitAxis("trunk_lean", "mean forward lean (gravity_x)", (), (-0.5, 0.5)),
        GaitAxis("tilt_mean", "mean tilt magnitude", (), (0.0, 0.5)),
        # --- motion -------------------------------------------------------- #
        GaitAxis("forward_speed", "forward speed [m/s]", (), (-0.1, 0.5)),
        GaitAxis("lateral_drift_rate", "lateral drift rate [m/s]", (), (0.0, 0.2)),
        GaitAxis("lateral_speed", "mean |lateral velocity| [m/s]", ("lin_vel_w",), (0.0, 0.3)),
        GaitAxis("yaw_rate", "mean |yaw rate| [rad/s]", ("ang_vel_b",), (0.0, 1.5)),
        GaitAxis("joint_speed", "mean |joint velocity| [rad/s]", ("joint_vel",), (0.0, 4.0)),
        # --- effort -------------------------------------------------------- #
        GaitAxis("power", "actuator power [W]", ("joint_vel", "qfrc_actuator"), (0.0, 6.0)),
        GaitAxis(
            "energy_per_meter",
            "actuator energy per metre [J/m]",
            ("joint_vel", "qfrc_actuator"),
            (0.0, 60.0),
        ),
        GaitAxis(
            "cost_of_transport",
            "cost of transport [-]",
            ("joint_vel", "qfrc_actuator"),
            (0.0, 8.0),
        ),
    )
}

AXIS_NAMES: tuple[str, ...] = tuple(AXES)


class GaitStats:
    """Per-env accumulator for every candidate axis, over the upright steps.

    Folded in by :class:`qd.common.RolloutMetrics`, which owns the fall latch
    and passes the same ``counted`` mask it uses for fitness — so no
    post-fall frame reaches any descriptor, exactly as none reaches the
    objective.

    Everything is a running sum over steps; the per-axis arithmetic happens
    once in :meth:`finalize`. The only per-step state that is not a sum is the
    previous foot-contact mask, which is what makes touchdown counting (and so
    step frequency and stride length) possible without storing a trajectory.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        control_dt: float = DEFAULT_CONTROL_DT,
    ):
        self.num_envs = num_envs
        self.device = device
        self.control_dt = float(control_dt)
        self.supplied: set[str] = set()
        """Extra channels actually seen, so a missing one reads NaN not zero."""
        f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
        i = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)
        self.start_pos = f(num_envs, 3)
        self.frozen_pos = f(num_envs, 3)
        self.contact_steps = i(num_envs, 2)
        self.both_steps = i(num_envs)
        self.flight_steps = i(num_envs)
        self.touchdowns = i(num_envs, 2)
        self.prev_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.sum_z = f(num_envs)
        self.sum_z2 = f(num_envs)
        self.sum_gx = f(num_envs)
        self.sum_tilt = f(num_envs)
        self.sum_abs_vy = f(num_envs)
        self.sum_abs_yaw = f(num_envs)
        self.sum_joint_speed = f(num_envs)
        self.sum_power = f(num_envs)

    def begin(self, base_pos: torch.Tensor) -> None:
        pos = torch.nan_to_num(base_pos.to(torch.float32))
        self.start_pos.copy_(pos)
        self.frozen_pos.copy_(pos)
        for t in (
            self.contact_steps,
            self.both_steps,
            self.flight_steps,
            self.touchdowns,
            self.sum_z,
            self.sum_z2,
            self.sum_gx,
            self.sum_tilt,
            self.sum_abs_vy,
            self.sum_abs_yaw,
            self.sum_joint_speed,
            self.sum_power,
        ):
            t.zero_()
        self.prev_contact.zero_()
        self.supplied = set()

    def update(
        self,
        counted: torch.Tensor,
        base_pos: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        foot_contact: torch.Tensor,
        extras: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Fold one control step in for the envs in ``counted``."""
        c = counted
        cf = c.float()
        pos = torch.nan_to_num(base_pos.to(torch.float32))
        grav = torch.nan_to_num(projected_gravity_b.to(torch.float32))
        contact = foot_contact.bool()

        self.frozen_pos = torch.where(c.unsqueeze(-1), pos, self.frozen_pos)

        both = contact.all(dim=-1)
        none = ~contact.any(dim=-1)
        touchdown = contact & ~self.prev_contact
        self.contact_steps += (contact & c.unsqueeze(-1)).long()
        self.both_steps += (both & c).long()
        self.flight_steps += (none & c).long()
        self.touchdowns += (touchdown & c.unsqueeze(-1)).long()
        # Only advance the edge detector for envs still upright, so the frame a
        # fallen robot's soles slap the floor is not a step.
        self.prev_contact = torch.where(c.unsqueeze(-1), contact, self.prev_contact)

        z = pos[:, 2]
        self.sum_z += cf * z
        self.sum_z2 += cf * z * z
        self.sum_gx += cf * grav[:, 0]
        self.sum_tilt += cf * torch.linalg.vector_norm(grav[:, :2], dim=-1)

        if not extras:
            return
        for key, value in extras.items():
            if value is None:
                continue
            self.supplied.add(key)
            v = torch.nan_to_num(value.to(torch.float32))
            if key == "lin_vel_w":
                self.sum_abs_vy += cf * v[:, 1].abs()
            elif key == "ang_vel_b":
                self.sum_abs_yaw += cf * v[:, 2].abs()
            elif key == "joint_vel":
                self.sum_joint_speed += cf * v.abs().mean(dim=-1)
            elif key == "qfrc_actuator":
                joint_vel = extras.get("joint_vel")
                if joint_vel is not None:
                    power = (v * torch.nan_to_num(joint_vel.to(torch.float32))).abs()
                    self.sum_power += cf * power.sum(dim=-1)

    def finalize(self, alive_steps: torch.Tensor) -> dict[str, np.ndarray]:
        """Every candidate axis for this rollout, as ``{name: (N,) float32}``.

        Axes whose extra channel was never supplied come back all-NaN, which is
        the honest reading: the harness did not measure that quantity.
        """
        steps = alive_steps.clamp_min(1).float()
        alive_t = steps * self.control_dt
        delta = self.frozen_pos - self.start_pos
        dx, dy = delta[:, 0], delta[:, 1]
        planar = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        touch_total = self.touchdowns.sum(dim=-1).float()

        mean_z = self.sum_z / steps
        var_z = (self.sum_z2 / steps - mean_z * mean_z).clamp_min(0.0)
        power = self.sum_power / steps
        energy = power * alive_t
        # Guarded against a policy that goes nowhere: below a centimetre the
        # ratio is meaningless, so it saturates at the range's top rather than
        # exploding into the grid's last bin from a division by ~0.
        travelled = planar.clamp_min(0.01)

        out = {
            "duty_left": self.contact_steps[:, 0].float() / steps,
            "duty_right": self.contact_steps[:, 1].float() / steps,
            "duty_mean": self.contact_steps.sum(dim=-1).float() / (2.0 * steps),
            "double_support": self.both_steps.float() / steps,
            "flight_fraction": self.flight_steps.float() / steps,
            "step_frequency": touch_total / alive_t,
            "stride_length": planar / touch_total.clamp_min(1.0),
            "torso_height_mean": mean_z,
            "torso_height_osc": var_z.sqrt(),
            "trunk_lean": self.sum_gx / steps,
            "tilt_mean": self.sum_tilt / steps,
            "forward_speed": dx / alive_t,
            "lateral_drift_rate": dy.abs() / alive_t,
            "lateral_speed": self.sum_abs_vy / steps,
            "yaw_rate": self.sum_abs_yaw / steps,
            "joint_speed": self.sum_joint_speed / steps,
            "power": power,
            "energy_per_meter": energy / travelled,
            "cost_of_transport": energy / (ROBOT_MASS_KG * 9.81 * travelled),
        }
        nan = float("nan")
        result: dict[str, np.ndarray] = {}
        for name, value in out.items():
            axis = AXES[name]
            missing = [c for c in axis.needs if c not in self.supplied]
            arr = torch.nan_to_num(value, nan=nan, posinf=nan, neginf=nan)
            result[name] = (
                np.full(self.num_envs, nan, dtype=np.float32)
                if missing
                else arr.detach().cpu().numpy().astype(np.float32)
            )
        return result


# --------------------------------------------------------------------------- #
# Which two axes an archive is binned on
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DescriptorCfg:
    """The archive's two behaviour axes and the grid range of each.

    The default is walking-v2's per-foot duty factor, so a v1/v2 command line
    that says nothing about the descriptor reproduces exactly what it used to.
    Walking-v3's chosen pair is :data:`V3_DESCRIPTOR`, selected by measurement
    in :mod:`qd.select_descriptor`.
    """

    axis_x: str = "duty_left"
    axis_y: str = "duty_right"
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        for name in (self.axis_x, self.axis_y):
            if name not in AXES:
                raise ValueError(
                    f"unknown descriptor axis {name!r}; known axes: "
                    + ", ".join(AXIS_NAMES)
                )
        for lo, hi in (self.x_range, self.y_range):
            if not hi > lo:
                raise ValueError(f"descriptor range ({lo}, {hi}) is not increasing")

    @property
    def names(self) -> tuple[str, str]:
        return (self.axis_x, self.axis_y)

    @property
    def labels(self) -> tuple[str, str]:
        return (AXES[self.axis_x].label, AXES[self.axis_y].label)

    @property
    def ranges(self) -> list[tuple[float, float]]:
        """As :func:`qd.common.make_archive` wants them."""
        return [tuple(self.x_range), tuple(self.y_range)]

    @property
    def needs(self) -> tuple[str, ...]:
        """Extra harness channels the chosen pair requires."""
        return tuple(
            dict.fromkeys(AXES[self.axis_x].needs + AXES[self.axis_y].needs)
        )

    def measures(self, axes: dict[str, np.ndarray]) -> np.ndarray:
        """``(N, 2)`` clipped into the grid, from :meth:`GaitStats.finalize`.

        Clipped rather than dropped: pyribs would put an out-of-range solution
        nowhere, and a walker just past the edge of the measured range is a
        real gait that belongs in the boundary cell. NaN (an axis whose channel
        was not supplied) becomes the range's lower bound so the archive never
        sees a non-finite measure.
        """
        cols = []
        for name, (lo, hi) in zip(self.names, self.ranges):
            col = np.nan_to_num(np.asarray(axes[name], dtype=np.float64), nan=lo)
            # A hair inside the top edge: pyribs bins on a half-open interval,
            # so a value exactly at the upper bound belongs to no cell. The
            # margin is 1e-7 of the range — a five-thousandth of a bin on a
            # 20-wide grid, and still representable after a float32 round-trip,
            # which a margin of 1e-9 was not.
            cols.append(np.clip(col, lo, hi - 1e-7 * (hi - lo)))
        return np.stack(cols, axis=-1)

    def to_meta(self) -> dict:
        return {
            "descriptor_axes": list(self.names),
            "descriptor_ranges": [list(r) for r in self.ranges],
        }

    @classmethod
    def from_meta(cls, meta: dict | None) -> "DescriptorCfg":
        """Recover the descriptor an archive was built with.

        A v1/v2 archive has no such key and gets the duty-factor default, which
        is what it was actually binned on — so an old archive replays, verifies
        and renders under its own axes rather than under v3's.
        """
        meta = meta or {}
        names = meta.get("descriptor_axes")
        if not names:
            return cls()
        ranges = meta.get("descriptor_ranges") or [
            AXES[n].default_range for n in names
        ]
        return cls(
            axis_x=names[0],
            axis_y=names[1],
            x_range=tuple(ranges[0]),
            y_range=tuple(ranges[1]),
        )


DUTY_FACTOR_DESCRIPTOR = DescriptorCfg()


def grid_indices(
    measures: np.ndarray, dims: tuple[int, int], ranges
) -> np.ndarray:
    """``(N, 2)`` per-axis bin indices, matching ``ribs.archives.GridArchive``.

    Reimplemented rather than asked of pyribs so a *re-measured* descriptor can
    be compared against an archived cell without rebuilding the archive — which
    is how :mod:`qd.verify_archive` asks whether an elite is still in the cell
    it was filed under.
    """
    cols = []
    for k, (lo, hi) in enumerate(ranges):
        idx = np.floor((np.asarray(measures)[:, k] - lo) / (hi - lo) * dims[k])
        cols.append(np.clip(idx, 0, dims[k] - 1).astype(int))
    return np.stack(cols, axis=-1)


def cell_index(measures: np.ndarray, dims: tuple[int, int], ranges) -> np.ndarray:
    """Flat cell index, the integer a checkpoint's ``index`` column holds."""
    grid = grid_indices(measures, dims, ranges)
    return np.ravel_multi_index((grid[:, 0], grid[:, 1]), tuple(dims))
