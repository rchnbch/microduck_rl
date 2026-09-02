"""P2' viability and the mode classifier — v4's replacement for the upright gate.

v3's gate asked *what posture is it in*. That question has one right answer for
a walker and no right answer for anything else: the verified-stable SIT
keyframe sits at trunk z = 0.061 m and prone rest at 0.075 m, so every
non-walking rest pose this robot has is "fallen" to v3, before the first step.

P2' asks a different question — **is it still going, and is it still the same
thing** — and that question has an answer for a crawl and a roll::

    viable  <=>  finite state throughout
             AND +x displacement >= d_min in every window from the second on
             AND one mode label across those same windows
             AND p95 |a_z| under the impact cap   (if the cap discriminates)

Window 1 is exempt because every mode spawns standing at HOME and needs the
first second to *become* the thing it is (a crawl has to get down there).

Why windows and not "moved forward overall": every degenerate v1 policy — the
divers, the face-plant-and-skid, the topple-forward — shares one signature,
**front-loaded progress that then stops**. A whole-episode displacement rule
admits all of them; a per-window rule admits none. And why the label must hold
constant: a windowed progress rule *alone* has a hole at the end of the episode
(walk for 6.5 s, cover the last window's 5 cm early, face-plant at 6.5 s), and
a late fall is exactly what flips ``f_body`` from ~0 to ~1 in the last window.

Three deliberate consequences, stated where they can be read rather than
discovered:

* **Standing still is no longer viable.** An in-place stepping stand scored 0
  in v3 and held a legitimate cell. This is an archive of forward locomotion;
  the zero-command stand is a runtime policy, not a QD mode.
* **A crash-lunge loop is admissible** if each lunge clears ``d_min``, the
  impact cap and a constant "other" label. By the letter of the rule that is a
  locomotion mode. It lands in the "other" sub-archive where a human looks at
  it.
* **The predicate now depends on the classifier**, so a bad threshold is a bad
  viability rule. That is why :mod:`qd.stage_a_prime` measures label stability
  on every probe before either is used, and why nothing here has a default
  threshold that was not put there by that measurement.

Nothing in this module imports mjlab or touches a GPU beyond the tensors it is
handed, so the whole predicate is unit-testable on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

MODES: tuple[str, ...] = ("roll", "hop", "crawl", "walk", "other")
"""Label set, **in classifier precedence order** (§2.4 of the design draft).

``hop`` is in the list and has no seed: Appendix A measured a full-effort
scripted launch at 0.19-0.30 m/s against the 0.44 m/s a 1 cm hop needs, so a
hop sub-archive would be empty by physics rather than by search. The label
stays so that a bounding gait the search stumbles on is *filed* rather than
mislabelled as walk.
"""

MODE_INDEX: dict[str, int] = {m: i for i, m in enumerate(MODES)}
OTHER = MODE_INDEX["other"]


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowCfg:
    """The sliding windows P2' clauses 2 and 3 are evaluated over.

    ``window_seconds`` must be a whole multiple of ``stride_seconds``: that is
    what lets the accumulator keep one running sum per *stride slot* and build
    every window by adding consecutive slots, instead of keeping one
    accumulator per overlapping window and writing to two of them every step.
    """

    episode_seconds: float = 7.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    control_dt: float = 0.02

    def __post_init__(self) -> None:
        if self.stride_seconds <= 0 or self.window_seconds <= 0:
            raise ValueError("window and stride must be positive")
        if self.window_seconds > self.episode_seconds:
            raise ValueError(
                f"window {self.window_seconds}s does not fit in a "
                f"{self.episode_seconds}s episode"
            )
        ratio = self.window_seconds / self.stride_seconds
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"window_seconds ({self.window_seconds}) must be a whole "
                f"multiple of stride_seconds ({self.stride_seconds})"
            )
        n_slots = self.episode_seconds / self.stride_seconds
        if abs(n_slots - round(n_slots)) > 1e-9:
            raise ValueError(
                f"episode_seconds ({self.episode_seconds}) must be a whole "
                f"multiple of stride_seconds ({self.stride_seconds})"
            )
        steps = self.stride_seconds / self.control_dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"stride_seconds ({self.stride_seconds}) must be a whole "
                f"number of {self.control_dt}s control steps"
            )

    @property
    def steps_per_slot(self) -> int:
        return round(self.stride_seconds / self.control_dt)

    @property
    def num_slots(self) -> int:
        return round(self.episode_seconds / self.stride_seconds)

    @property
    def slots_per_window(self) -> int:
        return round(self.window_seconds / self.stride_seconds)

    @property
    def num_windows(self) -> int:
        return self.num_slots - self.slots_per_window + 1

    @property
    def episode_steps(self) -> int:
        return self.num_slots * self.steps_per_slot


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassifierCfg:
    """Thresholds for the mode rules, applied in :data:`MODES` order.

    | order | mode  | rule                                    |
    | ----- | ----- | --------------------------------------- |
    | 1     | roll  | world-horizontal rotation rate >= ``roll_rate_min`` |
    | 2     | hop   | ``f_air`` >= ``hop_air_min``            |
    | 3     | crawl | ``f_body`` >= ``crawl_body_min``        |
    | 4     | walk  | ``f_body`` <= ``walk_body_max`` and ``f_air`` <= ``walk_air_max`` |
    | 5     | other | everything else                         |

    The defaults are the design draft's *initial* values. Stage A' replaces
    them with the midpoint between measured probe clusters, and refuses any
    threshold without at least ``min_margin_sds`` replica standard deviations
    of clearance on both sides — a threshold with no margin is a coin flip
    dressed as a rule, and it would leak the simulator's chaos straight into
    the archive's geography.
    """

    roll_rate_min: float = 0.8
    """rad/s of *supported* net rotation about a world-horizontal axis.

    About one revolution per 7 s. World-horizontal, so a robot spinning on the
    spot is not a roll however fast it spins — see :class:`ModeStats`."""

    hop_air_min: float = 0.10
    crawl_body_min: float = 0.5
    walk_body_max: float = 0.1
    walk_air_max: float = 0.05

    contact_force_n: float = 0.5
    """Force threshold for "this geom is touching" [N].

    Raw ``found`` chatters: the probe logs show single-step foot-contact
    dropouts during ordinary settling, and a dropout counted as flight inflates
    ``f_air`` — the one feature the hop rule reads. A force floor costs nothing
    and removes the chatter. Set to 0 to use ``found`` directly."""

    def label(
        self,
        f_body: np.ndarray,
        f_air: np.ndarray,
        rotation_rate: np.ndarray,
    ) -> np.ndarray:
        """Mode index for each entry, broadcasting over any shape.

        Rules are applied in precedence order, so an over-the-top roll that is
        also mostly on its shell is a roll, not a crawl.
        """
        f_body = np.asarray(f_body, dtype=np.float64)
        f_air = np.asarray(f_air, dtype=np.float64)
        rotation_rate = np.asarray(rotation_rate, dtype=np.float64)
        out = np.full(np.broadcast(f_body, f_air, rotation_rate).shape, OTHER, dtype=np.int64)
        walk = (f_body <= self.walk_body_max) & (f_air <= self.walk_air_max)
        out = np.where(walk, MODE_INDEX["walk"], out)
        out = np.where(f_body >= self.crawl_body_min, MODE_INDEX["crawl"], out)
        out = np.where(f_air >= self.hop_air_min, MODE_INDEX["hop"], out)
        out = np.where(rotation_rate >= self.roll_rate_min, MODE_INDEX["roll"], out)
        return out


# --------------------------------------------------------------------------- #
# Viability (P2')
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ViabilityCfg:
    """The P2' clauses and their thresholds."""

    windows: WindowCfg = field(default_factory=WindowCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)

    d_min: float = 0.05
    """Minimum +x displacement per scored window [m]. 2.5 cm/s at W = 2 s.

    An *initial* value. Stage A' sweeps (W, d_min) and picks the setting that
    maximises the margin between the worst positive probe and the best
    negative, subject to positives >= 0.95 and negatives <= 0.05 per replica.
    If no setting clears both bars, that is the finding — the bars do not
    move."""

    exempt_seconds: float = 1.0
    """Leading transition time excused from clauses 2 and 3.

    Every mode spawns standing at HOME (§4.1), so the start of the episode is
    the stand-to-mode transition: a crawl is still lowering itself, and its
    label is legitimately not yet "crawl".

    Expressed in **seconds**, not in windows, because the Stage A' sweep varies
    the stride: "exempt one window" would excuse the first 1 s at a 1 s stride
    and the first 0.5 s at a 0.5 s stride, so the settings being compared would
    differ in two things at once."""

    def exempt_windows(self, windows: WindowCfg) -> int:
        """How many leading windows ``exempt_seconds`` covers at this stride."""
        return min(
            windows.num_windows,
            max(0, round(self.exempt_seconds / windows.stride_seconds)),
        )

    impact_cap: float | None = None
    """Cap on p95 trunk ``|a_z|`` [m/s^2]; ``None`` disables clause 4.

    A percentile, never a max — a max is a luck-ranked operator, and v3's
    winner's curse applies to any max-like clause in a gate. Stage A' sets the
    cap at 1.5x the worst intended-mode probe and *keeps* it only if it
    excludes >= 90 % of the degenerates that pass clauses 2-3. If nothing
    degenerate passes in the first place, the cap is unnecessary and is
    reported as such rather than added for tidiness."""

    require_constant_label: bool = True
    """Clause 3. Off only for ablations — it is what closes P2's end-of-episode
    hole, where a walker banks the last window's progress early and falls."""


@dataclass
class ModeFeatures:
    """Everything P2' and the classifier read, for one batched rollout.

    Per-window arrays are ``(num_windows, N)``; per-episode arrays are ``(N,)``.
    All numpy, all host-side: this is the boundary where a rollout stops being
    a GPU thing and starts being evidence.
    """

    window_dx: np.ndarray
    window_f_body: np.ndarray
    window_f_air: np.ndarray
    window_f_feet: np.ndarray
    window_f_head: np.ndarray
    window_rotation_rate: np.ndarray

    f_body: np.ndarray
    f_air: np.ndarray
    f_feet: np.ndarray
    f_head: np.ndarray
    f_inverted: np.ndarray
    rotation_rate: np.ndarray
    rotation_rate_pitch: np.ndarray
    rotation_rate_roll: np.ndarray
    rotation_rate_yaw: np.ndarray
    p95_az: np.ndarray
    displacement: np.ndarray
    finite: np.ndarray

    PER_WINDOW: ClassVar[tuple[str, ...]] = (
        "window_dx",
        "window_f_body",
        "window_f_air",
        "window_f_feet",
        "window_f_head",
        "window_rotation_rate",
    )
    PER_EPISODE: ClassVar[tuple[str, ...]] = (
        "f_body",
        "f_air",
        "f_feet",
        "f_head",
        "f_inverted",
        "rotation_rate",
        "rotation_rate_pitch",
        "rotation_rate_roll",
        "rotation_rate_yaw",
        "p95_az",
        "displacement",
        "finite",
    )

    def to_info(self, prefix: str = "mode/") -> dict[str, np.ndarray]:
        """Flatten into the ``{name: (N, ...)}`` shape the harness's ``info`` uses.

        Per-window arrays are transposed to put the env axis first. That is not
        cosmetic: every consumer — chunked evaluation, ``qd.replay.reevaluate``,
        the permuted-replica un-shuffle — concatenates and indexes ``info``
        along axis 0, and a ``(num_windows, N)`` array quietly survives all
        three while meaning something different afterwards.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.PER_WINDOW:
            out[prefix + name] = np.ascontiguousarray(getattr(self, name).T)
        for name in self.PER_EPISODE:
            out[prefix + name] = getattr(self, name)
        return out

    @classmethod
    def from_info(cls, info: dict[str, np.ndarray], prefix: str = "mode/") -> ModeFeatures:
        """Inverse of :meth:`to_info`."""
        kwargs = {n: np.asarray(info[prefix + n]).T for n in cls.PER_WINDOW}
        kwargs.update({n: np.asarray(info[prefix + n]) for n in cls.PER_EPISODE})
        return cls(**kwargs)

    @classmethod
    def present_in(cls, info: dict[str, np.ndarray], prefix: str = "mode/") -> bool:
        return (prefix + cls.PER_EPISODE[0]) in info

    def window_labels(self, cfg: ClassifierCfg) -> np.ndarray:
        """``(num_windows, N)`` mode index per window."""
        return cfg.label(self.window_f_body, self.window_f_air, self.window_rotation_rate)

    def episode_labels(self, cfg: ClassifierCfg) -> np.ndarray:
        """``(N,)`` mode index from the whole-episode features."""
        return cfg.label(self.f_body, self.f_air, self.rotation_rate)


@dataclass
class ViabilityVerdict:
    """Per-env P2' outcome, with every clause kept separately.

    The clause breakdown is not decoration: "feasibility collapsed" and
    "feasibility collapsed *on the label clause*" call for opposite fixes, and
    v2/v3 both spent iterations on the wrong one for want of this split.
    """

    viable: np.ndarray
    finite: np.ndarray
    progress: np.ndarray
    constant_label: np.ndarray
    impact: np.ndarray
    label: np.ndarray
    """Episode mode index (valid whether or not the rollout is viable)."""

    def rates(self) -> dict[str, float]:
        return {
            "viable": float(np.mean(self.viable)),
            "finite": float(np.mean(self.finite)),
            "progress": float(np.mean(self.progress)),
            "constant_label": float(np.mean(self.constant_label)),
            "impact": float(np.mean(self.impact)),
        }


def evaluate_viability(
    features: ModeFeatures, cfg: ViabilityCfg | None = None
) -> ViabilityVerdict:
    """Apply P2' to one rollout's features."""
    cfg = cfg or ViabilityCfg()
    k = cfg.exempt_windows(cfg.windows)
    if features.window_dx.shape[0] != cfg.windows.num_windows:
        raise ValueError(
            f"features carry {features.window_dx.shape[0]} windows but the "
            f"viability cfg describes {cfg.windows.num_windows}; the features "
            "must be accumulated with the same WindowCfg the predicate uses"
        )
    scored_dx = features.window_dx[k:]
    progress = np.all(scored_dx >= cfg.d_min, axis=0)

    labels = features.window_labels(cfg.classifier)
    scored_labels = labels[k:]
    if cfg.require_constant_label and scored_labels.shape[0] > 0:
        constant = np.all(scored_labels == scored_labels[0], axis=0)
    else:
        constant = np.ones(features.f_body.shape, dtype=bool)

    if cfg.impact_cap is None:
        impact_ok = np.ones(features.f_body.shape, dtype=bool)
    else:
        impact_ok = features.p95_az <= cfg.impact_cap

    finite = features.finite.astype(bool)
    viable = finite & progress & constant & impact_ok
    return ViabilityVerdict(
        viable=viable,
        finite=finite,
        progress=progress,
        constant_label=constant,
        impact=impact_ok,
        label=features.episode_labels(cfg.classifier),
    )


def label_agreement(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modal label and its count, over the leading (replica) axis.

    ``labels`` is ``(R, N)``. Returns ``(mode (N,), agreeing_replicas (N,))``,
    which is what the insertion rule's "at least 7 of 8 replicas carry the same
    episode label" clause is checked against. A candidate whose replicas
    disagree is not a robust anything, and rejecting it here is what keeps the
    simulator's chaos out of the archive's *geography* as well as its fitness.
    """
    labels = np.asarray(labels)
    counts = np.stack([(labels == m).sum(axis=0) for m in range(len(MODES))])
    mode = counts.argmax(axis=0)
    return mode, counts.max(axis=0)


# --------------------------------------------------------------------------- #
# Accumulator
# --------------------------------------------------------------------------- #


class VerticalAccel:
    """Finite-difference ``a_z`` of the trunk, held across a rollout.

    The design draft said to read the ``imu_accel`` accelerometer. That is the
    wrong instrument for this clause and the repo already knows it: an
    accelerometer measures *specific force*, so it reads 9.81 m/s^2 on a robot
    lying perfectly still, and the impact cap would charge a resting crawl more
    than a gentle landing. ``trunk_vertical_accel_penalty`` in ``tasks/mdp.py``
    — the term this cap is meant to agree with — differences world-frame
    ``v_z`` instead, which is zero at rest and spikes on an impact. Same
    quantity here, so a policy trained under that penalty is judged by the
    measure it was trained against (AGENTS.md: a tracking reward and its gate
    must measure the same view).
    """

    def __init__(self, num_envs: int, device: str | torch.device, control_dt: float):
        self.control_dt = float(control_dt)
        self.prev_vz = torch.zeros(num_envs, dtype=torch.float32, device=device)

    def begin(self, lin_vel_w: torch.Tensor) -> None:
        self.prev_vz.copy_(torch.nan_to_num(lin_vel_w[:, 2].to(torch.float32)))

    def step(self, lin_vel_w: torch.Tensor) -> torch.Tensor:
        vz = torch.nan_to_num(lin_vel_w[:, 2].to(torch.float32))
        az = (vz - self.prev_vz) / self.control_dt
        self.prev_vz.copy_(vz)
        return az


class ModeStats:
    """Per-slot support/rotation sums and the two traces P2' needs.

    One running sum per *stride slot* rather than per overlapping window: with
    ``window_seconds`` a whole multiple of ``stride_seconds`` (enforced by
    :class:`WindowCfg`) a window is exactly a run of consecutive slots, so the
    per-step cost is one indexed add instead of one per window the step falls
    in.

    Two things are kept as full traces because no running sum can produce them:
    ``|a_z|`` (the impact clause needs a *percentile*, not a mean) and trunk z
    (the spectral axis, §2.3 A6). At 350 steps x ~1k worlds that is a couple of
    megabytes, which is cheaper than the second rollout the alternative costs.

    **Rotation is accumulated in the world frame about horizontal axes, not in
    the base frame.** The design draft asked for the body's lateral axis
    (``root_link_ang_vel_b[:, 1]``, the roulade accumulator's channel), which
    is correct for a somersault by an upright robot and wrong for anything
    lying down: in the side-lying spawn the body's lateral axis points straight
    *up*, so a robot pirouetting on the floor accumulated 1.69 rad/s and
    classified as a **roll** while covering -3 cm. World x and y are the two
    axes a body can roll *along the ground* about; world z is a ground spin,
    which is not locomotion by rolling and is now measured separately
    (``rotation_rate_yaw``) rather than mistaken for it.

    Unlike :class:`qd.descriptors.GaitStats` there is **no fall latch and no
    ``counted`` mask**: under P2' the whole 7 s is the measurement, and the
    thing the latch used to protect against — post-fall frames polluting a
    descriptor — is now exactly the signal clause 3 reads to reject a late
    fall.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        windows: WindowCfg,
        num_contact_geoms: int,
        foot_columns: tuple[int, ...],
        head_columns: tuple[int, ...] = (),
        contact_force_n: float = 0.5,
    ):
        self.num_envs = num_envs
        self.device = device
        self.windows = windows
        self.contact_force_n = float(contact_force_n)
        self.num_contact_geoms = num_contact_geoms

        cols = torch.zeros(num_contact_geoms, dtype=torch.bool, device=device)
        cols[list(foot_columns)] = True
        self._is_foot = cols
        head = torch.zeros(num_contact_geoms, dtype=torch.bool, device=device)
        if head_columns:
            head[list(head_columns)] = True
        self._is_head = head

        n_slots = windows.num_slots
        f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
        self.slot_steps = f(n_slots)
        self.slot_body = f(n_slots, num_envs)
        self.slot_air = f(n_slots, num_envs)
        self.slot_feet = f(n_slots, num_envs)
        self.slot_head = f(n_slots, num_envs)
        self.slot_rot_pitch = f(n_slots, num_envs)
        self.slot_rot_roll = f(n_slots, num_envs)
        self.slot_rot_yaw = f(n_slots, num_envs)
        self.slot_inverted = f(n_slots, num_envs)
        self.slot_x = f(n_slots + 1, num_envs)
        self.az_trace = f(windows.episode_steps, num_envs)
        self.z_trace = f(windows.episode_steps, num_envs)
        self.finite = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._step = 0

    def begin(self, base_pos: torch.Tensor) -> None:
        for t in (
            self.slot_steps,
            self.slot_body,
            self.slot_air,
            self.slot_feet,
            self.slot_head,
            self.slot_rot_pitch,
            self.slot_rot_roll,
            self.slot_rot_yaw,
            self.slot_inverted,
            self.slot_x,
            self.az_trace,
            self.z_trace,
        ):
            t.zero_()
        self.finite.fill_(True)
        self._step = 0
        x = torch.nan_to_num(base_pos[:, 0].to(torch.float32))
        self.slot_x[0].copy_(x)

    def update(
        self,
        base_pos: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        contact_found: torch.Tensor,
        ang_vel_w: torch.Tensor,
        trunk_az: torch.Tensor | None = None,
        contact_force: torch.Tensor | None = None,
    ) -> None:
        """Fold one control step in.

        Args:
            ang_vel_w: ``(N, 3)`` trunk angular velocity in the **world**
                frame. Not the base frame: see the class docstring.
            contact_found: ``(N, G)`` per-geom ground contact, in the column
                order the harness advertises.
            contact_force: ``(N, G, 3)`` or ``(N, G)`` net contact force; used
                with ``contact_force_n`` to de-chatter ``found``.
            trunk_az: ``(N,)`` vertical acceleration of the trunk, from
                :class:`VerticalAccel`. ``None`` leaves the trace at zero,
                which *disables* the impact clause rather than silently passing
                it: 0 is below any cap, and a clause that always passes should
                be reported as absent, not as satisfied.
        """
        w = self.windows
        step = self._step
        if step >= w.episode_steps:
            return
        slot = min(step // w.steps_per_slot, w.num_slots - 1)

        pos = base_pos.to(torch.float32)
        grav = projected_gravity_b.to(torch.float32)
        self.finite &= torch.isfinite(pos).all(dim=-1) & torch.isfinite(grav).all(dim=-1)
        pos = torch.nan_to_num(pos)
        grav = torch.nan_to_num(grav)

        touching = contact_found.to(torch.float32) > 0
        if contact_force is not None and self.contact_force_n > 0:
            force = torch.nan_to_num(contact_force.to(torch.float32))
            if force.dim() == 3:
                force = torch.linalg.vector_norm(force, dim=-1)
            touching = touching & (force >= self.contact_force_n)

        body_touch = (touching & ~self._is_foot).any(dim=-1)
        any_touch = touching.any(dim=-1)
        head_touch = (touching & self._is_head).any(dim=-1)

        self.slot_steps[slot] += 1.0
        self.slot_body[slot] += body_touch.float()
        self.slot_air[slot] += (~any_touch).float()
        self.slot_feet[slot] += (any_touch & ~body_touch).float()
        self.slot_head[slot] += head_touch.float()
        self.slot_inverted[slot] += (-grav[:, 2] < 0).float()

        # Supported rotation only: a ballistic spin is not locomotion, which is
        # the same gate the roulade env's accumulator uses. World frame, and
        # horizontal axes only: see the class docstring for the probe that
        # forced this.
        omega = torch.nan_to_num(ang_vel_w.to(torch.float32))
        supported = any_touch.float()
        dt = w.control_dt
        self.slot_rot_pitch[slot] += omega[:, 1] * dt * supported
        self.slot_rot_roll[slot] += omega[:, 0] * dt * supported
        self.slot_rot_yaw[slot] += omega[:, 2] * dt * supported

        self.slot_x[slot + 1] = torch.nan_to_num(pos[:, 0])
        self.z_trace[step] = torch.nan_to_num(pos[:, 2])
        if trunk_az is not None:
            self.az_trace[step] = torch.nan_to_num(trunk_az.to(torch.float32)).abs()
        self._step += 1

    def finalize(self) -> ModeFeatures:
        w = self.windows
        spw = w.slots_per_window
        steps = self.slot_steps.clamp_min(1.0).unsqueeze(-1)

        def windowed(slot_sum: torch.Tensor) -> torch.Tensor:
            """``(num_windows, N)`` fraction of steps, over runs of ``spw`` slots."""
            num = torch.stack(
                [slot_sum[k : k + spw].sum(dim=0) for k in range(w.num_windows)]
            )
            den = torch.stack(
                [steps[k : k + spw].sum(dim=0) for k in range(w.num_windows)]
            )
            return num / den.clamp_min(1.0)

        window_dx = torch.stack(
            [self.slot_x[k + spw] - self.slot_x[k] for k in range(w.num_windows)]
        )
        rot_pitch_w = torch.stack(
            [self.slot_rot_pitch[k : k + spw].sum(dim=0) for k in range(w.num_windows)]
        )
        rot_roll_w = torch.stack(
            [self.slot_rot_roll[k : k + spw].sum(dim=0) for k in range(w.num_windows)]
        )
        window_rot = torch.maximum(rot_pitch_w.abs(), rot_roll_w.abs()) / w.window_seconds

        total_steps = self.slot_steps.sum().clamp_min(1.0)
        ep = lambda slot_sum: slot_sum.sum(dim=0) / total_steps
        rot_pitch = self.slot_rot_pitch.sum(dim=0).abs() / w.episode_seconds
        rot_roll = self.slot_rot_roll.sum(dim=0).abs() / w.episode_seconds
        rot_yaw = self.slot_rot_yaw.sum(dim=0).abs() / w.episode_seconds

        used = max(self._step, 1)
        p95 = torch.quantile(self.az_trace[:used], 0.95, dim=0)

        n = lambda t: t.detach().cpu().numpy().astype(np.float32)
        return ModeFeatures(
            window_dx=n(window_dx),
            window_f_body=n(windowed(self.slot_body)),
            window_f_air=n(windowed(self.slot_air)),
            window_f_feet=n(windowed(self.slot_feet)),
            window_f_head=n(windowed(self.slot_head)),
            window_rotation_rate=n(window_rot),
            f_body=n(ep(self.slot_body)),
            f_air=n(ep(self.slot_air)),
            f_feet=n(ep(self.slot_feet)),
            f_head=n(ep(self.slot_head)),
            f_inverted=n(ep(self.slot_inverted)),
            rotation_rate=n(torch.maximum(rot_pitch, rot_roll)),
            rotation_rate_pitch=n(rot_pitch),
            rotation_rate_roll=n(rot_roll),
            rotation_rate_yaw=n(rot_yaw),
            p95_az=n(p95),
            displacement=n(self.slot_x[w.num_slots] - self.slot_x[0]),
            finite=self.finite.detach().cpu().numpy(),
        )


def dominant_frequency(trace: np.ndarray, control_dt: float) -> np.ndarray:
    """Dominant non-DC frequency of a ``(T, N)`` trace [Hz] — descriptor axis A6.

    The one candidate axis that says *how* a mode repeats rather than what it
    looks like on average. Kept here rather than in :mod:`qd.descriptors`
    because it is the only axis that needs a trajectory buffer instead of a
    running sum.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim == 1:
        trace = trace[:, None]
    centred = trace - trace.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=0))
    spectrum[0] = 0.0
    freqs = np.fft.rfftfreq(trace.shape[0], d=control_dt)
    return freqs[spectrum.argmax(axis=0)]


def percentile_over_replicas(values: np.ndarray, q: float) -> np.ndarray:
    """Median-style aggregation helper kept next to the clause that uses it."""
    return np.percentile(np.asarray(values), q, axis=0)


def rotation_period_edges(rate_max: float, control_dt: float = 0.0) -> np.ndarray:
    """Bin edges aligned to whole revolutions, for a within-roll rate axis.

    The *count* of completed rolls is discrete, so a policy sitting between two
    and three rolls per episode straddles any evenly-spaced bin. Edges placed
    at whole revolutions put the discreteness on a bin boundary instead of in
    the middle of one.
    """
    del control_dt
    n = max(1, math.floor(rate_max / (2.0 * math.pi) * 7.0) + 1)
    return np.array([2.0 * math.pi * k / 7.0 for k in range(n + 1)])
