"""Pieces shared by the vanilla-CPG (Phase 2) and PGA-ME (Phase 3) pipelines.

Nothing in here touches a GPU or imports mjlab, so it is unit-testable on CPU:

* :class:`FitnessCfg` / :class:`RolloutMetrics` — the objective (forward +x
  displacement with a fall penalty) and the 2-D behaviour descriptor
  (per-foot duty factor), accumulated over a batched rollout.
* archive construction, checkpointing and heatmap plotting, so both pipelines
  produce directly comparable artefacts.

The metrics accumulator deliberately takes plain tensors rather than an env, so
Phase 3 can feed it from a ``ManagerBasedRlEnv`` while Phase 2 feeds it from the
low-level ``Scene``/``Simulation`` harness in :mod:`qd.evaluate`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Objective and behaviour descriptor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FitnessCfg:
    """Objective + fall-detection thresholds.

    ``robot_walk.xml`` has the trunk/head collision geoms stripped (only the
    feet collide), so a fall is *not* observable as a trunk-ground contact.  It
    is detected from the base state instead: the trunk dropping below
    ``fall_height`` or tilting past ``fall_tilt_deg``.
    """

    episode_seconds: float = 7.0
    """Length of the scored rollout (excludes the settle phase)."""

    settle_seconds: float = 0.25
    """Held at HOME before the CPG starts, so the spawn drop is not scored.

    Long enough to land (both feet contact by ~0.05 s) and short enough that the
    passive forward droop has barely begun — ``qd.check_harness`` measures ~3 deg
    of tilt here, against ~6 deg by 0.4 s."""

    fall_height: float = 0.075
    """Base (trunk) height below which the robot counts as fallen [m].

    Measured, not assumed (AGENTS.md: never carry a target height across model
    revisions): at HOME the lowest foot vertex sits 0.117 m below the trunk
    frame, and ``qd.check_harness`` reports the settled standing height. 0.075 m
    is a collapse a walking crouch cannot reach, and leaves the tilt check to
    catch the topples that keep their height."""

    fall_tilt_deg: float = 60.0
    """Tilt of the base +z axis from world +z past which the robot has flipped."""

    fall_penalty: float = 0.25
    """Peak cost of falling, charged pro-rata for the time spent down [m].

    ``fitness = displacement_at_fall - fall_penalty * fraction_of_episode_fallen``

    Not a flat penalty, because on this robot a passive HOME hold topples in
    ~1.4 s (see ``qd.check_harness``): with no control *everything* falls, so a
    constant penalty is a constant offset and the search collapses to "who
    dives forward fastest". Charging by time-down instead makes falling late
    strictly cheaper than falling early, so a gait that survives beats a
    ballistic dive that covers the same ground and lies there. It is also
    rate-limited by construction — there is no state you can reach early and
    then farm (AGENTS.md, "No jackpots")."""

    min_fitness: float = -5.0
    """Floor applied to the objective; also catches non-finite sims."""

    @property
    def fall_tilt_cos(self) -> float:
        """``cos(fall_tilt_deg)`` — compared against ``-projected_gravity_b.z``."""
        return float(np.cos(np.deg2rad(self.fall_tilt_deg)))


class RolloutMetrics:
    """Per-env accumulator turning a batched rollout into (fitness, measures).

    Usage per rollout::

        m = RolloutMetrics(num_envs, cfg, device)
        m.begin(base_pos)
        for each control step:
            m.update(base_pos, projected_gravity_b, foot_contact)
        fitness, measures, info = m.finalize()

    A fall is *latched*: once an env falls, its displacement is frozen at the
    value it had when the fall was detected and it stops contributing to the
    duty-factor average.  Otherwise a robot that face-plants and then slides
    would keep accruing both objective and descriptor.
    """

    def __init__(self, num_envs: int, cfg: FitnessCfg, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        z = lambda dtype: torch.zeros(num_envs, dtype=dtype, device=device)
        self.start_x = z(torch.float32)
        self.frozen_x = z(torch.float32)
        self.fallen = z(torch.bool)
        self.alive_steps = z(torch.long)
        self.contact_steps = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self.total_steps = 0

    def begin(self, base_pos: torch.Tensor) -> None:
        """Latch the start position; call once, after the settle phase."""
        x = base_pos[:, 0].to(torch.float32)
        self.start_x.copy_(torch.nan_to_num(x))
        self.frozen_x.copy_(self.start_x)
        self.fallen.zero_()
        self.alive_steps.zero_()
        self.contact_steps.zero_()
        self.total_steps = 0

    def update(
        self,
        base_pos: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        foot_contact: torch.Tensor,
    ) -> None:
        """Fold one control step in.

        Args:
            base_pos: ``(N, 3)`` trunk position in world frame.
            projected_gravity_b: ``(N, 3)`` gravity direction in the base frame.
                Upright is ``(0, 0, -1)``.
            foot_contact: ``(N, 2)`` boolean/float left- and right-foot ground
                contact for this step.
        """
        cfg = self.cfg
        finite = (
            torch.isfinite(base_pos).all(dim=-1)
            & torch.isfinite(projected_gravity_b).all(dim=-1)
        )
        upright = -projected_gravity_b[:, 2]  # 1 upright, 0 on its side, -1 upside down
        too_low = base_pos[:, 2] < cfg.fall_height
        too_tilted = upright < cfg.fall_tilt_cos
        fall_now = (~finite) | too_low | too_tilted

        alive_before = ~self.fallen
        self.fallen |= alive_before & fall_now

        # The step on which the fall is detected still counts, so an env that
        # fails immediately has a non-zero duty-factor denominator.
        counted = alive_before
        self.alive_steps += counted.long()
        self.contact_steps += (foot_contact.bool() & counted.unsqueeze(-1)).long()
        self.frozen_x = torch.where(
            counted, torch.nan_to_num(base_pos[:, 0].to(torch.float32)), self.frozen_x
        )
        self.total_steps += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return ``(fitness (N,), measures (N, 2), info)`` as numpy arrays."""
        cfg = self.cfg
        displacement = self.frozen_x - self.start_x

        total = max(self.total_steps, 1)
        fallen_fraction = 1.0 - self.alive_steps.float() / total
        fitness = displacement - cfg.fall_penalty * fallen_fraction
        fitness = torch.nan_to_num(fitness, nan=cfg.min_fitness).clamp_min(cfg.min_fitness)

        denom = self.alive_steps.clamp_min(1).unsqueeze(-1).float()
        measures = (self.contact_steps.float() / denom).clamp(0.0, 1.0)

        info = {
            "displacement": displacement.cpu().numpy(),
            "fell": self.fallen.cpu().numpy(),
            "alive_steps": self.alive_steps.cpu().numpy(),
            "survival_fraction": (self.alive_steps.float() / total).cpu().numpy(),
        }
        return fitness.cpu().numpy(), measures.cpu().numpy(), info


# --------------------------------------------------------------------------- #
# Archive helpers
# --------------------------------------------------------------------------- #

DEFAULT_GRID_DIMS: tuple[int, int] = (20, 20)
DEFAULT_MEASURE_RANGES: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0)]
MEASURE_NAMES: tuple[str, str] = ("left_foot_duty_factor", "right_foot_duty_factor")


def make_archive(
    solution_dim: int,
    grid_dims: tuple[int, int] = DEFAULT_GRID_DIMS,
    measure_ranges: list[tuple[float, float]] | None = None,
    qd_score_offset: float = -1.0,
    seed: int | None = None,
):
    """A :class:`ribs.archives.GridArchive` over the duty-factor descriptor.

    ``qd_score_offset`` must sit at or below the worst objective that can be
    inserted, otherwise pyribs' QD-score can go down when a (still valid)
    negative-fitness elite is added.  The default matches the range produced by
    :class:`FitnessCfg` for a robot that falls immediately.
    """
    from ribs.archives import GridArchive

    return GridArchive(
        solution_dim=solution_dim,
        dims=list(grid_dims),
        ranges=measure_ranges or DEFAULT_MEASURE_RANGES,
        qd_score_offset=qd_score_offset,
        seed=seed,
    )


def archive_stats(archive) -> dict[str, float]:
    """Coverage / QD-score / best-cell fitness as plain floats."""
    stats = archive.stats
    return {
        "num_elites": float(stats.num_elites),
        "coverage": float(stats.coverage),
        "qd_score": float(stats.qd_score),
        "obj_max": float(stats.obj_max) if stats.obj_max is not None else float("nan"),
        "obj_mean": float(stats.obj_mean) if stats.obj_mean is not None else float("nan"),
    }


def save_archive(archive, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    """Checkpoint every elite (genome + fitness + descriptor + cell index).

    Written as a single ``.npz`` so a run can be resumed, replayed or compared
    without re-deriving anything from the pyribs object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = archive.data(return_type="dict")
    payload = {
        "solution": np.asarray(data["solution"]),
        "objective": np.asarray(data["objective"]),
        "measures": np.asarray(data["measures"]),
        "index": np.asarray(data["index"]),
        "grid_dims": np.asarray(archive.dims),
        "measure_ranges": np.asarray(
            list(zip(archive.lower_bounds, archive.upper_bounds))
        ),
        "meta_json": np.array(json.dumps(meta or {})),
    }
    np.savez_compressed(path, **payload)
    return path


def load_archive(path: str | Path) -> dict[str, Any]:
    """Inverse of :func:`save_archive`; ``meta`` comes back as a dict."""
    with np.load(Path(path), allow_pickle=False) as f:
        out = {k: f[k] for k in f.files if k != "meta_json"}
        out["meta"] = json.loads(str(f["meta_json"]))
    return out


def plot_archive(
    archive,
    path: str | Path,
    title: str = "MAP-Elites archive",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Save a duty-factor heatmap of the archive."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ribs.visualize import grid_archive_heatmap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=140)
    grid_archive_heatmap(archive, ax=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xlabel(MEASURE_NAMES[0].replace("_", " "))
    ax.set_ylabel(MEASURE_NAMES[1].replace("_", " "))
    stats = archive_stats(archive)
    ax.set_title(
        f"{title}\ncoverage {stats['coverage'] * 100:.1f}%  "
        f"QD-score {stats['qd_score']:.1f}  best {stats['obj_max']:.3f} m"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return str(o)

    path.write_text(json.dumps(payload, indent=2, default=default))
    return path
