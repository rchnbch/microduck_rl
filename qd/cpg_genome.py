"""Open-loop CPG genome for MAP-Elites gait discovery.

Genome
------
Ten leg joints (5 per leg, neck/head are pinned at HOME) each follow a sinusoid
that shares one global frequency::

    target_j(t) = offset_j + amplitude_j * sin(2*pi*freq*t + phase_j)

Parameter vector layout — 31 params, **blocked** (not interleaved) so the whole
batch evaluates as three contiguous slices::

    [ freq | amplitude_0..amplitude_9 | phase_0..phase_9 | offset_0..offset_9 ]

The joint order is :data:`LEG_JOINT_NAMES` (left leg 0-4, right leg 5-9), which
is the *servo* order the repo uses everywhere.  It is resolved by NAME against
the compiled model at rollout time (``qd/evaluate.py``), never by hard-coded
qpos slicing — the walk model has 14 servo joints but the backlash/roller
models interleave ``passive_*`` joints.

Bounds
------
* ``freq``      : :data:`FREQ_BOUNDS` Hz
* ``amplitude`` : ``[0, AMPLITUDE_RANGE_FRACTION * (hi - lo)]`` per joint
* ``phase``     : ``[0, 2*pi]``
* ``offset``    : the joint's *soft* limits (MJCF range shrunk about its
  midpoint by ``SOFT_LIMIT_FACTOR``, matching
  ``EntityArticulationInfoCfg.soft_joint_pos_limit_factor``)

``offset + amplitude`` can still leave the soft limits, which is deliberate —
the sinusoid is allowed to saturate against a limit the way a real gait does.
The saturation is made explicit by :func:`clip_targets`, which the evaluator
calls before every write to the sim.  Bounds are therefore enforced *twice*:
once by the emitter (per-dimension ``bounds=``) and once here, defensively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

# Servo order for the two legs: left 0-4, then right 5-9. Neck/head (servo
# indices 5-8 in the full 14-joint layout) are excluded and held at HOME.
LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

NUM_LEG_JOINTS = len(LEG_JOINT_NAMES)
GENOME_DIM = 1 + 3 * NUM_LEG_JOINTS  # 31

# Slices into the genome vector.
FREQ_SLICE = slice(0, 1)
AMP_SLICE = slice(1, 1 + NUM_LEG_JOINTS)
PHASE_SLICE = slice(1 + NUM_LEG_JOINTS, 1 + 2 * NUM_LEG_JOINTS)
OFFSET_SLICE = slice(1 + 2 * NUM_LEG_JOINTS, GENOME_DIM)

FREQ_BOUNDS: tuple[float, float] = (0.5, 3.0)
PHASE_BOUNDS: tuple[float, float] = (0.0, 2.0 * math.pi)

# Max half-swing as a fraction of the joint's full MJCF range. 0.25 gives a
# peak-to-peak sweep of up to half the joint range, which is ~45 deg on the
# pitch/knee joints and ~11 deg on hip_roll — plenty for a 25 cm biped without
# handing the emitter a search space that is mostly limit-saturated junk.
AMPLITUDE_RANGE_FRACTION: float = 0.25

# Mirrors EntityArticulationInfoCfg.soft_joint_pos_limit_factor on every
# microduck robot cfg.
SOFT_LIMIT_FACTOR: float = 0.9


@dataclass(frozen=True)
class GenomeSpace:
    """Per-dimension bounds and the derived MAP-Elites mutation sigma."""

    lower: np.ndarray  # (GENOME_DIM,)
    upper: np.ndarray  # (GENOME_DIM,)

    @property
    def dim(self) -> int:
        return int(self.lower.shape[0])

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Per-dimension ``(lo, hi)`` list, the form pyribs emitters want."""
        return [(float(lo), float(hi)) for lo, hi in zip(self.lower, self.upper)]

    def sigma(self, fraction: float = 0.1) -> np.ndarray:
        """Isotropic-per-dimension Gaussian sigma = ``fraction`` of each range."""
        return fraction * (self.upper - self.lower)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` uniform genomes inside the bounds."""
        return rng.uniform(self.lower, self.upper, size=(n, self.dim))

    def clip(self, genomes: np.ndarray) -> np.ndarray:
        return np.clip(genomes, self.lower, self.upper)


@lru_cache(maxsize=1)
def leg_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` MJCF ranges of :data:`LEG_JOINT_NAMES`, in that order.

    Compiles ``robot_walk.xml`` on the CPU — cheap, and keeps the genome bounds
    tied to the model instead of to a copied constant that silently rots when
    the MJCF is re-exported from Onshape.
    """
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    lo = np.empty(NUM_LEG_JOINTS)
    hi = np.empty(NUM_LEG_JOINTS)
    for i, name in enumerate(LEG_JOINT_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise KeyError(f"joint {name!r} not found in {MICRODUCK_WALK_XML}")
        lo[i], hi[i] = model.jnt_range[jnt_id]
    return lo, hi


def soft_leg_joint_limits(
    factor: float = SOFT_LIMIT_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """MJCF ranges shrunk about their midpoint by ``factor``."""
    lo, hi = leg_joint_limits()
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    return mid - half, mid + half


@lru_cache(maxsize=8)
def joint_limits(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` MJCF ranges of any named joints, in the order given.

    The v4 probes drive all fourteen servos, not the ten the CPG genome covers,
    so the clamp needs limits for the neck too."""
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    lo = np.empty(len(names))
    hi = np.empty(len(names))
    for i, name in enumerate(names):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            raise KeyError(f"joint {name!r} not found in {MICRODUCK_WALK_XML}")
        lo[i], hi[i] = model.jnt_range[jnt_id]
    return lo, hi


def soft_joint_limits(
    names, factor: float = SOFT_LIMIT_FACTOR
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`joint_limits` shrunk about the midpoint, as the entity cfg does."""
    lo, hi = joint_limits(tuple(names))
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def genome_space(
    freq_bounds: tuple[float, float] = FREQ_BOUNDS,
    amplitude_range_fraction: float = AMPLITUDE_RANGE_FRACTION,
    soft_limit_factor: float = SOFT_LIMIT_FACTOR,
) -> GenomeSpace:
    """Build the 31-dimensional search space from the model's joint limits."""
    raw_lo, raw_hi = leg_joint_limits()
    soft_lo, soft_hi = soft_leg_joint_limits(soft_limit_factor)

    lower = np.empty(GENOME_DIM)
    upper = np.empty(GENOME_DIM)

    lower[FREQ_SLICE] = freq_bounds[0]
    upper[FREQ_SLICE] = freq_bounds[1]

    lower[AMP_SLICE] = 0.0
    upper[AMP_SLICE] = amplitude_range_fraction * (raw_hi - raw_lo)

    lower[PHASE_SLICE] = PHASE_BOUNDS[0]
    upper[PHASE_SLICE] = PHASE_BOUNDS[1]

    lower[OFFSET_SLICE] = soft_lo
    upper[OFFSET_SLICE] = soft_hi

    return GenomeSpace(lower=lower, upper=upper)


def unpack(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a ``(B, 31)`` batch into ``(freq, amplitude, phase, offset)``.

    ``freq`` is ``(B, 1)``; the rest are ``(B, 10)``.
    """
    genomes = np.atleast_2d(np.asarray(genomes, dtype=np.float64))
    if genomes.shape[-1] != GENOME_DIM:
        raise ValueError(
            f"expected genomes with {GENOME_DIM} params, got {genomes.shape[-1]}"
        )
    return (
        genomes[:, FREQ_SLICE],
        genomes[:, AMP_SLICE],
        genomes[:, PHASE_SLICE],
        genomes[:, OFFSET_SLICE],
    )


def cpg_targets(genomes: np.ndarray, t: float) -> np.ndarray:
    """Leg-joint targets at time ``t`` for a whole genome batch.

    Args:
        genomes: ``(B, 31)`` parameter vectors.
        t: seconds since the CPG started (shared clock across the batch).

    Returns:
        ``(B, 10)`` joint-position targets in :data:`LEG_JOINT_NAMES` order,
        **unclipped** — call :func:`clip_targets` before writing to the sim.
    """
    freq, amp, phase, offset = unpack(genomes)
    return offset + amp * np.sin(2.0 * np.pi * freq * t + phase)


def clip_targets(
    targets: np.ndarray, soft_limit_factor: float = SOFT_LIMIT_FACTOR
) -> np.ndarray:
    """Defensive saturation of leg-joint targets against the soft limits.

    The second of the two bound enforcements (the emitter's per-dimension
    ``bounds=`` is the first).  Without it an out-of-range offset is silently
    absorbed by the actuator model and the archive fills with junk that cannot
    be reproduced on hardware.
    """
    lo, hi = soft_leg_joint_limits(soft_limit_factor)
    return np.clip(targets, lo, hi)
