"""The Stage A' probe set: hand-built behaviours with known intended labels.

v3's rule was that descriptors chosen by taste fail and descriptors chosen by
spread-to-noise on *known-distinct* probes work. For modes nobody has produced
yet there are no known-distinct gaits to measure, so they are built by hand and
the same discipline runs on them.

Each probe is an open-loop joint-target trajectory with an intended mode label
and a spawn pose. Two kinds:

* **positives** — a scripted crawl, a log-roll, a tuck-and-flop. If a positive
  cannot pass P2' at any (W, d_min), that is the physics answer for that mode,
  reported as such. It is not a reason to lower a bar.
* **negatives** — a dive, a twitcher, a passive HOME hold. Every one of them
  must fail P2'. Any that passes is reported *by name*; the threshold is not
  patched to exclude it, because a predicate tuned to reject the negatives it
  was shown is a predicate that has learned the probe set rather than the
  distinction.

Open-loop probes spawn in a rest pose (see :mod:`qd.spawn` for the measurements
that force this) and are therefore **measurement instruments, never archive
candidates**: the archive spawns standing, and nothing open-loop survives a
standing start on this robot.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class Sinusoid:
    """One joint's contribution: ``offset + amp * sin(2 pi f t + phase)``.

    ``offset`` is relative to the spawn pose's angle for that joint, so a probe
    reads as "what this joint does on top of lying there" rather than as a set
    of absolute angles that stop meaning anything if HOME moves.
    """

    amp: float = 0.0
    phase: float = 0.0
    offset: float = 0.0
    freq: float | None = None
    """Hz; ``None`` uses the probe's global frequency."""


@dataclass(frozen=True)
class Probe:
    """A scripted open-loop behaviour with an intended mode label."""

    name: str
    mode: str
    """The label this probe is *meant* to carry. Stage A' checks, never assumes."""

    spawn: str
    freq: float = 1.0
    joints: dict[str, Sinusoid] = field(default_factory=dict)
    ramp_seconds: float = 0.3
    """Blend from the spawn pose into the trajectory, so step 1 is not a snap."""

    positive: bool = True
    """False for a probe that P2' must reject."""

    note: str = ""

    def targets(
        self, home: torch.Tensor, joint_names: tuple[str, ...], t: float
    ) -> torch.Tensor:
        """``(N, 14)`` servo targets at time ``t``, on top of the spawn pose."""
        delta = torch.zeros_like(home)
        for j, name in enumerate(joint_names):
            s = self.joints.get(name)
            if s is None:
                continue
            f = self.freq if s.freq is None else s.freq
            delta[:, j] = s.offset + s.amp * math.sin(2.0 * math.pi * f * t + s.phase)
        if self.ramp_seconds > 0:
            delta = delta * min(1.0, t / self.ramp_seconds)
        return home + delta


def variants(base: Probe, **grids) -> list[Probe]:
    """Cartesian product of overrides on a base probe, named by their offsets.

    Used to sweep a scripted mode rather than hand-tune it: ``freq`` and a
    global ``sign`` (which negates every phase, reversing the stroke) are the
    two knobs that decide whether a drag pushes forward or backward, and
    guessing them costs a rollout each while sweeping them costs one.
    """
    import itertools

    keys = list(grids)
    out = []
    for combo in itertools.product(*(grids[k] for k in keys)):
        params = dict(zip(keys, combo))
        joints = dict(base.joints)
        freq = params.get("freq", base.freq)
        if params.get("sign", 1) < 0:
            joints = {
                n: dataclasses.replace(s, phase=-s.phase, amp=-s.amp)
                for n, s in joints.items()
            }
        scale = params.get("amp_scale", 1.0)
        if scale != 1.0:
            joints = {
                n: dataclasses.replace(s, amp=s.amp * scale) for n, s in joints.items()
            }
        suffix = "_".join(f"{k}{params[k]:g}" for k in keys)
        out.append(
            dataclasses.replace(
                base, name=f"{base.name}|{suffix}", freq=freq, joints=joints
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Positives
# --------------------------------------------------------------------------- #

HIP_L, HIP_R = "left_hip_pitch", "right_hip_pitch"
KNEE_L, KNEE_R = "left_knee", "right_knee"
ROLL_L, ROLL_R = "left_hip_roll", "right_hip_roll"
YAW_L, YAW_R = "left_hip_yaw", "right_hip_yaw"
ANKLE_L, ANKLE_R = "left_ankle", "right_ankle"
NECK, HEAD_P, HEAD_Y, HEAD_R = "neck_pitch", "head_pitch", "head_yaw", "head_roll"

PI = math.pi

_CRAWL_NOTE = (
    "Prone on the battery pack. Hip-pitch torque ~0.4 N.m over a 5-9 cm lever "
    "gives 4-8 N per leg against 7.2 N of weight, so even a 100%-on-trunk drag "
    "at mu = 1 is pushable; the uncertainty is style, not existence."
)

PROBES: tuple[Probe, ...] = (
    # -- crawl ------------------------------------------------------------- #
    Probe(
        "crawl_belly_push",
        mode="crawl",
        spawn="prone",
        freq=1.2,
        note=_CRAWL_NOTE + " Legs in antiphase: a diagonal drag.",
        joints={
            HIP_L: Sinusoid(amp=0.55, phase=0.0),
            HIP_R: Sinusoid(amp=0.55, phase=PI),
            KNEE_L: Sinusoid(amp=0.7, phase=PI / 2),
            KNEE_R: Sinusoid(amp=0.7, phase=3 * PI / 2),
            ANKLE_L: Sinusoid(amp=0.3, phase=PI / 2),
            ANKLE_R: Sinusoid(amp=0.3, phase=3 * PI / 2),
        },
    ),
    Probe(
        "crawl_belly_push_inphase",
        mode="crawl",
        spawn="prone",
        freq=1.0,
        note=_CRAWL_NOTE + " Both legs together: a breaststroke drag.",
        joints={
            HIP_L: Sinusoid(amp=0.6, phase=0.0),
            HIP_R: Sinusoid(amp=0.6, phase=0.0),
            KNEE_L: Sinusoid(amp=0.8, phase=PI / 2),
            KNEE_R: Sinusoid(amp=0.8, phase=PI / 2),
            ANKLE_L: Sinusoid(amp=0.35, phase=PI / 2),
            ANKLE_R: Sinusoid(amp=0.35, phase=PI / 2),
        },
    ),
    Probe(
        "crawl_knee",
        mode="crawl",
        spawn="prone",
        freq=1.4,
        note=(
            "Knees flexed under the body so the thighs, not the belly, carry "
            "the load — the pose the new upper-leg ground geoms exist for. "
            "Hip roll is only +-22 deg, so the recovery stroke has to come "
            "from knee flexion."
        ),
        joints={
            HIP_L: Sinusoid(amp=0.4, phase=0.0, offset=-0.5),
            HIP_R: Sinusoid(amp=0.4, phase=PI, offset=0.5),
            KNEE_L: Sinusoid(amp=0.5, phase=PI / 2, offset=0.9),
            KNEE_R: Sinusoid(amp=0.5, phase=3 * PI / 2, offset=-0.9),
            ROLL_L: Sinusoid(amp=0.15, phase=0.0),
            ROLL_R: Sinusoid(amp=0.15, phase=PI),
        },
    ),
    Probe(
        "crawl_chin_drag",
        mode="crawl",
        spawn="prone",
        freq=1.1,
        note=(
            "The head is 38% of body mass; a chin-drag crawl is a real option "
            "on this robot and it is what makes head involvement worth "
            "measuring separately from f_body."
        ),
        joints={
            HIP_L: Sinusoid(amp=0.5, phase=0.0),
            HIP_R: Sinusoid(amp=0.5, phase=PI),
            KNEE_L: Sinusoid(amp=0.6, phase=PI / 2),
            KNEE_R: Sinusoid(amp=0.6, phase=3 * PI / 2),
            NECK: Sinusoid(amp=0.3, phase=0.0, offset=-0.35),
            HEAD_P: Sinusoid(amp=0.3, phase=PI, offset=-0.35),
        },
    ),
    # -- roll -------------------------------------------------------------- #
    Probe(
        "log_roll",
        mode="roll",
        spawn="side",
        freq=0.6,
        note=(
            "Lying on the side, rotating about the long axis, moving +x. "
            "Mechanically the simpler roll on a body this shape — 0.28 m per "
            "turn on a ~9 cm cross-section — and nobody has tried it. Hip roll "
            "and head yaw in quadrature is the drive."
        ),
        joints={
            ROLL_L: Sinusoid(amp=0.38, phase=0.0),
            ROLL_R: Sinusoid(amp=0.38, phase=0.0),
            YAW_L: Sinusoid(amp=0.35, phase=PI / 2),
            YAW_R: Sinusoid(amp=0.35, phase=PI / 2),
            HEAD_Y: Sinusoid(amp=0.9, phase=PI / 2),
            HEAD_R: Sinusoid(amp=0.7, phase=0.0),
        },
    ),
    Probe(
        "log_roll_fast",
        mode="roll",
        spawn="side",
        freq=1.1,
        note="The same drive at ~2 revolutions per episode, to put a probe on "
        "each side of the roll rule's one-revolution threshold.",
        joints={
            ROLL_L: Sinusoid(amp=0.38, phase=0.0),
            ROLL_R: Sinusoid(amp=0.38, phase=0.0),
            YAW_L: Sinusoid(amp=0.35, phase=PI / 2),
            YAW_R: Sinusoid(amp=0.35, phase=PI / 2),
            HEAD_Y: Sinusoid(amp=0.9, phase=PI / 2),
            HEAD_R: Sinusoid(amp=0.7, phase=0.0),
        },
    ),
    Probe(
        "tuck_and_flop",
        mode="roll",
        spawn="sit",
        freq=0.45,
        note=(
            "Chin tucked, knees to the chest, then a hip-pitch throw: the "
            "over-the-head roll the roulade env spent five runs learning to "
            "start. Open-loop it will not chain, which is the point of "
            "measuring it."
        ),
        joints={
            NECK: Sinusoid(amp=0.4, offset=-0.9),
            HEAD_P: Sinusoid(amp=0.4, offset=0.9),
            HIP_L: Sinusoid(amp=0.9, phase=0.0, offset=-0.6),
            HIP_R: Sinusoid(amp=0.9, phase=0.0, offset=0.6),
            KNEE_L: Sinusoid(amp=0.6, phase=PI / 2, offset=1.0),
            KNEE_R: Sinusoid(amp=0.6, phase=PI / 2, offset=-1.0),
        },
    ),
    # -- negatives --------------------------------------------------------- #
    Probe(
        "dive",
        mode="crawl",
        spawn="stand",
        positive=False,
        freq=0.14,  # a single slow half-cycle over the episode: lean, push, lie
        ramp_seconds=0.0,
        note=(
            "Lean forward, push, lie still. v1's archive was 562 policies "
            "optimised to do exactly this, and a dive's median over 8 replicas "
            "is a confident +0.4 m — which is why 'insert on robust fitness' "
            "readmits v1 wholesale."
        ),
        joints={
            HIP_L: Sinusoid(amp=0.0, offset=0.7),
            HIP_R: Sinusoid(amp=0.0, offset=-0.7),
            ANKLE_L: Sinusoid(amp=0.0, offset=-0.6),
            ANKLE_R: Sinusoid(amp=0.0, offset=0.6),
        },
    ),
    Probe(
        "twitcher",
        mode="crawl",
        spawn="prone",
        positive=False,
        freq=2.6,
        note=(
            "Prone, every joint oscillating, no net progress. Reads f_body ~ 1 "
            "exactly like a crawl — the support-class axis cannot tell them "
            "apart and never should; the progress clause is what does."
        ),
        joints={
            HIP_L: Sinusoid(amp=0.35, phase=0.0),
            HIP_R: Sinusoid(amp=0.35, phase=0.0),
            KNEE_L: Sinusoid(amp=0.45, phase=PI / 2),
            KNEE_R: Sinusoid(amp=0.45, phase=PI / 2),
            ANKLE_L: Sinusoid(amp=0.3, phase=PI),
            ANKLE_R: Sinusoid(amp=0.3, phase=PI),
            HEAD_Y: Sinusoid(amp=0.5, phase=0.0),
        },
    ),
    Probe(
        "home_hold",
        mode="walk",
        spawn="stand",
        positive=False,
        freq=1.0,
        ramp_seconds=0.0,
        note="Servos holding HOME and nothing else. Topples at ~1.1-1.3 s: on "
        "this robot everything falls by default, which is why a flat fall "
        "penalty degenerates to 'dive forward fastest'.",
        joints={},
    ),
    Probe(
        "prone_still",
        mode="crawl",
        spawn="prone",
        positive=False,
        freq=1.0,
        ramp_seconds=0.0,
        note="Lying prone doing nothing — the cheapest state a crawl reward "
        "could be farmed from, and the one P2' has to exclude for the crawl "
        "sub-archive to mean anything.",
        joints={},
    ),
)

BASE_BY_NAME: dict[str, Probe] = {p.name: p for p in PROBES}


# --------------------------------------------------------------------------- #
# Tuned by measurement, not by taste
# --------------------------------------------------------------------------- #
#
# The hand-written parameters above all crawled BACKWARDS (-0.05 to -0.24 m over
# 7 s) and none of the roll probes rotated. Rather than hand-tune, the whole
# (base x direction x frequency x amplitude) grid was swept — 396 variants over
# six batched rollouts, because one probe per world makes a sweep cost the same
# as a single probe. These are the settings that came out, quoted with what they
# measured over 128 worlds so the numbers can be checked rather than trusted.
#
# `sign=-1` reverses the stroke (every phase and amplitude negated). That it is
# the winning direction for every crawl is not a surprise worth hiding: a drag
# gait's direction of travel is a property of the phase relationship, and it was
# a coin flip which way the hand-written version pointed.


def _tuned(base: str, *, spawn_pose: str, freq: float, amp_scale: float,
           sign: int, name: str, note: str, positive: bool = True) -> Probe:
    src = dataclasses.replace(BASE_BY_NAME[base], spawn=spawn_pose)
    built = variants(src, sign=(sign,), freq=(freq,), amp_scale=(amp_scale,))[0]
    return dataclasses.replace(built, name=name, note=note, positive=positive)


TUNED: tuple[Probe, ...] = (
    _tuned(
        "crawl_chin_drag", spawn_pose="prone", freq=2.0, amp_scale=1.8, sign=-1,
        name="crawl_chin_drag_tuned",
        note="+0.73 m / 7 s, worst 2 s window +0.216 m, f_body 0.71, "
             "p95|a_z| 11.3. The strongest crawl in the sweep that still "
             "classifies as one.",
    ),
    _tuned(
        "crawl_chin_drag", spawn_pose="prone", freq=1.4, amp_scale=1.8, sign=-1,
        name="crawl_chin_drag_slow",
        note="+0.72 m, worst window +0.181 m, f_body 0.68, p95|a_z| 10.1. The "
             "same gait at 1.4 Hz: a second crawl positive that is not a "
             "replica of the first.",
    ),
    _tuned(
        "crawl_chin_drag", spawn_pose="prone", freq=2.0, amp_scale=1.4, sign=-1,
        name="crawl_chin_drag_gentle",
        note="+0.56 m, worst window +0.166 m, f_body 0.74 — the gentlest "
             "setting that still sustains progress.",
    ),
    _tuned(
        "crawl_belly_push", spawn_pose="supine", freq=3.0, amp_scale=2.2, sign=-1,
        name="crawl_supine_push",
        note="+0.56 m, worst window +0.164 m, f_body 0.99 — flat on the back "
             "shell, the purest f_body = 1 crawl in the set.",
    ),
    _tuned(
        "crawl_belly_push_inphase", spawn_pose="prone", freq=3.0, amp_scale=1.4,
        sign=-1, name="crawl_inphase_tuned",
        note="+0.39 m, worst window +0.115 m, f_body 0.88 — a breaststroke "
             "drag, structurally different from the chin drag.",
    ),
    _tuned(
        "crawl_chin_drag", spawn_pose="prone", freq=2.0, amp_scale=2.6, sign=-1,
        name="thrash", positive=False,
        note="THE IMPACT PROBE. The same chin drag over-driven: it travels "
             "+0.77 m and sustains +0.209 m per window, so the progress and "
             "label clauses do NOT exclude it — but p95|a_z| is 27.9 against "
             "11.3 for the crawl it is a caricature of, and f_air crosses the "
             "hop threshold. This is the degenerate the impact cap exists for, "
             "and the only probe in the set that makes the cap earn its place.",
    ),
)

PROBES = PROBES + TUNED

def by_name(name: str) -> Probe:
    if name not in BY_NAME:
        raise KeyError(f"unknown probe {name!r}; known: {sorted(BY_NAME)}")
    return BY_NAME[name]


def positives() -> tuple[Probe, ...]:
    return tuple(p for p in PROBES if p.positive)


def negatives() -> tuple[Probe, ...]:
    return tuple(p for p in PROBES if not p.positive)


# --------------------------------------------------------------------------- #
# Running one
# --------------------------------------------------------------------------- #


def trajectory(
    probes_per_world: list[Probe],
    home: torch.Tensor,
    joint_names: tuple[str, ...],
    steps: int,
    control_dt: float,
) -> torch.Tensor:
    """``(T, N, 14)`` servo targets, one probe per world.

    Precomputed rather than evaluated in the loop, and *per world* rather than
    per batch: that is what turns "hand-tune a scripted crawl" into "sweep 64
    variants for the price of one rollout", which is the only honest way to
    find out whether a mode moves at all before deciding it does not.
    """
    n = home.shape[0]
    if len(probes_per_world) != n:
        raise ValueError(
            f"{len(probes_per_world)} probes for {n} worlds; the assignment "
            "must be one per world so a variant is identified by its slot"
        )
    t = np.arange(steps, dtype=np.float64) * control_dt
    delta = np.zeros((steps, n, len(joint_names)), dtype=np.float32)
    for w, probe in enumerate(probes_per_world):
        ramp = np.ones_like(t) if probe.ramp_seconds <= 0 else np.minimum(
            1.0, t / probe.ramp_seconds
        )
        for j, name in enumerate(joint_names):
            sin = probe.joints.get(name)
            if sin is None:
                continue
            f = probe.freq if sin.freq is None else sin.freq
            delta[:, w, j] = ramp * (
                sin.offset + sin.amp * np.sin(2.0 * np.pi * f * t + sin.phase)
            )
    out = torch.as_tensor(delta, device=home.device)
    return home.unsqueeze(0) + out


def run_probe_batch(
    harness,
    probes_per_world: list[Probe],
    pose=None,
    windows=None,
    settle_seconds: float = 0.4,
):
    """Roll one probe per world in a single batch; return ``ModeFeatures``.

    Every world gets its own trajectory, so a sweep of variants and a set of
    replicas are the same operation: repeat a probe k times in the list and its
    k slots are k replicas *across worlds* — which is the replica notion v3
    measured to be the right one (a world index carries a persistent bias, so
    re-running the same slot samples a sixth of the noise).
    """
    from qd import spawn as spawn_module
    from qd.modes import VerticalAccel, WindowCfg

    pose = pose or spawn_module.get(probes_per_world[0].spawn)
    windows = windows or WindowCfg(
        episode_seconds=harness.fitness.episode_seconds, control_dt=harness.control_dt
    )
    stats = harness.make_mode_stats(windows)

    harness.reset(pose)
    home = harness.spawn_servo_targets
    names = tuple(harness.servo_joint_names)
    traj = trajectory(
        probes_per_world, home, names, windows.episode_steps, harness.control_dt
    )

    for _ in range(round(settle_seconds / harness.control_dt)):
        harness.set_servo_targets(home)
        harness.step()

    stats.begin(harness.base_pos())
    accel = VerticalAccel(harness.num_envs, harness.device, harness.control_dt)
    accel.begin(harness.robot.data.root_link_lin_vel_w)
    for k in range(windows.episode_steps):
        harness.set_servo_targets(traj[k])
        harness.step()
        ch = harness.mode_channels()
        assert ch is not None
        stats.update(
            harness.base_pos(),
            harness.projected_gravity(),
            ch["contact_found"],
            ch["ang_vel_w"],
            trunk_az=accel.step(ch["lin_vel_w"]),
            contact_force=ch.get("contact_force"),
        )
    return stats.finalize()


def run_probe(harness, probe: Probe, pose=None, windows=None, settle_seconds: float = 0.4):
    """One probe in every world: ``num_envs`` replicas of a single behaviour."""
    features = run_probe_batch(
        harness,
        [probe] * harness.num_envs,
        pose=pose,
        windows=windows,
        settle_seconds=settle_seconds,
    )
    info = {
        "probe": probe.name,
        "intended_mode": probe.mode,
        "positive": probe.positive,
        "spawn": (pose or probe.spawn) if isinstance(pose, str) else probe.spawn,
        "median_displacement_m": float(np.median(features.displacement)),
    }
    return features, info


BY_NAME = {p.name: p for p in PROBES}
