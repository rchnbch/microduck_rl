"""Spawn poses for the v4 probes — and why the archive itself never uses them.

The archive spawns standing at HOME for *every* mode (design draft §4.1), so
displacement is measured from the same start and includes the stand-to-mode
transition — which is what a policy switch on the real robot costs. Window 1 of
P2' is exempt for exactly that reason.

The **probes** cannot. Measured on this robot, from a standing HOME spawn:

* a 0.4 s ramp to SIT topples at 0.36 s, before the pose is reached;
* 0 of 42 slow (2 s ramp) flat-foot squats held to the 3 s mark;
* the passive HOME hold itself is down by 1.1-1.3 s.

Balance here is closed-loop only, even for a crouch. An open-loop scripted
crawl started from standing would spend its episode toppling and measure
nothing about crawling, so Stage A' spawns each probe in its mode's rest pose
and the archive's own evaluations spawn standing. The window-1 exemption is
what keeps windows 2-6 comparable between the two.

Heights here are **spawn** heights, deliberately a few millimetres above the
settled rest height so the pose lands on the floor rather than starting
interpenetrated; ``qd.check_shell_contacts --rest-poses`` measures where each
one actually settles and whether it stays there. Do not copy a settled height
into this file from a previous model revision — the contact shell changed in
v4 (:mod:`mjlab_microduck.robot.shell_contacts`), and AGENTS.md has a story
about a 5 mm-wrong target height costing days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_S = math.sqrt(0.5)


@dataclass(frozen=True)
class SpawnPose:
    """A root orientation, a spawn height and optional joint overrides."""

    name: str
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    height: float = 0.125
    joint_pos: dict[str, float] = field(default_factory=dict)
    """``{joint-name regex: angle}``, applied over HOME."""

    description: str = ""


# The SIT keyframe from ``robot/microduck/scene.xml``, joint by joint. Written
# out by name rather than as a qpos vector: the joint *order* differs between
# the plain, backlash and roller models (AGENTS.md), and a positional copy is
# the exact bug that convention exists to prevent.
SIT_JOINTS: dict[str, float] = {
    r"^left_hip_yaw$": 0.0,
    r"^left_hip_roll$": 0.0,
    r"^left_hip_pitch$": -0.5236,
    r"^left_knee$": 1.0472,
    r"^left_ankle$": 0.0,
    r"^neck_pitch$": 0.5,
    r"^head_pitch$": 1.6,
    r"^head_yaw$": 0.0,
    r"^head_roll$": 0.0,
    r"^right_hip_yaw$": 0.0,
    r"^right_hip_roll$": 0.0,
    r"^right_hip_pitch$": 0.5236,
    r"^right_knee$": -1.0472,
    r"^right_ankle$": 0.0,
}

# Legs straightened and the head tucked: what a body actually lies on when it
# is prone, rather than HOME's standing crouch folded flat.
PRONE_JOINTS: dict[str, float] = {
    r"^(left|right)_hip_pitch$": 0.0,
    r"^(left|right)_knee$": 0.0,
    r"^(left|right)_ankle$": 0.0,
    r"^neck_pitch$": 0.0,
    r"^head_pitch$": 0.0,
}

STAND = SpawnPose(
    "stand",
    height=0.125,
    description="HOME, the archive's spawn for every mode.",
)
PRONE = SpawnPose(
    "prone",
    quat_wxyz=(_S, 0.0, _S, 0.0),  # +90 deg pitch: nose and belly to the floor
    height=0.045,
    joint_pos=PRONE_JOINTS,
    description="Face-down on the battery pack and side shells — the crawl probe's start.",
)
SUPINE = SpawnPose(
    "supine",
    quat_wxyz=(_S, 0.0, -_S, 0.0),  # -90 deg pitch: back to the floor
    height=0.055,
    joint_pos=PRONE_JOINTS,
    description="Face-up. Lower than prone: the back shell is flatter than the belly.",
)
SIDE = SpawnPose(
    "side",
    quat_wxyz=(_S, _S, 0.0, 0.0),  # +90 deg roll: lying on the left side
    height=0.072,
    joint_pos=PRONE_JOINTS,
    description="On its side, long axis along +x — the log-roll probe's start.",
)
SIT = SpawnPose(
    "sit",
    height=0.070,
    joint_pos=SIT_JOINTS,
    description="The SIT keyframe; the deepest pose measured stable for a full second.",
)

POSES: dict[str, SpawnPose] = {p.name: p for p in (STAND, PRONE, SUPINE, SIDE, SIT)}


def get(name: str) -> SpawnPose:
    if name not in POSES:
        raise KeyError(f"unknown spawn pose {name!r}; known: {sorted(POSES)}")
    return POSES[name]
