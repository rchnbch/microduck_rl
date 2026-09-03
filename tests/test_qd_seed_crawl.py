"""The clock in the CPG distillation, and the off-by-one it exists to prevent.

A CPG teacher is a function of the step index, not of the observation. On
DAgger round 0 it both *drives* the rollout and *labels* the states it drives
through — and if those are two separate calls, the clock advances twice per
step and every state gets labelled with the action from the step after it. The
symptom is a distilled seed that lags its teacher by one control step (20 ms),
which is not obviously wrong in a log and is very wrong on the robot.
"""

from __future__ import annotations

import torch
from qd.seed_crawl import _Clock


class FakeCpg:
    """A teacher whose output is its own step index, so drift is visible."""

    def __init__(self):
        self.step = 0

    def reset(self):
        self.step = 0

    def __call__(self, obs):
        out = torch.full((obs.shape[0], 1), float(self.step))
        self.step += 1
        return out


OBS = torch.zeros(3, 61)


def test_driving_and_labelling_the_same_step_advances_the_clock_once():
    teacher = FakeCpg()
    clock = _Clock(teacher)
    for expected in range(5):
        driven = clock.drive(OBS)
        labelled = clock.label(OBS)
        assert float(driven[0, 0]) == expected
        assert float(labelled[0, 0]) == expected, "label lagged the drive"
    assert teacher.step == 5, "the clock advanced once per step, not twice"


def test_labelling_alone_still_advances_the_clock():
    # DAgger rounds 1+ : the STUDENT drives and the teacher only labels, so the
    # clock has to advance on the label call instead.
    teacher = FakeCpg()
    clock = _Clock(teacher)
    for expected in range(5):
        assert float(clock.label(OBS)[0, 0]) == expected
    assert teacher.step == 5


def test_a_naive_teacher_would_drift_which_is_what_the_clock_prevents():
    # The bug, made explicit: two direct calls per step double the rate.
    teacher = FakeCpg()
    for _ in range(5):
        teacher(OBS)  # drive
        teacher(OBS)  # label
    assert teacher.step == 10


def test_reset_between_rollouts_puts_the_teacher_back_at_t_zero():
    teacher = FakeCpg()
    clock = _Clock(teacher)
    clock.drive(OBS)
    clock.label(OBS)
    teacher.reset()
    assert float(_Clock(teacher).drive(OBS)[0, 0]) == 0.0
