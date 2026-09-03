"""The CPG teacher's two roles, and the two bugs that cost a run each.

A CPG is a clock: its action is a function of time, not of state. That breaks
DAgger, whose expert must be able to answer "given where you are, what would
you do". Making it answerable took two corrections, and both produced
plausible-looking logs while being wrong:

* **the off-by-one** — on round 0 the teacher both drives and labels, and two
  separate calls advance the clock twice per step, labelling every state with
  the action from the step *after* it: a seed lagging its teacher by 20 ms;
* **the tracker driving** — the phase tracker matches a live pose against the
  commanded trajectory, and the robot's joints lag the command, so a tracker
  allowed to drive compounds that lag into a slower gait. Measured: the teacher
  itself went from +0.500 m to **-0.133 m**, turning a crawl into a backwards
  shuffle.
"""

from __future__ import annotations

import torch
from qd.seed_crawl import _Clock


class FakeCpg:
    """A teacher whose output is its own step index, so drift is visible.

    Mirrors the real ``CpgTeacher`` interface: ``drive`` advances the clock,
    ``label`` answers for a given state without advancing it when the phase
    tracker is doing the work.
    """

    def __init__(self, phase_tracking: bool = False):
        self.step = 0
        self.phase_tracking = phase_tracking
        self.labels_asked = 0

    def reset(self):
        self.step = 0

    def drive(self, obs):
        out = torch.full((obs.shape[0], 1), float(self.step))
        self.step += 1
        return out

    def label(self, obs):
        self.labels_asked += 1
        if self.phase_tracking:
            # Answers from state; does not touch the clock.
            return torch.full((obs.shape[0], 1), -1.0)
        out = torch.full((obs.shape[0], 1), float(self.step))
        self.step += 1
        return out


OBS = torch.zeros(3, 61)


# --------------------------------------------------------------------------- #
# The off-by-one
# --------------------------------------------------------------------------- #


def test_driving_and_labelling_the_same_step_advances_the_clock_once():
    teacher = FakeCpg()
    clock = _Clock(teacher)
    for expected in range(5):
        driven = clock.drive(OBS)
        labelled = clock.label(OBS)
        assert float(driven[0, 0]) == expected
        assert float(labelled[0, 0]) == expected, "label lagged the drive"
    assert teacher.step == 5, "the clock advanced once per step, not twice"


def test_on_round_zero_the_label_is_the_driven_action_not_a_recomputation():
    # The states being labelled are the ones the teacher itself produced, so
    # asking the teacher again is at best redundant and at worst a clock tick.
    teacher = FakeCpg(phase_tracking=True)
    clock = _Clock(teacher)
    clock.drive(OBS)
    assert float(clock.label(OBS)[0, 0]) == 0.0
    assert teacher.labels_asked == 0


def test_labelling_alone_still_answers_when_the_student_drives():
    # DAgger rounds 1+: the STUDENT drives, so the teacher only labels.
    teacher = FakeCpg(phase_tracking=True)
    clock = _Clock(teacher)
    for _ in range(5):
        assert float(clock.label(OBS)[0, 0]) == -1.0
    assert teacher.labels_asked == 5
    assert teacher.step == 0, "labelling from state must not advance the clock"


def test_a_naive_teacher_would_drift_which_is_what_the_clock_prevents():
    teacher = FakeCpg()
    for _ in range(5):
        teacher.drive(OBS)
        teacher.label(OBS)
    assert teacher.step == 10


# --------------------------------------------------------------------------- #
# The lag the tracker must not drive through
# --------------------------------------------------------------------------- #


def test_the_lag_calibration_is_a_median_of_measured_offsets():
    from qd.seed_crawl import CpgTeacher

    t = CpgTeacher.__new__(CpgTeacher)
    t._lag_samples = [3.0, 4.0, 4.0, 5.0, 40.0]  # one outlier
    t.lag = 0
    assert t.calibrate_lag() == 4, "a mean would have been dragged to 11"


def test_lag_calibration_is_zero_before_anything_is_measured():
    from qd.seed_crawl import CpgTeacher

    t = CpgTeacher.__new__(CpgTeacher)
    t._lag_samples = []
    t.lag = 0
    assert t.calibrate_lag() == 0
