"""CPU regression tests for walking-v3's behaviour-descriptor axes.

Two classes of thing are locked here, and they are the two ways a descriptor
can be silently wrong:

* **Arithmetic.** Each axis is a time average over the *upright* steps, so a
  hand-computable rollout must produce the hand-computed number — including
  the touchdown edge detection that step frequency and stride length are built
  on, which is the only per-step state in the accumulator.
* **Contamination.** A fallen robot must contribute nothing to its own
  descriptor. v1 filled its archive with policies whose duty factor was set by
  how they happened to collapse; every axis added in v3 has the same failure
  mode available to it, so the mask is tested once per class of axis.
"""

import numpy as np
import pytest
import torch

from qd.common import FitnessCfg, RolloutMetrics
from qd.descriptors import AXES, DescriptorCfg, GaitStats

UPRIGHT = torch.tensor([[0.0, 0.0, -1.0]])
DT = 0.02


def _pos(x: float, z: float = 0.12, y: float = 0.0) -> torch.Tensor:
    return torch.tensor([[x, y, z]])


def _extras(vy=0.0, yaw=0.0, qvel=0.0, tau=0.0, dof=14):
    return {
        "lin_vel_w": torch.tensor([[0.0, vy, 0.0]]),
        "ang_vel_b": torch.tensor([[0.0, 0.0, yaw]]),
        "joint_vel": torch.full((1, dof), float(qvel)),
        "qfrc_actuator": torch.full((1, dof), float(tau)),
    }


def _run(steps, descriptor=None, cfg=None, control_dt=DT):
    """``steps`` is a list of ``(pos, contact, extras)``; returns ``(measures, axes)``."""
    m = RolloutMetrics(
        1, cfg or FitnessCfg(), "cpu", descriptor=descriptor, control_dt=control_dt
    )
    m.begin(steps[0][0])
    for pos, contact, extras in steps:
        m.update(pos, UPRIGHT, torch.as_tensor([contact]), extras)
    _, measures, info = m.finalize()
    axes = {k[5:]: v for k, v in info.items() if k.startswith("axis/")}
    return measures, axes


def test_every_catalogued_axis_is_produced():
    """A name in AXES with no value in finalize() would be a silent KeyError
    the moment someone selected it as a descriptor."""
    _, axes = _run([(_pos(0.0), [True, True], _extras())] * 5)
    assert set(axes) == set(AXES)


def test_step_frequency_counts_touchdowns_not_contacts():
    """Two feet, one touchdown each per 10 steps (0.2 s) -> 10 Hz."""
    steps = []
    for i in range(20):
        down = (i % 10) < 5  # one rising edge per foot per 10 steps
        steps.append((_pos(0.0), [down, down], None))
    _, axes = _run(steps)
    # 2 rising edges per foot over 20 steps = 0.4 s -> 4 touchdowns / 0.4 s.
    assert axes["step_frequency"][0] == pytest.approx(10.0)


def test_stride_length_is_distance_per_touchdown():
    steps = [(_pos(0.05 * i), [i % 2 == 0, i % 2 == 1], None) for i in range(20)]
    _, axes = _run(steps)
    # 0.95 m of planar travel over 10 + 10 touchdowns.
    assert axes["stride_length"][0] == pytest.approx(0.95 / 20, abs=1e-4)


def test_posture_axes_are_time_averages():
    heights = [0.10, 0.14, 0.10, 0.14]
    steps = [(_pos(0.0, z), [True, True], None) for z in heights]
    _, axes = _run(steps)
    assert axes["torso_height_mean"][0] == pytest.approx(0.12)
    assert axes["torso_height_osc"][0] == pytest.approx(0.02, abs=1e-6)


def test_power_needs_both_channels_and_is_the_sum_of_products():
    steps = [(_pos(0.0), [True, True], _extras(qvel=2.0, tau=0.5, dof=14))] * 10
    _, axes = _run(steps)
    assert axes["power"][0] == pytest.approx(14 * 2.0 * 0.5)
    assert axes["joint_speed"][0] == pytest.approx(2.0)


def test_an_axis_whose_channel_was_never_supplied_reads_nan():
    """NaN, not zero: 'nobody measured this' must not look like 'it was zero',
    or a descriptor selected on a harness that cannot feed it would bin every
    genome into the same cell and read as perfectly repeatable."""
    _, axes = _run([(_pos(0.0), [True, True], None)] * 4)
    assert np.isnan(axes["power"][0])
    assert np.isnan(axes["yaw_rate"][0])
    assert np.isfinite(axes["duty_left"][0])


def test_post_fall_steps_reach_no_axis():
    """Everything *after* the fall-detection frame is invisible to every axis.

    The detection frame itself still counts — it is the last honest step, and
    the same one :class:`RolloutMetrics` freezes the displacement on — so it is
    given honest values here. What must not reach an axis is the 50 frames of
    thrashing that follow.
    """
    cfg = FitnessCfg(fall_height=0.09)
    honest = _extras(qvel=1.0, tau=1.0, yaw=0.1)
    steps = [
        (_pos(0.0, 0.12), [True, False], honest),
        (_pos(0.0, 0.12), [True, False], honest),
        (_pos(0.0, 0.08), [True, False], honest),  # fall detected on this frame
        *[
            (_pos(9.0, 0.02, y=9.0), [True, True], _extras(qvel=99.0, tau=99.0, yaw=9.0))
            for _ in range(50)
        ],
    ]
    _, axes = _run(steps, cfg=cfg)
    assert axes["duty_left"][0] == pytest.approx(1.0)
    assert axes["duty_right"][0] == pytest.approx(0.0)
    assert axes["joint_speed"][0] == pytest.approx(1.0)
    assert axes["yaw_rate"][0] == pytest.approx(0.1)
    assert axes["lateral_drift_rate"][0] == pytest.approx(0.0)
    assert axes["torso_height_mean"][0] == pytest.approx((0.12 + 0.12 + 0.08) / 3)


def test_touchdown_edges_do_not_fire_on_the_frames_after_a_fall():
    """A robot skidding on its face slaps the floor; those are not steps."""
    cfg = FitnessCfg(fall_height=0.09)
    steps = [
        (_pos(0.0, 0.12), [False, False], None),
        (_pos(0.0, 0.08), [False, False], None),  # fall detected, feet still off
    ]
    for i in range(40):
        steps.append((_pos(0.0, 0.02), [i % 2 == 0, i % 2 == 0], None))
    _, axes = _run(steps, cfg=cfg)
    assert axes["step_frequency"][0] == pytest.approx(0.0)
    assert axes["stride_length"][0] == pytest.approx(0.0)


def test_measures_are_the_two_chosen_axes_clipped_into_the_grid():
    d = DescriptorCfg("torso_height_mean", "joint_speed", (0.11, 0.13), (0.0, 2.0))
    steps = [(_pos(0.0, 0.20), [True, True], _extras(qvel=99.0, tau=1.0))] * 4
    measures, _ = _run(steps, descriptor=d)
    assert measures.shape == (1, 2)
    # Both are past the top of their range and must land inside the last bin,
    # not on the boundary, where pyribs' half-open interval has no cell.
    assert 0.11 <= measures[0, 0] < 0.13
    assert 0.0 <= measures[0, 1] < 2.0


def test_the_default_descriptor_is_still_v2s_duty_factor():
    """v1/v2 code paths must keep measuring exactly what they used to."""
    d = DescriptorCfg()
    assert d.names == ("duty_left", "duty_right")
    assert d.ranges == [(0.0, 1.0), (0.0, 1.0)]
    assert d.needs == ()
    steps = [(_pos(0.0), [True, False], None)] * 10
    measures, _ = _run(steps)
    np.testing.assert_allclose(measures[0], [1.0, 0.0])


def test_an_unknown_axis_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown descriptor axis"):
        DescriptorCfg("not_an_axis", "duty_right")


def test_descriptor_round_trips_through_archive_meta():
    d = DescriptorCfg("torso_height_mean", "joint_speed", (0.11, 0.13), (0.5, 2.5))
    back = DescriptorCfg.from_meta(d.to_meta())
    assert back == d
    # A v1/v2 archive has no descriptor key and must come back as duty factor,
    # so an old checkpoint is never silently re-binned on v3's axes.
    assert DescriptorCfg.from_meta({"algorithm": "pga_me_mlp"}) == DescriptorCfg()


def test_gait_stats_begin_clears_state_between_rollouts():
    s = GaitStats(1, "cpu", DT)
    s.begin(_pos(0.0))
    counted = torch.tensor([True])
    s.update(counted, _pos(1.0), UPRIGHT, torch.tensor([[True, True]]), _extras(qvel=3.0))
    s.begin(_pos(0.0))
    s.update(counted, _pos(0.0), UPRIGHT, torch.tensor([[False, False]]), None)
    axes = s.finalize(torch.tensor([1]))
    assert axes["duty_left"][0] == pytest.approx(0.0)
    assert np.isnan(axes["joint_speed"][0])
