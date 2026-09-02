"""P2' and the mode classifier, on synthetic rollouts with known verdicts.

Every case here is one of the behaviours the design draft argues about, built
by hand so the predicate can be checked against the thing it was written to
admit or exclude — a dive, a late faller, a steady crawl — without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from qd.modes import (
    MODE_INDEX,
    MODES,
    ClassifierCfg,
    ModeStats,
    VerticalAccel,
    ViabilityCfg,
    WindowCfg,
    dominant_frequency,
    evaluate_viability,
    label_agreement,
)

DT = 0.02
GEOMS = 4
FOOT_COLS = (0, 1)
HEAD_COLS = (3,)


def make_stats(num_envs: int, windows: WindowCfg | None = None) -> ModeStats:
    return ModeStats(
        num_envs,
        "cpu",
        windows or WindowCfg(control_dt=DT),
        num_contact_geoms=GEOMS,
        foot_columns=FOOT_COLS,
        head_columns=HEAD_COLS,
        contact_force_n=0.0,
    )


def run(
    num_envs: int,
    x_of_t,
    contact_of_t,
    omega_of_t=None,
    omega_axis: int = 1,
    az_of_t=None,
    gravity_z: float = -1.0,
    windows: WindowCfg | None = None,
):
    """Drive a ModeStats through a whole scripted episode."""
    w = windows or WindowCfg(control_dt=DT)
    stats = make_stats(num_envs, w)
    pos0 = torch.zeros(num_envs, 3)
    pos0[:, 0] = x_of_t(0.0)
    pos0[:, 2] = 0.12
    stats.begin(pos0)
    grav = torch.zeros(num_envs, 3)
    grav[:, 2] = gravity_z
    for k in range(w.episode_steps):
        t = (k + 1) * DT
        pos = torch.zeros(num_envs, 3)
        pos[:, 0] = x_of_t(t)
        pos[:, 2] = 0.12
        contact = contact_of_t(t).expand(num_envs, GEOMS).clone()
        omega = torch.zeros(num_envs, 3)
        if omega_of_t is not None:
            omega[:, omega_axis] = omega_of_t(t)
        az = None
        if az_of_t is not None:
            az = torch.full((num_envs,), float(az_of_t(t)))
        stats.update(pos, grav, contact, omega, trunk_az=az)
    return stats.finalize()


FEET_ONLY = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
BODY_ONLY = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
AIRBORNE = torch.tensor([[0.0, 0.0, 0.0, 0.0]])


# --------------------------------------------------------------------------- #
# Window geometry
# --------------------------------------------------------------------------- #


def test_window_geometry_is_the_design_draft_s():
    w = WindowCfg()
    assert w.steps_per_slot == 50
    assert w.num_slots == 7
    assert w.slots_per_window == 2
    assert w.num_windows == 6
    assert w.episode_steps == 350


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": 1.5},  # not a whole multiple of the stride
        {"episode_seconds": 7.5},  # not a whole number of slots
        {"stride_seconds": 0.015},  # not a whole number of control steps
        {"window_seconds": 8.0},  # does not fit the episode
    ],
)
def test_window_geometry_refuses_configurations_it_cannot_accumulate(kwargs):
    with pytest.raises(ValueError):
        WindowCfg(**kwargs)


def test_window_of_1_5s_is_expressible_at_a_0_5s_stride():
    # The Stage A' sweep asks for W in {1.5, 2, 3}; 1.5 needs the finer stride.
    w = WindowCfg(window_seconds=1.5, stride_seconds=0.5)
    assert w.num_slots == 14
    assert w.slots_per_window == 3
    assert w.num_windows == 12


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #


def test_classifier_precedence_is_roll_hop_crawl_walk_other():
    cfg = ClassifierCfg()
    # A rolling body is rolling even though it is also mostly on its shell.
    assert cfg.label(0.9, 0.0, 1.2) == MODE_INDEX["roll"]
    # Airborne beats prone, but not rotation.
    assert cfg.label(0.9, 0.5, 0.0) == MODE_INDEX["hop"]
    assert cfg.label(0.9, 0.0, 0.0) == MODE_INDEX["crawl"]
    assert cfg.label(0.0, 0.0, 0.0) == MODE_INDEX["walk"]
    # A knee-assisted shuffle: f_body between the walk and crawl thresholds.
    assert cfg.label(0.3, 0.0, 0.0) == MODE_INDEX["other"]


def test_classifier_broadcasts_over_windows():
    cfg = ClassifierCfg()
    f_body = np.array([[0.0, 0.9], [0.0, 0.9]])
    out = cfg.label(f_body, np.zeros_like(f_body), np.zeros_like(f_body))
    assert out.shape == (2, 2)
    assert out[0, 0] == MODE_INDEX["walk"]
    assert out[1, 1] == MODE_INDEX["crawl"]


def test_hop_label_exists_but_has_no_seed_path():
    # Appendix A measured hopping out of reach; the label stays so a bounding
    # gait is filed rather than mislabelled.
    assert "hop" in MODES


# --------------------------------------------------------------------------- #
# The accumulator's arithmetic
# --------------------------------------------------------------------------- #


def test_steady_walker_has_one_metre_of_progress_in_every_window():
    feats = run(3, lambda t: 0.5 * t, lambda t: FEET_ONLY)
    assert feats.window_dx.shape == (6, 3)
    np.testing.assert_allclose(feats.window_dx, 1.0, atol=1e-4)
    np.testing.assert_allclose(feats.f_feet, 1.0, atol=1e-6)
    np.testing.assert_allclose(feats.f_body, 0.0, atol=1e-6)
    np.testing.assert_allclose(feats.f_air, 0.0, atol=1e-6)
    np.testing.assert_allclose(feats.displacement, 3.5, atol=1e-4)


def test_support_fractions_partition_the_episode():
    feats = run(2, lambda t: 0.0, lambda t: BODY_ONLY if t < 3.5 else AIRBORNE)
    total = feats.f_feet + feats.f_body + feats.f_air
    np.testing.assert_allclose(total, 1.0, atol=1e-6)
    np.testing.assert_allclose(feats.f_body, 0.5, atol=0.01)
    np.testing.assert_allclose(feats.f_air, 0.5, atol=0.01)


def test_rotation_rate_counts_only_supported_rotation():
    # 1 rad/s about a world-horizontal axis, but airborne throughout.
    feats = run(2, lambda t: 0.0, lambda t: AIRBORNE, omega_of_t=lambda t: 1.0)
    np.testing.assert_allclose(feats.rotation_rate, 0.0, atol=1e-6)
    # The same rotation while something is touching is a roll.
    feats = run(2, lambda t: 0.0, lambda t: BODY_ONLY, omega_of_t=lambda t: 1.0)
    np.testing.assert_allclose(feats.rotation_rate, 1.0, atol=0.01)


def test_a_ground_spin_is_not_a_roll():
    # The measurement that forced the world frame: a side-lying robot
    # pirouetting on the floor accumulated 1.69 rad/s on the body's lateral
    # axis and classified as a roll while covering -3 cm.
    feats = run(2, lambda t: 0.0, lambda t: BODY_ONLY, omega_axis=2, omega_of_t=lambda t: 3.0)
    np.testing.assert_allclose(feats.rotation_rate, 0.0, atol=1e-6)
    np.testing.assert_allclose(feats.rotation_rate_yaw, 3.0, atol=0.01)
    assert ClassifierCfg().label(
        feats.f_body, feats.f_air, feats.rotation_rate
    ).tolist() == [MODE_INDEX["crawl"]] * 2


def test_a_log_roll_about_world_x_is_a_roll():
    feats = run(2, lambda t: 0.1 * t, lambda t: BODY_ONLY, omega_axis=0, omega_of_t=lambda t: 1.2)
    np.testing.assert_allclose(feats.rotation_rate, 1.2, atol=0.01)
    assert ClassifierCfg().label(
        feats.f_body, feats.f_air, feats.rotation_rate
    ).tolist() == [MODE_INDEX["roll"]] * 2


def test_net_rotation_cancels_for_a_rocking_body():
    feats = run(
        2,
        lambda t: 0.0,
        lambda t: BODY_ONLY,
        omega_of_t=lambda t: 3.0 if int(t * 2) % 2 == 0 else -3.0,
    )
    assert np.all(feats.rotation_rate < 0.2)


def test_p95_az_is_a_percentile_not_a_max():
    # One violent frame in 350 must not set the number the cap is compared to.
    feats = run(2, lambda t: 0.0, lambda t: FEET_ONLY, az_of_t=lambda t: 100.0 if t < 0.03 else 1.0)
    np.testing.assert_allclose(feats.p95_az, 1.0, atol=1e-3)


def test_force_threshold_removes_single_step_contact_chatter():
    windows = WindowCfg(control_dt=DT)
    stats = ModeStats(1, "cpu", windows, GEOMS, FOOT_COLS, HEAD_COLS, contact_force_n=0.5)
    pos = torch.zeros(1, 3)
    grav = torch.tensor([[0.0, 0.0, -1.0]])
    stats.begin(pos)
    for k in range(windows.episode_steps):
        found = FEET_ONLY.clone()
        # `found` says contact every step; the force says it is a 0.1 N graze
        # on every tenth step, which raw `found` would count as solid support.
        force = torch.full((1, GEOMS), 5.0)
        if k % 10 == 0:
            force[:] = 0.1
        stats.update(pos, grav, found, torch.zeros(1, 3), contact_force=force)
    feats = stats.finalize()
    np.testing.assert_allclose(feats.f_air, 0.1, atol=0.01)


# --------------------------------------------------------------------------- #
# P2'
# --------------------------------------------------------------------------- #


def test_steady_walker_is_viable():
    feats = run(2, lambda t: 0.2 * t, lambda t: FEET_ONLY)
    v = evaluate_viability(feats)
    assert v.viable.all()
    assert (v.label == MODE_INDEX["walk"]).all()


def test_a_dive_is_excluded_because_its_progress_is_front_loaded():
    # 0.4 m in the first second, then lies still: the v1 archive's signature.
    feats = run(2, lambda t: min(t, 1.0) * 0.4, lambda t: BODY_ONLY)
    v = evaluate_viability(feats)
    assert not v.viable.any()
    assert not v.progress.any()


def test_a_prone_crawl_is_viable_and_labelled_crawl():
    feats = run(2, lambda t: 0.06 * t, lambda t: BODY_ONLY)
    v = evaluate_viability(feats, ViabilityCfg(d_min=0.05))
    assert v.viable.all()
    assert (v.label == MODE_INDEX["crawl"]).all()


def test_a_late_faller_passes_progress_and_fails_the_label_clause():
    # Walks the whole episode, banks the last window's progress before 6.5 s,
    # then face-plants: P2 admits it, P2' does not.
    def x(t):
        return 0.5 * min(t, 6.4)

    def contact(t):
        return FEET_ONLY if t < 6.4 else BODY_ONLY

    feats = run(2, x, contact)
    v = evaluate_viability(feats)
    assert v.progress.all(), "the hole P2 alone leaves open"
    assert not v.constant_label.any()
    assert not v.viable.any()


def test_the_leading_transition_is_exempt_so_a_mode_may_take_a_second_to_start():
    # Stands still for the first 1.5 s (as a crawl lowering itself does), then
    # crawls steadily.
    def x(t):
        return 0.0 if t < 1.5 else 0.06 * (t - 1.5)

    def contact(t):
        return FEET_ONLY if t < 1.0 else BODY_ONLY

    feats = run(2, x, contact)
    v = evaluate_viability(feats, ViabilityCfg(d_min=0.02))
    assert v.viable.all()


def test_standing_still_is_not_viable():
    feats = run(2, lambda t: 0.0, lambda t: FEET_ONLY)
    assert not evaluate_viability(feats).viable.any()


def test_impact_cap_excludes_a_violent_rollout_when_enabled():
    feats = run(2, lambda t: 0.2 * t, lambda t: FEET_ONLY, az_of_t=lambda t: 60.0)
    assert evaluate_viability(feats).viable.all(), "no cap configured -> clause off"
    v = evaluate_viability(feats, ViabilityCfg(impact_cap=40.0))
    assert not v.viable.any()
    assert not v.impact.any()


def test_a_non_finite_rollout_is_never_viable():
    windows = WindowCfg(control_dt=DT)
    stats = make_stats(2, windows)
    stats.begin(torch.zeros(2, 3))
    grav = torch.tensor([[0.0, 0.0, -1.0]] * 2)
    for k in range(windows.episode_steps):
        pos = torch.zeros(2, 3)
        pos[:, 0] = 0.2 * (k + 1) * DT
        if k == 100:
            pos[1, 0] = float("nan")
        stats.update(pos, grav, FEET_ONLY.expand(2, GEOMS), torch.zeros(2, 3))
    v = evaluate_viability(stats.finalize())
    assert v.viable[0] and not v.viable[1]


def test_clause_rates_are_reported_separately():
    feats = run(4, lambda t: min(t, 1.0) * 0.4, lambda t: BODY_ONLY)
    rates = evaluate_viability(feats).rates()
    assert rates["finite"] == 1.0
    assert rates["progress"] == 0.0
    assert set(rates) == {"viable", "finite", "progress", "constant_label", "impact"}


# --------------------------------------------------------------------------- #
# Replica agreement
# --------------------------------------------------------------------------- #


def test_label_agreement_reports_the_modal_label_and_its_count():
    walk, crawl = MODE_INDEX["walk"], MODE_INDEX["crawl"]
    labels = np.array([[walk, crawl], [walk, crawl], [walk, walk], [walk, crawl]])
    mode, count = label_agreement(labels)
    assert list(mode) == [walk, crawl]
    assert list(count) == [4, 3]


def test_vertical_accel_is_zero_at_rest_and_spikes_on_an_impact():
    # The reason this is a finite difference of v_z and not the accelerometer:
    # an IMU reads 9.81 on a robot lying perfectly still.
    accel = VerticalAccel(2, "cpu", DT)
    still = torch.zeros(2, 3)
    accel.begin(still)
    assert torch.allclose(accel.step(still), torch.zeros(2))
    landing = torch.zeros(2, 3)
    landing[:, 2] = -1.0
    accel.step(landing)
    assert torch.allclose(accel.step(still), torch.full((2,), 50.0))


def test_dominant_frequency_finds_a_planted_tone():
    t = np.arange(350) * DT
    trace = np.stack([np.sin(2 * np.pi * 4.6 * t), np.sin(2 * np.pi * 0.57 * t)], axis=1)
    freqs = dominant_frequency(trace, DT)
    assert abs(freqs[0] - 4.57) < 0.2
    assert abs(freqs[1] - 0.57) < 0.2
