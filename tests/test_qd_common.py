"""CPU regression tests for the QD objective / behaviour descriptor.

The fall latch is the part worth locking: without it a face-plant that skids
forward keeps accruing both fitness and duty factor, which is exactly the
degenerate solution the archive would then fill up with.
"""

import numpy as np
import pytest
import torch
from qd.common import FitnessCfg, RolloutMetrics, load_archive, save_archive

UPRIGHT = torch.tensor([0.0, 0.0, -1.0])
ON_ITS_SIDE = torch.tensor([-1.0, 0.0, 0.0])


def _pos(x: float, z: float = 0.12) -> torch.Tensor:
    return torch.tensor([[x, 0.0, z]])


def _grav(vec: torch.Tensor) -> torch.Tensor:
    return vec.reshape(1, 3)


def test_upright_walk_scores_its_displacement():
    cfg = FitnessCfg(fall_penalty=0.25)
    m = RolloutMetrics(1, cfg, "cpu")
    m.begin(_pos(0.0))
    for i in range(10):
        m.update(_pos(0.05 * (i + 1)), _grav(UPRIGHT), torch.tensor([[True, False]]))
    fitness, measures, info = m.finalize()
    assert fitness[0] == pytest.approx(0.5, abs=1e-5)
    assert not info["fell"][0]
    # Left foot down every step, right foot never.
    np.testing.assert_allclose(measures[0], [1.0, 0.0])


def test_duty_factor_is_the_contact_fraction():
    m = RolloutMetrics(1, FitnessCfg(), "cpu")
    m.begin(_pos(0.0))
    for i in range(10):
        left = bool(i % 2 == 0)  # 50% duty
        right = i < 3  # 30% duty
        m.update(_pos(0.0), _grav(UPRIGHT), torch.tensor([[left, right]]))
    _, measures, _ = m.finalize()
    np.testing.assert_allclose(measures[0], [0.5, 0.3], atol=1e-6)


def test_a_fall_freezes_the_displacement_and_costs_the_penalty():
    """A face-plant that keeps sliding must not keep earning."""
    cfg = FitnessCfg(fall_penalty=0.25, fall_height=0.09)
    m = RolloutMetrics(1, cfg, "cpu")
    m.begin(_pos(0.0))
    m.update(_pos(0.10), _grav(UPRIGHT), torch.tensor([[True, True]]))
    m.update(_pos(0.20, z=0.04), _grav(UPRIGHT), torch.tensor([[True, True]]))  # falls
    for x in (0.5, 1.0, 5.0):  # skids a long way after falling
        m.update(_pos(x, z=0.04), _grav(UPRIGHT), torch.tensor([[True, True]]))
    fitness, _, info = m.finalize()
    assert info["fell"][0]
    # Frozen at the position where the fall was detected; the penalty is
    # charged for the 3 of 5 steps spent down.
    assert info["survival_fraction"][0] == pytest.approx(2 / 5)
    assert fitness[0] == pytest.approx(0.20 - 0.25 * 3 / 5, abs=1e-5)


def test_falling_late_is_cheaper_than_falling_early():
    """The penalty is pro-rata, so a ballistic dive cannot beat a gait that
    covers the same ground and stays up longer."""
    cfg = FitnessCfg(fall_penalty=0.25, fall_height=0.09)

    def run(fall_step: int) -> float:
        m = RolloutMetrics(1, cfg, "cpu")
        m.begin(_pos(0.0))
        for i in range(10):
            down = i >= fall_step
            m.update(
                _pos(0.3 if down else 0.03 * (i + 1), z=0.02 if down else 0.12),
                _grav(UPRIGHT),
                torch.tensor([[True, True]]),
            )
        return float(m.finalize()[0][0])

    diver = run(1)  # 0.03 m banked, down for 9/10 of the episode
    survivor = run(9)  # walks the whole way, trips at the end
    assert survivor > diver


def test_a_flip_is_detected_from_tilt_not_height():
    """robot_walk.xml has no trunk collision geoms, so the tilt check is the
    only thing that catches a robot that topples at full height."""
    cfg = FitnessCfg(fall_penalty=0.25, fall_tilt_deg=60.0)
    m = RolloutMetrics(1, cfg, "cpu")
    m.begin(_pos(0.0))
    m.update(_pos(0.1), _grav(UPRIGHT), torch.tensor([[True, True]]))
    m.update(_pos(0.2), _grav(ON_ITS_SIDE), torch.tensor([[True, True]]))
    _, _, info = m.finalize()
    assert info["fell"][0]


def test_post_fall_steps_do_not_pollute_the_descriptor():
    cfg = FitnessCfg(fall_height=0.09)
    m = RolloutMetrics(1, cfg, "cpu")
    m.begin(_pos(0.0))
    m.update(_pos(0.0), _grav(UPRIGHT), torch.tensor([[True, False]]))
    m.update(_pos(0.0, z=0.02), _grav(UPRIGHT), torch.tensor([[True, False]]))  # falls
    for _ in range(50):  # lying down, both feet touching
        m.update(_pos(0.0, z=0.02), _grav(UPRIGHT), torch.tensor([[True, True]]))
    _, measures, info = m.finalize()
    assert info["alive_steps"][0] == 2
    np.testing.assert_allclose(measures[0], [1.0, 0.0])


def test_nan_state_falls_and_floors_the_fitness():
    cfg = FitnessCfg(min_fitness=-5.0)
    m = RolloutMetrics(1, cfg, "cpu")
    m.begin(_pos(0.0))
    m.update(
        torch.tensor([[float("nan"), 0.0, float("nan")]]),
        _grav(UPRIGHT),
        torch.tensor([[False, False]]),
    )
    fitness, measures, info = m.finalize()
    assert info["fell"][0]
    assert np.isfinite(fitness[0]) and fitness[0] >= cfg.min_fitness
    assert np.all(np.isfinite(measures))


def test_measures_stay_inside_the_archive_ranges():
    """Anything outside [0, 1]^2 would be silently dropped by the GridArchive."""
    rng = np.random.default_rng(0)
    n = 32
    m = RolloutMetrics(n, FitnessCfg(), "cpu")
    pos = torch.zeros(n, 3)
    pos[:, 2] = 0.12
    m.begin(pos)
    for _ in range(20):
        contact = torch.as_tensor(rng.random((n, 2)) > 0.5)
        m.update(pos, UPRIGHT.repeat(n, 1), contact)
    _, measures, _ = m.finalize()
    assert measures.shape == (n, 2)
    assert np.all((measures >= 0.0) & (measures <= 1.0))


def test_batched_envs_latch_independently():
    cfg = FitnessCfg(fall_penalty=0.25, fall_height=0.09)
    m = RolloutMetrics(2, cfg, "cpu")
    pos = torch.tensor([[0.0, 0.0, 0.12], [0.0, 0.0, 0.12]])
    m.begin(pos)
    contact = torch.tensor([[True, True], [True, True]])
    grav = UPRIGHT.repeat(2, 1)
    m.update(torch.tensor([[0.3, 0.0, 0.12], [0.3, 0.0, 0.02]]), grav, contact)
    m.update(torch.tensor([[0.6, 0.0, 0.12], [9.0, 0.0, 0.02]]), grav, contact)
    fitness, _, info = m.finalize()
    assert list(info["fell"]) == [False, True]
    assert fitness[0] == pytest.approx(0.6, abs=1e-5)
    # Env 1 fell on step 1 of 2, so it is down for half the episode.
    assert fitness[1] == pytest.approx(0.3 - 0.25 * 0.5, abs=1e-5)


def test_archive_checkpoint_round_trips(tmp_path):
    ribs = pytest.importorskip("ribs")
    del ribs
    from qd.common import make_archive

    archive = make_archive(solution_dim=31, grid_dims=(20, 20), qd_score_offset=-5.0)
    rng = np.random.default_rng(0)
    sols = rng.random((10, 31))
    archive.add(sols, rng.random(10), rng.random((10, 2)))

    path = save_archive(archive, tmp_path / "a.npz", meta={"algorithm": "test"})
    data = load_archive(path)
    assert data["solution"].shape[1] == 31
    assert data["measures"].shape[1] == 2
    assert data["meta"]["algorithm"] == "test"
    np.testing.assert_array_equal(data["grid_dims"], [20, 20])
