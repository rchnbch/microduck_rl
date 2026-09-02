"""The v4 insertion rule, folded over replicas — §4.2 of the design draft.

Three asymmetries are tested here because each removes a specific kind of luck,
and v1-v3 each paid for one of them: viability unanimous (v2's 2-of-2 gate let
coin-flips in), fitness the median (v1 ranked cells by whose occupant got the
luckiest distance), and the mode label agreeing across replicas (new in v4 —
without it the chaos leaks into the archive's *geography*, not only its
fitness).
"""

from __future__ import annotations

import numpy as np
import pytest
from qd.modes import MODE_INDEX, MODES, ModeFeatures, ViabilityCfg, WindowCfg
from qd.pga.run_modes import ModeRewardCfg, Verdict, fold_replicas, insert

WINDOWS = WindowCfg()
NW = WINDOWS.num_windows


def features(n=1, *, dx=0.2, f_body=0.0, f_air=0.0, rot=0.0, az=1.0, finite=True):
    """A synthetic ModeFeatures batch with the same value in every window."""
    col = lambda v: np.full((NW, n), float(v), dtype=np.float32)
    ep = lambda v: np.full(n, float(v), dtype=np.float32)
    return ModeFeatures(
        window_dx=col(dx),
        window_f_body=col(f_body),
        window_f_air=col(f_air),
        window_f_feet=col(1.0 - f_body - f_air),
        window_f_head=col(0.0),
        window_rotation_rate=col(rot),
        f_body=ep(f_body),
        f_air=ep(f_air),
        f_feet=ep(1.0 - f_body - f_air),
        f_head=ep(0.0),
        f_inverted=ep(0.0),
        rotation_rate=ep(rot),
        rotation_rate_pitch=ep(rot),
        rotation_rate_roll=ep(0.0),
        rotation_rate_yaw=ep(0.0),
        p95_az=ep(az),
        displacement=ep(dx * NW),
        finite=np.full(n, finite, dtype=bool),
    )


def axes(n=1, height=0.119, speed=1.5):
    return {
        "torso_height_mean": np.full(n, height),
        "joint_speed": np.full(n, speed),
        "yaw_rate": np.zeros(n),
    }


def replicas(specs, n=1):
    return [(features(n, **kw), axes(n)) for kw in specs]


CFG = ViabilityCfg(windows=WINDOWS, d_min=0.05)


# --------------------------------------------------------------------------- #
# Viability is unanimous
# --------------------------------------------------------------------------- #


def test_a_candidate_viable_in_every_replica_is_admitted():
    v = fold_replicas(replicas([{}] * 8), CFG, unanimous=True, agreement_min=7)
    assert v.viable.tolist() == [True]


def test_one_failing_replica_out_of_eight_is_enough_to_reject():
    # v2's lesson: a policy that fails one time in N is a policy that fails,
    # and admitting it is how an archive fills with coin-flips.
    specs = [{}] * 7 + [{"dx": 0.0}]
    v = fold_replicas(replicas(specs), CFG, unanimous=True, agreement_min=7)
    assert v.viable.tolist() == [False]


def test_the_measured_exception_relaxes_to_seven_of_eight():
    specs = [{}] * 7 + [{"dx": 0.0}]
    v = fold_replicas(replicas(specs), CFG, unanimous=False, agreement_min=7)
    assert v.viable.tolist() == [True]


def test_six_of_eight_still_fails_the_relaxed_rule():
    specs = [{}] * 6 + [{"dx": 0.0}] * 2
    v = fold_replicas(replicas(specs), CFG, unanimous=False, agreement_min=7)
    assert v.viable.tolist() == [False]


# --------------------------------------------------------------------------- #
# Fitness is the median
# --------------------------------------------------------------------------- #


def test_fitness_is_the_median_not_the_max():
    # One lucky replica must not win a cell: that is the luck-ranking v3 removed.
    specs = [{"dx": 0.1}] * 7 + [{"dx": 10.0}]
    v = fold_replicas(replicas(specs), CFG, unanimous=True, agreement_min=7)
    assert v.fitness[0] == pytest.approx(0.1 * NW)


def test_fitness_is_the_median_not_the_mean():
    # And one catastrophic replica must not drag a good elite down.
    specs = [{"dx": 0.2}] * 7 + [{"dx": -20.0}]
    v = fold_replicas(replicas(specs), CFG, unanimous=False, agreement_min=7)
    assert v.fitness[0] == pytest.approx(0.2 * NW)


def test_descriptor_axes_are_medians_too():
    reps = [(features(1), axes(1, height=h)) for h in (0.11, 0.12, 0.13)]
    v = fold_replicas(reps, CFG, unanimous=True, agreement_min=3)
    assert v.axes["torso_height_mean"][0] == pytest.approx(0.12)


# --------------------------------------------------------------------------- #
# The label must agree
# --------------------------------------------------------------------------- #


def test_a_candidate_that_sometimes_walks_and_sometimes_crawls_is_rejected():
    specs = [{"f_body": 0.0}] * 5 + [{"f_body": 0.9}] * 3
    v = fold_replicas(replicas(specs), CFG, unanimous=True, agreement_min=7)
    assert v.viable.tolist() == [False]
    assert v.agreement[0] == 5


def test_seven_of_eight_agreeing_on_the_label_is_enough():
    specs = [{"f_body": 0.9}] * 7 + [{"f_body": 0.0}]
    v = fold_replicas(replicas(specs), CFG, unanimous=False, agreement_min=7)
    assert v.viable.tolist() == [True]
    assert MODES[int(v.label[0])] == "crawl"


def test_the_reported_label_is_the_modal_one():
    specs = [{"f_body": 0.9}] * 6 + [{"f_body": 0.0}] * 2
    v = fold_replicas(replicas(specs), CFG, unanimous=True, agreement_min=6)
    assert MODES[int(v.label[0])] == "crawl"


# --------------------------------------------------------------------------- #
# Clause reporting
# --------------------------------------------------------------------------- #


def test_clause_rates_separate_a_progress_collapse_from_a_label_collapse():
    # "Feasibility collapsed" and "feasibility collapsed ON THE LABEL CLAUSE"
    # call for opposite fixes, and v2/v3 both spent iterations on the wrong one.
    stalled = fold_replicas(
        replicas([{"dx": 0.0}] * 4), CFG, unanimous=True, agreement_min=4
    )
    assert stalled.clause_rates["progress"] == 0.0
    assert stalled.clause_rates["constant_label"] == 1.0

    moving = fold_replicas(replicas([{}] * 4), CFG, unanimous=True, agreement_min=4)
    assert moving.clause_rates["progress"] == 1.0


def test_the_impact_clause_is_reported_even_when_no_cap_is_configured():
    v = fold_replicas(replicas([{}] * 4), CFG, unanimous=True, agreement_min=4)
    assert v.clause_rates["impact"] == 1.0


# --------------------------------------------------------------------------- #
# Insertion routing
# --------------------------------------------------------------------------- #


def test_each_candidate_goes_to_the_sub_archive_its_label_names():
    import torch
    from qd.hierarchy import ModeArchives

    archives = ModeArchives(solution_dim=4, seed=0)
    genomes = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    verdict = Verdict(
        viable=np.array([True, True]),
        fitness=np.array([2.0, 0.4]),
        label=np.array([MODE_INDEX["walk"], MODE_INDEX["crawl"]]),
        agreement=np.array([8, 8]),
        axes={
            "torso_height_mean": np.array([0.119, 0.05]),
            "joint_speed": np.array([1.5, 1.5]),
            "yaw_rate": np.array([0.0, 0.0]),
        },
        clause_rates={},
    )
    rates = insert(archives, genomes, verdict)
    assert archives.occupancy()["walk"] == 1
    assert archives.occupancy()["crawl"] == 1
    assert rates["per_mode"]["walk"]["inserted"] == 1
    assert rates["feasible_rate"] == 1.0


def test_non_viable_candidates_are_offered_to_nothing():
    import torch
    from qd.hierarchy import ModeArchives

    archives = ModeArchives(solution_dim=4, seed=0)
    verdict = Verdict(
        viable=np.array([False]),
        fitness=np.array([2.0]),
        label=np.array([MODE_INDEX["walk"]]),
        agreement=np.array([8]),
        axes=axes(1),
        clause_rates={},
    )
    rates = insert(archives, torch.zeros(1, 4), verdict)
    assert len(archives) == 0
    assert rates["insertion_rate"] == 0.0


# --------------------------------------------------------------------------- #
# The critic reward
# --------------------------------------------------------------------------- #


def test_the_mode_critic_reward_does_not_pay_for_being_upright():
    # A critic that pays an alive bonus and an `upright` term teaches PG
    # variation to stand every crawl up. Both are gone.
    cfg = ModeRewardCfg()
    assert not hasattr(cfg, "upright_weight")
    assert not hasattr(cfg, "alive_bonus")
    assert cfg.vel_weight == 1.0
    assert cfg.impact_weight > 0.0
