"""The per-mode archive, its parent budget, and the incumbent re-test.

The two mechanisms tested here are the ones a single archive cannot express,
and both were named as v3's open problems before they were built: a mature mode
monopolising variation, and the winner's curse on the *survival* predicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from qd.hierarchy import (
    V3_WALK_DESCRIPTOR,
    ModeArchiveCfg,
    ModeArchives,
    default_mode_cfgs,
)

DIM = 6


def axes_for(height, speed, n=None):
    """The two axes every default mode grid reads, as GaitStats would hand them."""
    height = np.atleast_1d(np.asarray(height, dtype=float))
    speed = np.atleast_1d(np.asarray(speed, dtype=float))
    n = n or len(height)
    return {
        "torso_height_mean": np.broadcast_to(height, (n,)).copy(),
        "joint_speed": np.broadcast_to(speed, (n,)).copy(),
        "yaw_rate": np.zeros(n),
    }


def make(seed=0):
    return ModeArchives(solution_dim=DIM, seed=seed)


def genomes(n, rng):
    return rng.normal(size=(n, DIM))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_walk_keeps_v3s_measured_axes_byte_for_byte():
    cfgs = default_mode_cfgs()
    assert cfgs["walk"].descriptor == V3_WALK_DESCRIPTOR
    assert cfgs["walk"].descriptor.x_range == (0.11667, 0.12184)
    assert cfgs["walk"].descriptor.y_range == (0.89228, 2.50831)


def test_every_mode_has_a_sub_archive_including_the_unseeded_one():
    a = make()
    assert set(a.archives) == {"walk", "crawl", "roll", "hop", "other"}


def test_an_unknown_mode_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unknown modes"):
        ModeArchives(DIM, {"gallop": ModeArchiveCfg(V3_WALK_DESCRIPTOR)})


def test_each_mode_bins_on_its_own_axes():
    a = make()
    rng = np.random.default_rng(0)
    # The same descriptor values land in different cells in walk and crawl,
    # because the ranges differ by an order of magnitude.
    axes = axes_for(0.05, 2.0, n=1)
    a.add("crawl", genomes(1, rng), np.array([1.0]), axes)
    assert len(a.archives["crawl"]) == 1
    assert len(a.archives["walk"]) == 0


def test_modes_never_compete_for_a_cell():
    # A 2 m walker and a 0.4 m crawl with overlapping descriptors both survive:
    # separate archives are what make "best crawl" a meaningful cell.
    a = make()
    rng = np.random.default_rng(1)
    a.add("walk", genomes(1, rng), np.array([2.0]), axes_for(0.119, 1.5, n=1))
    a.add("crawl", genomes(1, rng), np.array([0.4]), axes_for(0.05, 1.5, n=1))
    assert a.occupancy()["walk"] == 1
    assert a.occupancy()["crawl"] == 1


# --------------------------------------------------------------------------- #
# Parent budget
# --------------------------------------------------------------------------- #


def test_a_three_hundred_walker_archive_does_not_own_the_offspring():
    a = make()
    rng = np.random.default_rng(2)
    h = np.linspace(0.1167, 0.1218, 300)
    s = np.linspace(0.9, 2.5, 300)
    a.add("walk", genomes(300, rng), np.linspace(1, 2, 300), axes_for(h, s))
    a.add("crawl", genomes(5, rng), np.linspace(0.3, 0.4, 5),
          axes_for(np.linspace(0.03, 0.08, 5), np.linspace(1, 3, 5)))
    budget = a.parent_budget(100)
    assert budget == {"crawl": 50, "walk": 50}


def test_empty_modes_get_no_budget():
    a = make()
    rng = np.random.default_rng(3)
    a.add("walk", genomes(4, rng), np.ones(4), axes_for(np.linspace(0.117, 0.121, 4), np.linspace(1, 2, 4)))
    assert set(a.parent_budget(10)) == {"walk"}
    assert a.parent_budget(10)["walk"] == 10


def test_sample_parents_returns_the_budgeted_count():
    a = make()
    rng = np.random.default_rng(4)
    a.add("walk", genomes(10, rng), np.ones(10), axes_for(np.linspace(0.117, 0.121, 10), np.linspace(1, 2, 10)))
    a.add("crawl", genomes(3, rng), np.ones(3), axes_for(np.linspace(0.03, 0.08, 3), np.linspace(1, 3, 3)))
    parents = a.sample_parents(21, np.random.default_rng(5))
    assert parents.shape == (21, DIM)


def test_sampling_an_empty_archive_is_empty_not_an_error():
    assert make().sample_parents(10, np.random.default_rng(0)).shape == (0, DIM)


# --------------------------------------------------------------------------- #
# Incumbent re-testing
# --------------------------------------------------------------------------- #


def _fill(a, n=10, seed=6):
    rng = np.random.default_rng(seed)
    g = genomes(n, rng)
    a.add("walk", g, np.ones(n), axes_for(np.linspace(0.117, 0.1215, n), np.linspace(1, 2.4, n)))
    return g


def test_a_single_failed_retest_does_not_evict_a_robust_elite():
    a = make()
    _fill(a)
    sample = a.sample_incumbents(1.0, np.random.default_rng(7))
    # Seven passes then one failure: running rate 7/8, still at the bar.
    for _ in range(7):
        a.record_retest([(m, c, g, True) for m, c, g in sample], min_pass_rate=0.875)
    out = a.record_retest([(m, c, g, False) for m, c, g in sample], min_pass_rate=0.875)
    assert out.evicted == 0
    assert out.pass_rate == 0.0


def test_an_elite_whose_running_rate_falls_below_the_gate_is_evicted():
    a = make()
    _fill(a)
    before = len(a)
    sample = a.sample_incumbents(1.0, np.random.default_rng(8))
    out = a.record_retest([(m, c, g, False) for m, c, g in sample], min_pass_rate=0.875)
    assert out.evicted == before
    assert len(a) == 0


def test_eviction_only_removes_the_failing_cells():
    a = make()
    _fill(a, n=10)
    sample = a.sample_incumbents(1.0, np.random.default_rng(9))
    verdicts = [(m, c, g, i % 2 == 0) for i, (m, c, g) in enumerate(sample)]
    out = a.record_retest(verdicts, min_pass_rate=0.875)
    assert out.evicted == sum(1 for _m, _c, _g, ok in verdicts if not ok)
    assert len(a) == len(sample) - out.evicted


def test_the_pass_record_follows_the_elite_not_the_cell():
    # A cell that changes hands must not inherit the previous occupant's record,
    # or a fresh elite starts life already condemned (or already excused).
    a = make()
    rng = np.random.default_rng(10)
    axes = axes_for(0.119, 1.5, n=1)
    first = genomes(1, rng)
    a.add("walk", first, np.array([1.0]), axes)
    cell = next(iter(a.records["walk"]))
    a.record_retest([("walk", cell, first[0], False)], min_pass_rate=0.0)
    assert a.records["walk"][cell].attempts == 1

    better = genomes(1, rng)
    a.add("walk", better, np.array([2.0]), axes)
    assert a.records["walk"][cell].attempts == 0, "record survived a cell handover"


def test_a_retest_verdict_for_a_replaced_elite_is_discarded():
    a = make()
    rng = np.random.default_rng(11)
    axes = axes_for(0.119, 1.5, n=1)
    stale = genomes(1, rng)
    a.add("walk", stale, np.array([1.0]), axes)
    cell = next(iter(a.records["walk"]))
    a.add("walk", genomes(1, rng), np.array([2.0]), axes)
    out = a.record_retest([("walk", cell, stale[0], False)], min_pass_rate=0.875)
    assert out.evicted == 0
    assert len(a) == 1


def test_pass_rates_are_nan_before_anything_is_retested():
    a = make()
    _fill(a)
    assert np.isnan(a.pass_rates()["walk"])


def test_sample_incumbents_takes_a_tenth_across_modes():
    a = make()
    _fill(a, n=40)
    rng = np.random.default_rng(12)
    a.add("crawl", genomes(10, rng), np.ones(10),
          axes_for(np.linspace(0.03, 0.08, 10), np.linspace(1, 3, 10)))
    sample = a.sample_incumbents(0.1, np.random.default_rng(13))
    assert len(sample) == round(0.1 * len(a))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_each_mode_saves_a_standalone_archive_carrying_its_own_descriptor(tmp_path):
    from qd.common import load_archive

    a = make()
    _fill(a)
    paths = a.save(tmp_path, {"algorithm": "v4"})
    data = load_archive(paths["walk"])
    assert data["meta"]["mode"] == "walk"
    assert data["meta"]["descriptor_axes"] == ["torso_height_mean", "joint_speed"]
    assert data["meta"]["descriptor_ranges"][0] == [0.11667, 0.12184]
