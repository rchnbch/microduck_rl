"""The latent CVT archive: insertion, re-encoding, and the parent-sampling
guarantee j017 asks for — no region that holds an elite can be starved.
"""

from __future__ import annotations

import numpy as np
import pytest
from qd.latent_archive import LatentArchive

CENTROIDS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32)


def make(n_elites: int, seed: int = 0) -> LatentArchive:
    rng = np.random.default_rng(seed)
    arc = LatentArchive(CENTROIDS, solution_dim=3)
    g = rng.normal(size=(n_elites, 3)).astype(np.float32)
    f = rng.normal(size=(n_elites, 4)).astype(np.float32)
    z = CENTROIDS[rng.integers(0, len(CENTROIDS), size=n_elites)] + rng.normal(0, 0.05, size=(n_elites, 2))
    arc.add(g, rng.uniform(size=n_elites), f, z.astype(np.float32))
    return arc


def test_add_keeps_best_per_cell_within_and_across_batches():
    arc = LatentArchive(CENTROIDS, 2)
    z = np.array([[0.01, 0.0], [0.0, 0.02]], dtype=np.float32)
    status = arc.add(np.zeros((2, 2)), np.array([0.5, 0.9]), np.zeros((2, 1)), z)
    assert sorted(status.tolist()) == [0, 1]
    assert len(arc) == 1 and arc.cells[0].fitness == pytest.approx(0.9)
    status = arc.add(np.ones((1, 2)), np.array([0.7]), np.zeros((1, 1)), z[:1])
    assert status.tolist() == [0]
    status = arc.add(np.ones((1, 2)), np.array([1.1]), np.zeros((1, 1)), z[:1])
    assert status.tolist() == [2] and arc.cells[0].fitness == pytest.approx(1.1)


def test_every_elite_has_positive_parent_weight_and_lonely_ones_more():
    """The no-starvation guarantee: weights are 1/(1+neighbours), never zero."""
    arc = LatentArchive(CENTROIDS, 1)
    # four elites within 1.5 of each other in the unit square, one alone at (5, 5)
    z = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]], dtype=np.float32)
    arc.add(np.arange(5.0).reshape(5, 1), np.ones(5), np.zeros((5, 1)), z)
    w, _ = arc.parent_weights(radius=1.5)
    assert (w > 0).all()
    assert w.sum() == pytest.approx(1.0)
    # the isolated elite carries the single largest weight
    assert np.argmax(w) == list(arc.cells).index(4)
    # sampled parents reach it
    rng = np.random.default_rng(0)
    parents = arc.sample_parents(2000, rng, radius=1.5)
    assert np.sum(parents[:, 0] == 4.0) > 200
    # uniform ablation: also reaches everyone
    parents = arc.sample_parents(2000, rng, radius=None)
    assert len(np.unique(parents[:, 0])) == 5


def test_re_encode_moves_elites_and_reports_losses():
    arc = make(40)
    before = len(arc)
    fitness_before = [e.fitness for e in arc.cells.values()]
    # collapse everything to one point: every elite lands in one cell, best survives
    moved = arc.re_encode(lambda f: np.zeros((len(f), 2), np.float32))
    assert moved["before"] == before and moved["after"] == 1 and moved["lost"] == before - 1
    assert len(arc) == 1
    survivor = next(iter(arc.cells.values()))
    assert survivor.fitness == max(fitness_before)


def test_re_encode_with_new_centroids_changes_geometry():
    arc = make(30)
    new_c = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    arc.re_encode(lambda f: np.zeros((len(f), 2), np.float32) + 100.0, centroids=new_c)
    assert arc.num_cells == 2 and list(arc.cells) == [1]


def test_retest_evicts_below_running_rate():
    arc = make(10)
    cell, elite = next(iter(arc.cells.items()))
    outcome = arc.record_retest([(cell, elite.genome, False)] * 1, min_pass_rate=0.6)
    assert outcome.evicted == 1 and cell not in arc.cells
    cell2, elite2 = next(iter(arc.cells.items()))
    arc.record_retest([(cell2, elite2.genome, True), (cell2, elite2.genome, True), (cell2, elite2.genome, False)], 0.6)
    assert cell2 in arc.cells and arc.cells[cell2].record.rate == pytest.approx(2 / 3)
    # a stale genome (cell changed hands) is ignored
    arc.record_retest([(cell2, elite2.genome + 1.0, False)], 0.6)
    assert cell2 in arc.cells


def test_save_load_round_trip(tmp_path):
    arc = make(12)
    arc.record_retest([(c, e.genome, True) for c, e in list(arc.cells.items())[:3]], 0.5)
    path = arc.save(tmp_path / "a.npz", {"k": 1})
    back, meta = LatentArchive.load(path)
    assert meta == {"k": 1}
    assert sorted(back.cells) == sorted(arc.cells)
    for c in arc.cells:
        assert np.allclose(back.cells[c].genome, arc.cells[c].genome)
        assert back.cells[c].record.attempts == arc.cells[c].record.attempts
    assert np.allclose(back.centroids, arc.centroids)


def test_entropy_is_one_when_uniform():
    arc = LatentArchive(CENTROIDS, 1)
    z = CENTROIDS[:4]
    arc.add(np.arange(4.0).reshape(4, 1), np.ones(4), np.zeros((4, 1)), z)
    assert arc.parent_weight_entropy(radius=0.01) == pytest.approx(1.0)
    # add a fifth, isolated elite: the distribution is no longer uniform
    arc.add(np.array([[9.0]]), np.ones(1), np.zeros((1, 1)), CENTROIDS[4:5])
    assert arc.parent_weight_entropy(radius=1.01) < 1.0
