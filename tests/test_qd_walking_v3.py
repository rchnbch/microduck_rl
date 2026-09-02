"""CPU regression tests for walking-v3's insertion rule.

The rule is "eight replicas per candidate, unanimous survival, median fitness",
and there are two ways to get it silently wrong. Both are locked here.

* **Alignment.** Each replica now puts a candidate in a different *world*,
  because on this simulator a world index carries a persistent bias and
  replicating in place samples a fraction of the noise verification measures.
  Permuting means un-permuting, and un-permuting with the forward index instead
  of its inverse produces a run that looks perfectly healthy while every
  fitness is attached to the wrong genome.
* **Combination.** Survival unanimous, fitness median — not mean, which one
  catastrophic replica drags around, and not max, which is the luck-ranking the
  whole rule exists to remove.
"""

import numpy as np
import pytest
import torch

from qd.pga.run_pga_me import combine_replicas, evaluate_replicas


class _SlotHarness:
    """A fake harness whose result depends on the WORLD, not on the genome.

    This is the pathology being corrected, in its purest form: world *k* always
    reports ``k``. A run that replicates in place therefore learns nothing new
    from its extra replicas, and a run that permutes sees the whole spread.
    """

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def rollout(self, block):
        self.calls += 1
        slot = np.arange(self.n, dtype=np.float64)
        genome = block[:, 0].cpu().numpy().astype(np.float64)
        # Fitness is the genome's own id plus its slot's bias, so a
        # misalignment between the two is visible in the output.
        fitness = genome + 0.001 * slot
        measures = np.stack([genome, slot], axis=-1)
        info = {
            "displacement": fitness.copy(),
            "fell": slot >= self.n - 1,  # only the last world ever falls
            "alive_steps": np.full(self.n, 350),
            "survival_fraction": np.ones(self.n),
        }
        return fitness, measures, info, None


def _block(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).reshape(n, 1)


def test_permuted_replicas_stay_aligned_with_their_genomes():
    """Row i of the result must still be the genome on row i of the block."""
    n = 16
    h = _SlotHarness(n)
    gen = torch.Generator().manual_seed(0)
    fitness, measures, _ = evaluate_replicas(h, _block(n), 8, gen, permute=True)
    # measures[:, 0] is the genome id the harness saw; it must match the row.
    np.testing.assert_allclose(measures[:, 0], np.arange(n))
    # Fitness is genome id + a small slot term, so it stays within one of the id.
    assert np.all(np.abs(fitness - np.arange(n)) < 0.02)


def test_permuting_actually_visits_different_worlds():
    """Without this the extra replicas are seven repeats of the same number."""
    n = 16
    gen = torch.Generator().manual_seed(0)
    _, in_place, _ = evaluate_replicas(_SlotHarness(n), _block(n), 8, gen, permute=False)
    _, permuted, _ = evaluate_replicas(_SlotHarness(n), _block(n), 8, gen, permute=True)
    # measures[:, 1] is the slot; its median over replicas is the slot itself
    # when nothing moves, and drifts toward the middle when everything does.
    np.testing.assert_allclose(in_place[:, 1], np.arange(n))
    assert not np.allclose(permuted[:, 1], np.arange(n))


def test_a_genome_that_falls_in_any_replica_is_not_a_survivor():
    """Unanimity is the point: the fake harness fails whichever genome lands in
    the last world, so over eight permuted replicas most genomes get caught."""
    n = 8
    gen = torch.Generator().manual_seed(3)
    _, _, info = evaluate_replicas(_SlotHarness(n), _block(n), 8, gen, permute=True)
    assert info["fell"].sum() > 1, "permuted replicas should catch several genomes"

    gen = torch.Generator().manual_seed(3)
    _, _, in_place = evaluate_replicas(_SlotHarness(n), _block(n), 8, gen, permute=False)
    assert int(in_place["fell"].sum()) == 1, "replicating in place catches only one"


def test_every_replica_is_banked():
    n, reps = 4, 8
    seen = []
    gen = torch.Generator().manual_seed(0)
    evaluate_replicas(
        _SlotHarness(n), _block(n), reps, gen, permute=True, bank=seen.append
    )
    assert len(seen) == reps


def test_fitness_is_the_median_not_the_mean_or_the_max():
    runs = [
        (np.array([1.0]), np.array([[0.5, 0.5]]), _info(True, 1.0)),
        (np.array([2.0]), np.array([[0.5, 0.5]]), _info(True, 2.0)),
        (np.array([9.0]), np.array([[0.5, 0.5]]), _info(True, 9.0)),
    ]
    fitness, _, info = combine_replicas(runs)
    assert fitness[0] == pytest.approx(2.0)
    assert info["displacement"][0] == pytest.approx(2.0)


def test_one_fallen_replica_makes_the_candidate_fallen():
    runs = [
        (np.array([1.0]), np.array([[0.5, 0.5]]), _info(True, 1.0)),
        (np.array([1.0]), np.array([[0.5, 0.5]]), _info(False, 1.0)),
        (np.array([1.0]), np.array([[0.5, 0.5]]), _info(True, 1.0)),
    ]
    _, _, info = combine_replicas(runs)
    assert bool(info["fell"][0])


def _info(upright: bool, displacement: float) -> dict:
    return {
        "displacement": np.array([displacement]),
        "fell": np.array([not upright]),
        "alive_steps": np.array([350 if upright else 10]),
        "survival_fraction": np.array([1.0 if upright else 0.03]),
    }
