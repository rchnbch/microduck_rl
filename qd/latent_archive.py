"""A CVT archive in a learned latent space, with parents drawn from everywhere.

v4's failure was structural, not a tuning miss: ``ModeArchives.sample_parents``
budgeted parents across the *non-empty* modes, so a mode with no elites drew
no parents, forever, and the only way to reach one was by accident. This
archive has no modes to budget between. Its cells are the Voronoi regions of a
fixed centroid set in latent space, its parents are drawn from the whole elite
set, and the only bias on that draw runs the *other* way:

**Sparsity-weighted parent selection** (:meth:`LatentArchive.parent_weights`).
Each elite is weighted ``1 / (1 + n_i)``, ``n_i`` being the number of other
elites within ``radius`` of it in latent space. An elite in a crowded region is
sampled less; an elite alone at the frontier is sampled most. Every weight is
strictly positive, so no elite — and no region that holds one — can be starved
(the guarantee j017 asks for is the ``1 +`` in the denominator, and the absence
of any per-region budget). It is the explicit novelty pressure: fitness alone
measurably found zero new modes in v4.

Re-encoding (:meth:`re_encode`) is what makes an AURORA-style retrain cheap:
the archive keeps every elite's raw feature vector, so a new encoder means
"encode the features again, re-place, keep the best per cell" and never a
re-simulation.

Incumbent re-testing with eviction is carried over from v4 unchanged in
spirit: the running pass rate belongs to the elite, and the cell's occupant is
followed by genome fingerprint.

Nothing here imports mjlab or torch. numpy only, testable on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qd.behaviour_space import nearest_centroid
from qd.hierarchy import PassRecord, RetestOutcome, _fingerprint


@dataclass
class Elite:
    genome: np.ndarray
    fitness: float
    features: np.ndarray
    latent: np.ndarray
    record: PassRecord = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.record is None:
            self.record = PassRecord(fingerprint=_fingerprint(self.genome))


class LatentArchive:
    """``dict[cell, Elite]`` over a fixed centroid set."""

    def __init__(self, centroids: np.ndarray, solution_dim: int):
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.solution_dim = int(solution_dim)
        self.cells: dict[int, Elite] = {}

    # -- geometry ------------------------------------------------------------ #

    @property
    def num_cells(self) -> int:
        return len(self.centroids)

    @property
    def latent_dim(self) -> int:
        return self.centroids.shape[1]

    def cell_index(self, latents: np.ndarray) -> np.ndarray:
        return nearest_centroid(np.atleast_2d(latents), self.centroids)

    def centroid_spacing(self) -> float:
        """Median nearest-neighbour distance between centroids."""
        c = self.centroids
        if len(c) < 2:
            return 1.0
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        d[np.eye(len(c), dtype=bool)] = np.inf
        return float(np.median(d.min(axis=1)))

    # -- insertion ----------------------------------------------------------- #

    def add(
        self,
        genomes: np.ndarray,
        fitness: np.ndarray,
        features: np.ndarray,
        latents: np.ndarray,
    ) -> np.ndarray:
        """Offer candidates; returns status per candidate (0 rejected, 1 new cell, 2 improved).

        Ties within one batch are resolved by fitness, so a batch offering two
        candidates to an empty cell keeps the better one."""
        genomes = np.asarray(genomes, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float64)
        features = np.asarray(features, dtype=np.float32)
        latents = np.asarray(latents, dtype=np.float32)
        n = len(genomes)
        status = np.zeros(n, dtype=int)
        if n == 0:
            return status
        cells = self.cell_index(latents)
        for i in np.argsort(-fitness):  # best first, so within-batch ties go to the best
            c = int(cells[i])
            cur = self.cells.get(c)
            if cur is None:
                self.cells[c] = Elite(genomes[i], float(fitness[i]), features[i], latents[i])
                status[i] = 1
            elif fitness[i] > cur.fitness:
                self.cells[c] = Elite(genomes[i], float(fitness[i]), features[i], latents[i])
                status[i] = 2
        return status

    def re_encode(self, encode, centroids: np.ndarray | None = None) -> dict:
        """Re-place every elite under a new encoder (and optionally new centroids).

        ``encode`` maps ``(N, D)`` raw features to ``(N, L)`` latents. Elites
        that now share a cell are resolved by fitness; the losers are gone,
        which is the price AURORA pays for a space that moves, and the count is
        returned so the log shows it."""
        elites = list(self.cells.values())
        if centroids is not None:
            self.centroids = np.asarray(centroids, dtype=np.float32)
        self.cells = {}
        if not elites:
            return {"before": 0, "after": 0, "lost": 0}
        feats = np.stack([e.features for e in elites])
        z = encode(feats)
        cells = self.cell_index(z)
        order = np.argsort([-e.fitness for e in elites])
        for i in order:
            c = int(cells[i])
            if c not in self.cells:
                e = elites[i]
                e.latent = z[i]
                self.cells[c] = e
        return {"before": len(elites), "after": len(self.cells), "lost": len(elites) - len(self.cells)}

    # -- parents ------------------------------------------------------------- #

    def parent_weights(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """``(weights, latents)`` over elites in ``self.cells`` order.

        Weight ``1 / (1 + n_i)`` with ``n_i`` the number of OTHER elites within
        ``radius``. Strictly positive for every elite, by construction."""
        elites = list(self.cells.values())
        if not elites:
            return np.zeros(0), np.zeros((0, self.latent_dim))
        z = np.stack([e.latent for e in elites])
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        within = (d2 <= radius * radius).sum(axis=1) - 1
        w = 1.0 / (1.0 + within)
        return w / w.sum(), z

    def sample_parents(
        self, n: int, rng: np.random.Generator, radius: float | None = None
    ) -> np.ndarray:
        """``(n, solution_dim)`` parents from the whole archive, sparsity-weighted.

        ``radius=None`` (or 0) is uniform sampling — every elite, every region,
        equal odds — kept for the ablation. There is no code path here that
        conditions on which region a parent comes from."""
        if n <= 0 or not self.cells:
            return np.zeros((0, self.solution_dim), dtype=np.float32)
        elites = list(self.cells.values())
        if radius:
            p, _z = self.parent_weights(radius)
        else:
            p = np.full(len(elites), 1.0 / len(elites))
        pick = rng.choice(len(elites), size=n, replace=True, p=p)
        return np.stack([elites[i].genome for i in pick])

    def parent_weight_entropy(self, radius: float) -> float:
        """Normalised entropy of the parent distribution (1 = uniform)."""
        p, _ = self.parent_weights(radius)
        if len(p) <= 1:
            return 1.0
        h = -np.sum(p * np.log(np.maximum(p, 1e-12)))
        return float(h / np.log(len(p)))

    # -- incumbent re-testing --------------------------------------------- #

    def sample_incumbents(
        self, fraction: float, rng: np.random.Generator, capacity: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        pool = list(self.cells.items())
        if not pool:
            return []
        k = max(1, round(fraction * len(pool)))
        if capacity:
            k = max(k, min(capacity, len(pool)))
        pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [(c, e.genome) for c, e in (pool[i] for i in pick)]

    def record_retest(
        self, results: list[tuple[int, np.ndarray, bool]], min_pass_rate: float
    ) -> RetestOutcome:
        outcome = RetestOutcome(tested=len(results))
        evict = set()
        passes = 0
        for cell, genome, passed in results:
            e = self.cells.get(cell)
            if e is None or e.record.fingerprint != _fingerprint(genome):
                continue
            e.record.attempts += 1
            e.record.passes += int(passed)
            passes += int(passed)
            if e.record.rate < min_pass_rate:
                evict.add(cell)
        outcome.pass_rate = passes / len(results) if results else float("nan")
        for c in evict:
            self.cells.pop(c, None)
        outcome.evicted = len(evict)
        outcome.evicted_by_mode = {"all": len(evict)}
        return outcome

    def running_pass_rate(self) -> float:
        tested = [e.record.rate for e in self.cells.values() if e.record.attempts]
        return float(np.mean(tested)) if tested else float("nan")

    # -- reporting ----------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.cells)

    def data(self) -> dict[str, np.ndarray]:
        if not self.cells:
            return {
                "solution": np.zeros((0, self.solution_dim), np.float32),
                "objective": np.zeros(0),
                "measures": np.zeros((0, self.latent_dim), np.float32),
                "features": np.zeros((0, 0), np.float32),
                "index": np.zeros(0, int),
            }
        cells = sorted(self.cells)
        es = [self.cells[c] for c in cells]
        return {
            "solution": np.stack([e.genome for e in es]),
            "objective": np.array([e.fitness for e in es]),
            "measures": np.stack([e.latent for e in es]),
            "features": np.stack([e.features for e in es]),
            "index": np.array(cells),
        }

    def stats(self) -> dict[str, float]:
        if not self.cells:
            return {"size": 0, "coverage": 0.0, "max_fitness": float("nan"), "mean_fitness": float("nan")}
        f = np.array([e.fitness for e in self.cells.values()])
        return {
            "size": len(self.cells),
            "coverage": len(self.cells) / self.num_cells,
            "max_fitness": float(f.max()),
            "mean_fitness": float(f.mean()),
        }

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.data()
        payload = dict(d)
        payload["centroids"] = self.centroids
        payload["retest_passes"] = np.array(
            [self.cells[c].record.passes for c in d["index"]], dtype=int
        )
        payload["retest_attempts"] = np.array(
            [self.cells[c].record.attempts for c in d["index"]], dtype=int
        )
        payload["meta_json"] = np.array(json.dumps(meta or {}))
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple[LatentArchive, dict]:
        with np.load(Path(path), allow_pickle=False) as f:
            arc = cls(f["centroids"], f["solution"].shape[1])
            meta = json.loads(str(f["meta_json"]))
            for i, c in enumerate(f["index"]):
                e = Elite(f["solution"][i], float(f["objective"][i]), f["features"][i], f["measures"][i])
                if "retest_passes" in f.files:
                    e.record.passes = int(f["retest_passes"][i])
                    e.record.attempts = int(f["retest_attempts"][i])
                arc.cells[int(c)] = e
        return arc, meta
