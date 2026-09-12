"""The v5 loop's CPU-side pieces: the reservoir and the data-driven search centroids."""

from __future__ import annotations

import numpy as np
from qd.behaviour import ReplicaVerdict
from qd.behaviour_space import TrainCfg, fit_space
from qd.latent_archive import LatentArchive
from qd.pga.run_aurora import Args, Reservoir, search_centroids


def _space(d=6):
    x = np.random.default_rng(0).normal(size=(300, d)).astype(np.float32)
    return fit_space(x, tuple(f"f{i}" for i in range(d)), TrainCfg(kind="pca", latent_dim=2))


def _verdict(n, d, viable_frac, rng):
    feats = rng.normal(size=(n, d)).astype(np.float32)
    viable = rng.uniform(size=n) < viable_frac
    return ReplicaVerdict(viable, viable.astype(int) * 5, rng.uniform(size=n), feats, {}), np.stack(
        [feats + 0.01 * i for i in range(3)]
    )


def test_reservoir_tracks_viable_rows_and_groups():
    rng = np.random.default_rng(1)
    res = Reservoir(size=500, rng=rng)
    for _ in range(4):
        v, stack = _verdict(200, 6, 0.3, rng)
        res.add(v, stack)
    assert len(res.features()) <= 600  # deque of blocks, trimmed to ~size
    assert len(res.viable_features()) == len(res.groups)
    assert res.groups[0].shape == (3, 6)


def test_search_centroids_falls_back_then_fits_data():
    rng = np.random.default_rng(2)
    space = _space()
    args = Args(search_centroids=16)
    arc = LatentArchive(rng.normal(size=(4, 2)).astype(np.float32) * 5, solution_dim=1)
    res = Reservoir(size=10_000, rng=rng)
    # too few viable rows -> box CVT fallback, still k centroids
    v, stack = _verdict(20, 6, 0.2, rng)
    res.add(v, stack)
    c = search_centroids(space, arc, res, args)
    assert c.shape == (16, 2)
    # enough viable rows -> k-means on their latents; centroids sit inside the data
    for _ in range(5):
        v, stack = _verdict(200, 6, 0.5, rng)
        res.add(v, stack)
    c = search_centroids(space, arc, res, args)
    z = space.encode(res.viable_features())
    assert c.shape == (16, 2)
    assert (c.min(0) >= z.min(0) - 1e-3).all() and (c.max(0) <= z.max(0) + 1e-3).all()
    # and the archive re-encodes onto them without error
    arc.add(np.zeros((3, 1)), np.arange(3.0), rng.normal(size=(3, 6)).astype(np.float32), space.encode(rng.normal(size=(3, 6)).astype(np.float32)))
    moved = arc.re_encode(space.encode, c)
    assert moved["before"] >= 1 and arc.num_cells == 16
