"""v5: the label-free gate, the behaviour features and the learned space — CPU.

The cases are the ones v4 argued about, re-asked without a mode name in the
question: a steady walker, a steady crawl, a walker that falls late, and two
clusters a space must separate without being told they are two.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from qd.behaviour import (
    BehaviourStats,
    ViabilityV5Cfg,
    evaluate_viability_v5,
    feature_names,
    fold_replicas_v5,
    support_deltas,
)
from qd.behaviour_space import (
    MIN_MODE_ELITES,
    NOISE_MULTIPLIER,
    BehaviourSpace,
    TrainCfg,
    cluster_modes,
    coverage,
    fit_space,
    latent_bounds,
    make_centroids,
    nearest_centroid,
    replica_noise,
)
from qd.modes import ModeStats, WindowCfg

DT = 0.02
GEOMS = 4
FOOT_COLS = (0, 1)
HEAD_COLS = (3,)
NJ = 3


def scripted(num_envs, x_of_t, contact_of_t, z_of_t=lambda t: 0.12, az=0.0):
    """Drive ModeStats + BehaviourStats through one scripted episode."""
    w = WindowCfg(control_dt=DT)
    ms = ModeStats(num_envs, "cpu", w, GEOMS, FOOT_COLS, HEAD_COLS, contact_force_n=0.0)
    bs = BehaviourStats(num_envs, "cpu", w, NJ, tuple(f"g{i}" for i in range(GEOMS)))
    pos = torch.zeros(num_envs, 3)
    pos[:, 0] = x_of_t(0.0)
    pos[:, 2] = z_of_t(0.0)
    ms.begin(pos)
    bs.begin()
    grav = torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1)
    for step in range(w.episode_steps):
        t = (step + 1) * DT
        pos = torch.zeros(num_envs, 3)
        pos[:, 0] = x_of_t(t)
        pos[:, 2] = z_of_t(t)
        contact = torch.zeros(num_envs, GEOMS, dtype=torch.bool)
        for c in contact_of_t(t):
            contact[:, c] = True
        omega = torch.zeros(num_envs, 3)
        v = torch.zeros(num_envs, 3)
        q = torch.zeros(num_envs, NJ)
        q[:, 0] = np.sin(2 * np.pi * 2.0 * t)
        dq = torch.zeros(num_envs, NJ)
        a = torch.full((num_envs,), az)
        ms.update(pos, grav, contact, omega, trunk_az=a)
        bs.update(pos, grav, v, omega, q, dq, contact, None, a)
    return ms.finalize(), bs.finalize()


def test_steady_walker_is_viable_without_a_label():
    mf, bf = scripted(2, lambda t: 0.2 * t, lambda t: [0] if int(t / 0.3) % 2 else [1])
    v = evaluate_viability_v5(mf, ViabilityV5Cfg())
    assert v.viable.all()
    assert v.stationary.all()
    assert bf.vector.shape == (2, len(bf.names))
    assert np.isfinite(bf.vector).all()


def test_steady_crawl_is_viable_without_a_label():
    # belly on the ground the whole time (geom 2 is a body geom), slow progress
    mf, _ = scripted(1, lambda t: 0.03 * t, lambda t: [0, 1, 2])
    v = evaluate_viability_v5(mf, ViabilityV5Cfg())
    assert v.viable.all()


def test_late_fall_fails_stationarity_not_a_label_clause():
    # walks on feet, then the trunk lands at 6.0 s and progress stops
    def contacts(t):
        return [0, 1] if t < 6.0 else [0, 1, 2]

    def x(t):
        return 0.2 * min(t, 6.0)

    mf, _ = scripted(1, x, contacts)
    v = evaluate_viability_v5(mf, ViabilityV5Cfg(delta_max=0.15))
    # the last window [5,7] has f_body = 0.5 against 0 in [4,6]
    assert not v.stationary.all()
    assert v.max_delta[0] == pytest.approx(0.5, abs=0.05)
    assert not v.viable.all()


def test_support_deltas_ignores_exempt_window():
    mf, _ = scripted(1, lambda t: 0.2 * t, lambda t: [2] if t < 0.5 else [0, 1])
    # a transition entirely inside the first (exempt) second is not charged
    assert support_deltas(mf, 1)[0] == pytest.approx(0.0, abs=1e-6)
    assert support_deltas(mf, 0)[0] > 0.2


def test_feature_vector_is_phase_invariant():
    """Two identical gaits offset in phase must give (nearly) the same features."""

    def make(phase):
        return scripted(
            1,
            lambda t: 0.2 * t,
            lambda t: [0] if int((t + phase) / 0.3) % 2 else [1],
            z_of_t=lambda t: 0.12 + 0.005 * np.sin(2 * np.pi * 3.3 * t + phase),
        )[1].vector

    a, b = make(0.0), make(0.11)
    names = feature_names(NJ, tuple(f"g{i}" for i in range(GEOMS)))
    for name, va, vb in zip(names, a[0], b[0]):
        if name in ("z_dom_freq",):
            assert va == vb
        else:
            assert abs(va - vb) < 0.05, (name, va, vb)


def test_feature_names_match_vector_width():
    _, bf = scripted(1, lambda t: 0.2 * t, lambda t: [0, 1])
    assert len(bf.names) == bf.vector.shape[1]
    assert bf.names == feature_names(NJ, tuple(f"g{i}" for i in range(GEOMS)))


def test_fold_replicas_v5_is_k_of_n_and_median():
    good, _ = scripted(1, lambda t: 0.2 * t, lambda t: [0, 1])
    bad, _ = scripted(1, lambda t: 0.0, lambda t: [0, 1])
    feats = np.ones((1, 4), dtype=np.float32)
    reps = [(good, feats * 1), (good, feats * 2), (bad, feats * 100), (good, feats * 3), (good, feats * 4)]
    cfg = ViabilityV5Cfg()
    v = fold_replicas_v5(reps, cfg, viable_min=4)
    assert v.viable.all() and v.viable_count[0] == 4
    assert v.features[0, 0] == pytest.approx(3.0)  # median of 1,2,100,3,4
    v2 = fold_replicas_v5(reps, cfg, viable_min=5)
    assert not v2.viable.any()


# --------------------------------------------------------------------------- #
# the learned space
# --------------------------------------------------------------------------- #


def two_clusters(n=200, d=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.05, size=(n, d)) + np.r_[np.ones(d // 2), np.zeros(d - d // 2)]
    b = rng.normal(0.0, 0.05, size=(n, d)) + np.r_[np.zeros(d // 2), np.ones(d - d // 2)]
    x = np.concatenate([a, b]).astype(np.float32)
    ref = np.r_[np.zeros(n, int), np.ones(n, int)]
    return x, ref


@pytest.mark.parametrize("kind", ["pca", "ae"])
def test_space_recovers_two_clusters_unsupervised(kind, tmp_path):
    x, ref = two_clusters()
    names = tuple(f"f{i}" for i in range(x.shape[1]))
    space = fit_space(x, names, TrainCfg(kind=kind, latent_dim=2, epochs=60, patience=20))
    # replica groups: each genome's replicas scatter like the within-cluster sd
    rng = np.random.default_rng(5)
    groups = [
        x[i] + rng.normal(0, 0.05, size=(4, x.shape[1])).astype(np.float32)
        for i in range(0, len(x), 4)
    ]
    space.calibrate_to_noise(groups)
    z = space.encode(x)
    assert z.shape == (len(x), 2)
    zr = np.stack([space.encode(np.stack([g[r] for g in groups])) for r in range(4)])
    noise, _ = replica_noise(zr)
    # in noise-calibrated units the RMS replica distance over L axes is ~sqrt(L)
    assert 0.5 < noise < 3.0
    bounds = latent_bounds(z)
    report = cluster_modes(z, noise, bounds)
    assert report.n_modes == 2
    for c in (0, 1):
        members = ref[report.labels == c]
        assert len(np.unique(members)) == 1  # pure
    # persistence round-trip
    path = space.save(tmp_path / f"space_{kind}.npz")
    back = BehaviourSpace.load(path)
    assert np.allclose(back.encode(x), z, atol=1e-5)


def test_cluster_modes_size_floor_and_noise_units():
    rng = np.random.default_rng(1)
    big = rng.normal(0, 0.01, size=(20, 3))
    small = rng.normal(0, 0.01, size=(MIN_MODE_ELITES - 1, 3)) + 10.0
    lone = np.array([[100.0, 0, 0]])
    z = np.concatenate([big, small, lone])
    report = cluster_modes(z, noise=0.05, bounds=latent_bounds(z))
    assert report.eps == pytest.approx(NOISE_MULTIPLIER * 0.05)
    assert report.n_modes == 1
    assert report.n_fragments == MIN_MODE_ELITES - 1
    assert report.n_noise == 1
    assert report.modes[0]["elites"] == 20
    assert report.modes[0]["axis_bins_occupied"][0] >= 1


def test_centroids_fixed_and_coverage_counts_unique_cells():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    c1 = make_centroids(16, bounds, seed=3, samples=2000)
    c2 = make_centroids(16, bounds, seed=3, samples=2000)
    assert np.allclose(c1, c2)
    z = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]])
    n, idx = coverage(z, c1)
    assert n == 2 and len(idx) == 2
    assert nearest_centroid(z, c1)[0] == nearest_centroid(z, c1)[1]


def test_dead_features_are_zeroed_not_exploded():
    x = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    x[:, 1] = 0.5  # never varies
    space = fit_space(x, ("a", "b", "c"), TrainCfg(kind="pca", latent_dim=2))
    assert space.train_info["dead_features"] == ["b"]
    assert np.isfinite(space.encode(x)).all()
    assert space.standardise(x)[:, 1].max() == 0.0
