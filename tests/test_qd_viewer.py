"""The gait viewer's CVT (v5) path: cell placement, clip selection, the map.

CPU only — nothing here rolls anything out. The invariants: a verified v5
elite lands in the cell it was *filed* in, clip selection on a CVT tab covers
both postures before it sweeps, the projection is a PCA whose explained
variance sums to one, and the grid path is byte-for-byte what it was.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qd import build_viewer, render_gaits


def _grid_entries(n: int = 30, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(
            {
                "row": i,
                "cell": [int(rng.integers(0, 20)), int(rng.integers(0, 20))],
                "bytes": 100,
                "archived_fitness": float(rng.normal()),
                "displacement_m": float(rng.normal()),
                "survived": bool(i % 7 == 0),
            }
        )
    return out


def _cvt_entries(n_walk: int = 5, n_crawl: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_walk + n_crawl):
        walk = i < n_walk
        centre = np.array([-3.0, -5.0, 2.0, 4.0]) if walk else np.array([7.0, 19.0, -2.0, -8.0])
        out.append(
            {
                "row": i,
                "cell": [i],
                "bytes": 100,
                "archived_fitness": float(2.0 - 0.1 * i if walk else 1.0 - 0.01 * i),
                "displacement_m": 1.0,
                "survived": walk,
                "posture": "walk" if walk else "crawl",
                "latent": (centre + rng.normal(0, 1.0 if walk else 5.0, size=4)).tolist(),
            }
        )
    return out


def test_grid_selection_puts_top_first_then_survivors_then_sweep():
    entries = _grid_entries()
    best = sorted(entries, key=lambda e: -e["archived_fitness"])
    # A budget of three clips: exactly the top three, whatever survived.
    chosen = build_viewer._select_clips(entries, 300, top=16, top_first=3)
    assert chosen == {e["row"] for e in best[:3]}
    # A budget of six: the top three, then survivors by displacement.
    chosen = build_viewer._select_clips(entries, 600, top=16, top_first=3)
    survivors = sorted((e for e in entries if e["survived"]), key=lambda e: -e["displacement_m"])
    extra = [e for e in survivors if e not in best[:3]][:3]
    assert chosen == {e["row"] for e in best[:3] + extra}
    # Everything fits: every row is chosen, in any order.
    assert build_viewer._select_clips(entries, 1e9, 16, 3) == {e["row"] for e in entries}


def test_grid_selection_with_top_first_zero_is_the_old_survivors_first_rule():
    entries = _grid_entries()
    survivors = sorted((e for e in entries if e["survived"]), key=lambda e: -e["displacement_m"])
    chosen = build_viewer._select_clips(entries, 200, top=16, top_first=0)
    assert chosen == {e["row"] for e in survivors[:2]}


def test_cvt_selection_covers_both_postures_before_sweeping():
    entries = _cvt_entries()
    chosen = build_viewer._select_clips_cvt(entries, 400, top_per_posture=2)
    picked = [e for e in entries if e["row"] in chosen]
    assert sum(e["posture"] == "walk" for e in picked) == 2
    assert sum(e["posture"] == "crawl" for e in picked) == 2
    walk_best = sorted((e for e in entries if e["posture"] == "walk"), key=lambda e: -e["archived_fitness"])[:2]
    assert {e["row"] for e in walk_best} <= chosen


def test_cvt_sweep_spreads_along_the_crawl_continuum():
    """Farthest-point in the 4-d latent: 20 clips should not all sit at the crawl peak."""
    entries = _cvt_entries(n_walk=5, n_crawl=200, seed=1)
    chosen = build_viewer._select_clips_cvt(entries, 2000, top_per_posture=2)
    crawl = [e for e in entries if e["row"] in chosen and e["posture"] == "crawl"]
    z = np.array([e["latent"] for e in crawl])
    all_crawl = np.array([e["latent"] for e in entries if e["posture"] == "crawl"])
    # The chosen crawlers span most of the continuum's extent on every axis.
    assert np.all(np.ptp(z, axis=0) > 0.6 * np.ptp(all_crawl, axis=0))


def test_projection_is_pca_with_variance_that_sums_to_one():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(1024, 4)) * np.array([5.0, 3.0, 1.0, 0.2])
    mean, comps, explained = build_viewer._project(c)
    assert comps.shape == (2, 4)
    assert explained[0] > explained[1] > 0
    assert sum(explained) == pytest.approx(1.0)
    assert explained[0] + explained[1] > 0.9
    plane = (c - mean) @ comps.T
    assert plane.shape == (1024, 2)
    assert np.allclose(plane.mean(axis=0), 0, atol=1e-9)


def test_place_in_cells_matches_rows_through_all_verified(tmp_path):
    rng = np.random.default_rng(0)
    n, keep = 8, np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    solution = rng.normal(size=(n, 5)).astype(np.float32)
    index = np.arange(10, 10 + n)
    measures = rng.normal(size=(n, 4)).astype(np.float32)
    np.savez(
        tmp_path / "final.npz",
        solution=solution, index=index, measures=measures,
        centroids=rng.normal(size=(32, 4)).astype(np.float32),
        meta_json=np.array(json.dumps({"insertion_replicas": 8})),
    )
    data = {
        "solution": solution[keep],
        "all_verified": keep,
        "mode": np.array([1, 0, 0, 1, 0, 0]),
        "v4_label": np.array([3, 2, 2, 3, 2, 1]),  # walk, crawl, crawl, walk, crawl, hop
    }
    cvt = render_gaits._place_in_cells(data, tmp_path / "final.npz")
    assert cvt["index"].tolist() == index[keep].tolist()
    assert np.array_equal(cvt["latent"], measures[keep])
    assert cvt["postures"] == {"0": "crawl", "1": "walk"}
    assert cvt["posture"] == ["walk", "crawl", "crawl", "walk", "crawl", "crawl"]

    data["solution"] = data["solution"] + 1
    with pytest.raises(SystemExit):
        render_gaits._place_in_cells(data, tmp_path / "final.npz")
    with pytest.raises(SystemExit):
        render_gaits._place_in_cells(data, None)


def test_upright_mask_reads_the_free_joint_pose():
    import mujoco

    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.02'/></body></worldbody></mujoco>"
    )
    fit = render_gaits.FitnessCfg()
    tilt = np.radians(70.0)
    qpos = np.array(
        [
            [0, 0, 0.12, 1, 0, 0, 0],  # standing
            [0, 0, 0.05, 1, 0, 0, 0],  # collapsed
            [0, 0, 0.12, np.cos(tilt / 2), np.sin(tilt / 2), 0, 0],  # toppled past 60 deg
            [0, 0, 0.12, np.cos(0.2), np.sin(0.2), 0, 0],  # leaning 23 deg
        ]
    )
    assert render_gaits._upright_mask(m, qpos, fit).tolist() == [True, False, False, True]


def test_load_tolerates_a_verified_file_without_meta(tmp_path):
    np.savez(tmp_path / "v.npz", solution=np.zeros((2, 3)), objective=np.zeros(2))
    data = render_gaits._load(tmp_path / "v.npz")
    assert data["meta"] == {} and data["solution"].shape == (2, 3)
