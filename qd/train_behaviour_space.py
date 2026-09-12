"""Fit the frozen EVALUATION behaviour space and write the kill-gate report.

Reads the dataset :mod:`qd.collect_behaviour` produced, fits the encoder
(AE, and PCA as the linear control), measures the replica noise, fixes the
centroid set, and then asks the question the job is allowed to stop on:

    *does the learned space, told nothing, recover the two modes v4 already
    knows exist?*

Everything written to ``--out`` is frozen from this point: ``space_ae.npz``
(or ``space_pca.npz``) is the encoder every archive in this job is scored
with, ``centroids.npz`` the fixed centroid set, ``noise`` the unit of
distance. The search may retrain its *own* encoder AURORA-style; it never
touches these.

    uv run python -m qd.train_behaviour_space --data logs/qd/v5/behaviour_data.npz \\
        --out logs/qd/v5/space_eval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.behaviour_space import (
    DEFAULT_CENTROIDS,
    BehaviourSpace,
    TrainCfg,
    cluster_modes,
    coverage,
    fit_space,
    latent_bounds,
    make_centroids,
    purity,
)
from qd.common import write_json
from qd.modes import MODES


@dataclass
class Args:
    data: Path = Path("logs/qd/v5/behaviour_data.npz")
    out: Path = Path("logs/qd/v5/space_eval")
    train: TrainCfg = field(default_factory=TrainCfg)
    centroids: int = DEFAULT_CENTROIDS
    viable_min: int = 5
    v4_sets: tuple[str, ...] = ("v4_walk", "v4_crawl")
    train_sets: tuple[str, ...] = ()
    """Sets to train on; empty = every set in the file."""
    plot: bool = True


def load_data(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def embed_dataset(space: BehaviourSpace, feats: np.ndarray) -> np.ndarray:
    """``(R, M, D)`` -> ``(R, M, L)``."""
    r, m, d = feats.shape
    return space.encode(feats.reshape(r * m, d)).reshape(r, m, -1)


def viable_noise(space: BehaviourSpace, data: dict, mask: np.ndarray, viable_min: int) -> tuple[float, int]:
    """Replica noise over genomes viable in >= k replicas, using their viable replicas."""
    z = embed_dataset(space, data["features"][:, mask])
    v = data["viable"][:, mask]
    per = []
    for j in range(z.shape[1]):
        ok = v[:, j]
        if ok.sum() >= viable_min:
            zz = z[ok, j]
            med = np.median(zz, axis=0, keepdims=True)
            per.append(float(np.sqrt(((zz - med) ** 2).sum(-1).mean())))
    if not per:
        return float("nan"), 0
    return float(np.median(per)), len(per)


def score_sets(
    space: BehaviourSpace, data: dict, sets: tuple[str, ...], centroids: np.ndarray,
    bounds: np.ndarray, noise: float, viable_min: int,
) -> dict:
    """Verified-elite clustering + coverage for a group of sets, on the fixed centroids."""
    mask = np.isin(data["set"], sets)
    v = data["viable"][:, mask]
    count = v.sum(axis=0)
    verified = count >= viable_min
    feats = np.median(data["features"][:, mask], axis=0)
    z = space.encode(feats)
    zv = z[verified]
    modal = np.array(
        [np.bincount(col, minlength=len(MODES)).argmax() for col in data["v4_label"][:, mask].T]
    )
    disp = np.median(data["displacement"][:, mask], axis=0)
    extra = {
        "z_mean": feats[:, list(space.names).index("z_mean")][verified],
        "dq_rms_mean": feats[:, [i for i, n in enumerate(space.names) if n.startswith("dq")]].mean(1)[verified],
        "displacement": disp[verified],
    }
    report = cluster_modes(zv, noise, bounds, extra=extra)
    n_cov, _ = coverage(zv, centroids)
    n_cov_all, _ = coverage(z, centroids)
    return {
        "sets": list(sets),
        "genomes": int(mask.sum()),
        "verified_5of8": int(verified.sum()),
        "robustness": float(verified.mean()) if mask.any() else float("nan"),
        "per_set_verified": {
            s: int(np.sum(verified & (data["set"][mask] == s))) for s in sets
        },
        "per_set_robustness": {
            s: float(np.mean(verified[data["set"][mask] == s])) for s in sets
            if np.any(data["set"][mask] == s)
        },
        "coverage_verified": n_cov,
        "coverage_verified_fraction": n_cov / len(centroids),
        "coverage_all_filed": n_cov_all,
        "n_modes": report.n_modes,
        "n_fragments": report.n_fragments,
        "n_noise": report.n_noise,
        "eps": report.eps,
        "min_inter_mode_distance_over_eps": report.min_inter_mode_distance_over_eps,
        "modes": report.modes,
        "purity_vs_v4_label": {
            c: {MODES[int(k)]: n for k, n in d.items()}
            for c, d in purity(report.labels, modal[verified]).items()
        },
        "best_median_displacement": float(disp[verified].max()) if verified.any() else float("nan"),
    }


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_data(args.data)
    names = tuple(str(n) for n in data["feature_names"])
    feats = data["features"]  # (R, M, D)
    r, m, d = feats.shape
    train_mask = (
        np.isin(data["set"], args.train_sets) if args.train_sets else np.ones(m, bool)
    )
    x_train = feats[:, train_mask].reshape(-1, d)
    print(f"training rows {len(x_train)} from {int(train_mask.sum())} genomes x {r} replicas", flush=True)

    # genomes viable in >= k replicas, each contributing its viable replicas:
    # the noise unit is measured on the population the archive will hold
    viable = data["viable"]
    groups = [
        feats[viable[:, j], j]
        for j in range(m)
        if viable[:, j].sum() >= args.viable_min
    ]
    print(f"noise calibration on {len(groups)} viable genomes", flush=True)

    spaces = {}
    for kind in ("ae", "pca"):
        cfg = TrainCfg(**{**args.train.__dict__, "kind": kind})
        spaces[kind] = fit_space(x_train, names, cfg)
        cal = spaces[kind].calibrate_to_noise(groups)
        spaces[kind].save(args.out / f"space_{kind}.npz")
        print(f"{kind}: {spaces[kind].train_info}", flush=True)
        print(f"{kind}: spread/noise per axis {np.round(cal['spread_to_noise_per_axis'], 2)}", flush=True)

    report: dict = {"data": str(args.data), "latent_dim": args.train.latent_dim, "kinds": {}}
    for kind, space in spaces.items():
        z_all = embed_dataset(space, feats)
        bounds = latent_bounds(z_all.reshape(-1, space.latent_dim))
        centroids = make_centroids(args.centroids, bounds, seed=args.train.seed)
        noise, n_noise_genomes = viable_noise(space, data, np.ones(m, bool), args.viable_min)
        np.savez_compressed(
            args.out / f"centroids_{kind}.npz", centroids=centroids, bounds=bounds,
            noise=np.float32(noise),
        )
        v4 = score_sets(space, data, args.v4_sets, centroids, bounds, noise, args.viable_min)
        per_set = {}
        for s in np.unique(data["set"]):
            per_set[str(s)] = score_sets(space, data, (str(s),), centroids, bounds, noise, args.viable_min)
        # separation of the two reference labels among verified v4 elites, told nothing
        report["kinds"][kind] = {
            "train_info": space.train_info,
            "bounds": bounds.tolist(),
            "noise": noise,
            "noise_genomes": n_noise_genomes,
            "eps": noise * 3.0,
            "v4": v4,
            "per_set": per_set,
        }
        print(
            f"[{kind}] noise {noise:.3f} (n={n_noise_genomes}) | v4 verified {v4['verified_5of8']}/{v4['genomes']} "
            f"| modes {v4['n_modes']} (frag {v4['n_fragments']}, noise {v4['n_noise']}) "
            f"| coverage {v4['coverage_verified']}/{args.centroids} | purity {v4['purity_vs_v4_label']}",
            flush=True,
        )
        if args.plot:
            _plot(space, data, args.out / f"latent_{kind}.png", args.viable_min)

    # delta_max calibration: consecutive-window support change on v4 elites'
    # replicas that clear the other clauses
    v4mask = np.isin(data["set"], args.v4_sets)
    ok = data["progress"][:, v4mask] & data["impact"][:, v4mask] & data["finite"][:, v4mask]
    md = data["max_delta"][:, v4mask][ok]
    report["delta_max_calibration"] = {
        "replicas": int(ok.sum()),
        "percentiles_50_90_95_99_100": np.percentile(md, [50, 90, 95, 99, 100]).tolist(),
        "share_below_0.10": float(np.mean(md <= 0.10)),
        "share_below_0.15": float(np.mean(md <= 0.15)),
        "share_below_0.20": float(np.mean(md <= 0.20)),
    }
    write_json(args.out / "checkpoint.json", report)
    print(f"wrote {args.out}/checkpoint.json")


def _plot(space: BehaviourSpace, data: dict, path: Path, viable_min: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = np.median(data["features"], axis=0)
    z = space.encode(feats)
    verified = data["viable"].sum(axis=0) >= viable_min
    sets = data["set"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sel, title in ((axes[0], np.ones(len(z), bool), "all genomes"), (axes[1], verified, "verified 5-of-8")):
        for s in np.unique(sets):
            m = sel & (sets == s)
            if m.any():
                ax.scatter(z[m, 0], z[m, 1], s=6, alpha=0.6, label=f"{s} ({m.sum()})")
        ax.set_title(f"{space.kind} latent 0/1 — {title}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
