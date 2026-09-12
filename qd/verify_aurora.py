"""Independent verification in the FROZEN evaluation space — the v5 numbers.

One tool, applied identically to every archive being compared (v5's final
archive, v4's walk + crawl archives as one set, anything else with a
``solution`` key):

* every elite is re-rolled in **8 fresh world-permuted replicas**; it is
  *verified* if P2'' holds in **>= 5 of 8** — the insertion gate's own k,
  declared before any result existed and not moved;
* fitness is the **median** displacement over those replicas; archive
  optimism = filed objective - verified median;
* descriptors are the frozen evaluation encoder's latents of the median
  feature vector; **coverage** is counted on the fixed centroid set;
* **modes** are DBSCAN clusters of verified elites at ``eps = 3 x noise``
  with >= 5 members (:func:`qd.behaviour_space.cluster_modes`), reported with
  inter-mode distance and per-axis occupancy, so a one-genome sweep along one
  axis is visible as such;
* robustness is reported **per mode and in aggregate**, with the full k=1..8
  strictness sweep;
* v4's classifier labels are attached to each verified elite **for
  reporting** (purity tables); nothing here routes on them.

    uv run python -m qd.verify_aurora --space logs/qd/v5/space_eval/space_ae.npz \\
        --centroids logs/qd/v5/space_eval/centroids_ae.npz \\
        --archives v5 logs/qd/aurora_v5/final.npz --out logs/qd/v5/verify_v5
    uv run python -m qd.verify_aurora ... --archives v4 qd-run-archives/j007/modes_v4/final/archive_walk.npz \\
        v4b qd-run-archives/j007/modes_v4/final/archive_crawl.npz --out logs/qd/v5/verify_v4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.behaviour import BehaviourCfg, ViabilityV5Cfg
from qd.behaviour_space import BehaviourSpace, cluster_modes, coverage, purity
from qd.collect_behaviour import Args as CollectArgs
from qd.collect_behaviour import rollout_sets
from qd.common import FitnessCfg, load_archive, write_json
from qd.modes import MODES, ViabilityCfg


@dataclass
class Args:
    archives: dict[str, Path] = field(default_factory=dict)
    """``name path`` pairs; all are verified as ONE set (so v4 walk + crawl
    become one archive scored on one centroid set)."""
    space: Path = Path("logs/qd/v5/space_eval/space_ae.npz")
    centroids: Path = Path("logs/qd/v5/space_eval/centroids_ae.npz")
    out: Path = Path("logs/qd/v5/verify")
    replicas: int = 8
    viable_min: int = 5
    max_envs: int = 1024
    viability: ViabilityV5Cfg = field(default_factory=ViabilityV5Cfg)
    behaviour: BehaviourCfg = field(default_factory=BehaviourCfg)
    fitness: FitnessCfg = field(default_factory=lambda: FitnessCfg(latch_fall=False))
    seed: int = 0
    device: str = "cuda:0"


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    space = BehaviourSpace.load(args.space)
    with np.load(args.centroids) as f:
        centroids, bounds, noise = f["centroids"], f["bounds"], float(f["noise"])

    genomes, objective, source = [], [], []
    for name, path in args.archives.items():
        a = load_archive(path)
        genomes.append(np.asarray(a["solution"], dtype=np.float32))
        objective.append(np.asarray(a["objective"], dtype=np.float64))
        source += [name] * len(a["solution"])
    genomes = np.concatenate(genomes)
    objective = np.concatenate(objective)
    source = np.array(source)
    print(f"verifying {len(genomes)} elites from {dict(zip(*np.unique(source, return_counts=True)))}", flush=True)

    cargs = CollectArgs(
        replicas=args.replicas, max_envs=args.max_envs, viability=args.viability,
        v4_viability=ViabilityCfg(), behaviour=args.behaviour, fitness=args.fitness,
        seed=args.seed, device=args.device,
    )
    data, names = rollout_sets(genomes, args.replicas, cargs)
    assert tuple(names) == tuple(space.names), "feature layout differs from the space's"

    count = data["viable"].sum(axis=0)
    verified = count >= args.viable_min
    median_dx = np.median(data["displacement"], axis=0)
    feats = np.median(data["features"], axis=0)
    z = space.encode(feats)
    modal_label = np.array(
        [np.bincount(c, minlength=len(MODES)).argmax() for c in data["v4_label"].T]
    )

    n_cov, _cov_idx = coverage(z[verified], centroids)
    fi = {n: i for i, n in enumerate(space.names)}
    dq_cols = [i for n, i in fi.items() if n.startswith("dq")]
    extra = {
        "z_mean": feats[verified, fi["z_mean"]],
        "dq_rms_mean": feats[verified][:, dq_cols].mean(1),
        "f_air": feats[verified, fi["f_air"]],
        "n_contacts_mean": feats[verified, fi["n_contacts_mean"]],
        "displacement": median_dx[verified],
    }
    report = cluster_modes(z[verified], noise, bounds, extra=extra)

    # per-mode robustness and composition
    v_idx = np.flatnonzero(verified)
    for m in report.modes:
        members = v_idx[report.labels == m["mode"]]
        vc = count[members]
        m["mean_viable_replicas"] = float(vc.mean() / args.replicas)
        m["strictness_sweep"] = {k: int(np.sum(vc >= k)) for k in range(1, args.replicas + 1)}
        m["v4_label_composition"] = {
            MODES[int(k)]: int(n) for k, n in zip(*np.unique(modal_label[members], return_counts=True))
        }
        m["source_composition"] = {
            str(k): int(n) for k, n in zip(*np.unique(source[members], return_counts=True))
        }
        m["best_median_displacement"] = float(median_dx[members].max())
        m["coverage_cells"] = int(coverage(z[members], centroids)[0])
        m["optimism_m"] = float(np.mean(objective[members] - median_dx[members]))

    summary = {
        "archives": {k: str(v) for k, v in args.archives.items()},
        "space": str(args.space),
        "centroids": str(args.centroids),
        "num_centroids": len(centroids),
        "noise": noise,
        "eps": report.eps,
        "replicas": args.replicas,
        "viable_min": args.viable_min,
        "gate": {"d_min": args.viability.d_min, "impact_cap": args.viability.impact_cap,
                 "delta_max": args.viability.delta_max},
        "filed": len(genomes),
        "verified": int(verified.sum()),
        "robustness_aggregate": float(verified.mean()),
        "mean_viable_replicas": float(count.mean() / args.replicas),
        "strictness_sweep": {k: int(np.sum(count >= k)) for k in range(1, args.replicas + 1)},
        "per_source": {
            str(s): {
                "filed": int(np.sum(source == s)),
                "verified": int(np.sum(verified & (source == s))),
                "robustness": float(np.mean(verified[source == s])),
            }
            for s in np.unique(source)
        },
        "coverage_verified": n_cov,
        "coverage_fraction": n_cov / len(centroids),
        "coverage_all_filed": int(coverage(z, centroids)[0]),
        "best_median_displacement": float(median_dx[verified].max()) if verified.any() else float("nan"),
        "archive_optimism_m": float(np.mean(objective[verified] - median_dx[verified])) if verified.any() else float("nan"),
        "clause_rates": {c: float(data[c].mean()) for c in ("finite", "progress", "stationary", "impact")},
        "n_modes": report.n_modes,
        "n_fragments": report.n_fragments,
        "n_noise": report.n_noise,
        "min_inter_mode_distance_over_eps": report.min_inter_mode_distance_over_eps,
        "modes": report.modes,
        "purity_vs_v4_label": {
            c: {MODES[int(k)]: n for k, n in d.items()}
            for c, d in purity(report.labels, modal_label[verified]).items()
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    np.savez_compressed(
        args.out / "verified.npz",
        solution=genomes[verified], objective=objective[verified],
        median_displacement=median_dx[verified], latent=z[verified],
        features=feats[verified], mode=report.labels, source=source[verified].astype("U32"),
        v4_label=modal_label[verified], viable_count=count[verified],
        all_latent=z, all_verified=verified, all_median_displacement=median_dx,
        all_source=source.astype("U32"), feature_names=np.array(names, dtype="U64"),
    )
    _print(summary)
    print(f"wrote {args.out}/summary.json")


def _print(s: dict) -> None:
    print(f"\nfiled {s['filed']} | verified {s['verified']} ({s['robustness_aggregate'] * 100:.1f}%) at {s['viable_min']}-of-{s['replicas']}")
    print(f"strictness sweep: {s['strictness_sweep']}")
    print(f"per source: {s['per_source']}")
    print(f"coverage (verified) {s['coverage_verified']}/{s['num_centroids']} = {s['coverage_fraction'] * 100:.1f}%")
    print(f"best median {s['best_median_displacement']:+.3f} m | optimism {s['archive_optimism_m']:+.3f} m")
    print(f"modes {s['n_modes']} (fragments {s['n_fragments']}, noise {s['n_noise']}) | eps {s['eps']:.3f} | min inter-mode dist/eps {s['min_inter_mode_distance_over_eps']}")
    for m in s["modes"]:
        print(
            f"  mode {m['mode']}: {m['elites']} elites, cells {m['coverage_cells']}, robust {m['mean_viable_replicas']:.3f}, "
            f"axes>=2bins {m['axes_spanned_ge2_bins']}/{len(m['axis_bins_occupied'])} {m['axis_bins_occupied']}, "
            f"best {m['best_median_displacement']:+.3f} m, z {m['z_mean_range']}, v4 {m['v4_label_composition']}"
        )


if __name__ == "__main__":
    main()
