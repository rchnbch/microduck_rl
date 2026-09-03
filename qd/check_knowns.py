"""Checkpoint 2 — does the v4 gate do the right thing to things we already know?

Before a single new genome is trained, the gate is pointed at populations whose
verdict is known in advance:

* **must fail** — v1's 292 PGA-ME divers, v1's 340 CPG elites, 1024 random
  MLPs, and the scripted degenerates (dive, twitcher, HOME hold, prone-still,
  thrash);
* **must pass, and classify as walk** — the six distilled PPO seeds and the
  five best v3 elites.

One exception is already known and is *not* a failure of the gate: five of
v1's CPG elites turned out to be **real crawls** (checkpoint 1, §5), verified
over 128 world-permuted replicas. They are named here rather than quietly
subtracted, and the "must fail" count is reported both with and without them.

Everything is computed from the Stage A' feature caches, so this costs no GPU
and can be re-run whenever a threshold moves::

    uv run python -m qd.check_knowns --features logs/qd/stage_a_prime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np
import tyro

from qd.common import write_json
from qd.modes import (
    MODES,
    ClassifierCfg,
    ViabilityCfg,
    evaluate_viability,
)
from qd.stage_a_prime import load_results

RESCUED_CRAWLS: tuple[str, ...] = (
    "v1_cpg_175",
    "v1_cpg_277",
    "v1_cpg_169",
    "v1_cpg_4",
    "v1_cpg_6",
)
"""v1 CPG elites that pass P2' because they are crawls, not because it leaks.

Verified at 128 world-permuted replicas: 175 and 277 are unanimously viable,
unanimously labelled crawl, per-window constancy 1.000, travelling +0.543 m and
+0.359 m from a standing spawn. Named here so the "every v1 elite must fail"
check reports an exception rather than hiding one.
"""


@dataclass
class Args:
    features: Path = Path("logs/qd/stage_a_prime")
    out: Path | None = None
    viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    viable_min: int = 5
    label_agreement_min: int = 7
    replicas: int = 8
    """Replica count the gate is scored at."""

    label_over_viable_only: bool = True
    """Whether label agreement is asked of the viable replicas only.

    See :func:`qd.pga.run_modes.fold_replicas`. Both readings are reported."""


def at_least(p: np.ndarray, k: int, n: int) -> np.ndarray:
    """P(at least k of n) for a measured per-replica rate."""
    p = np.asarray(p, dtype=float)
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def admission(
    viable: np.ndarray,
    label_ok: np.ndarray,
    viable_min: int,
    label_min: int,
    replicas: int,
    draws: int = 4000,
    seed: int = 0,
    label_over_viable_only: bool = True,
) -> float:
    """Empirical P(the gate admits this genome), resampling groups of replicas.

    Not the product of two binomials. Viability and label agreement are *not*
    independent here — a replica in which the walker falls late is both
    non-viable and label-flipped, because the fall is what flips the label — so
    multiplying the two marginals overstates admission. Resampling the measured
    per-replica outcomes jointly gets the correlation for free.

    With fewer replicas than the gate draws (the bulk negatives have exactly 8)
    the sample is drawn with replacement, which is the honest fallback: it says
    what a gate applied to a genome like this one would do, not what it did on
    one particular batch.
    """
    rng = np.random.default_rng(seed)
    n = len(viable)
    if n == 0:
        return float("nan")
    idx = rng.integers(0, n, size=(draws, replicas))
    vi = np.asarray(viable, dtype=bool)[idx]
    li = np.asarray(label_ok, dtype=bool)[idx]
    v = vi.sum(axis=1)
    if label_over_viable_only:
        agree = (li & vi).sum(axis=1) / np.maximum(v, 1) * replicas
    else:
        agree = li.sum(axis=1)
    return float(np.mean((v >= viable_min) & (agree >= label_min)))


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    results = []
    for name in ("scripted", "genomes", "cpg"):
        path = args.features / f"features_{name}.npz"
        if path.exists():
            results += load_results(path)
    if not results:
        raise SystemExit(f"no feature caches under {args.features}")

    cfg = ViabilityCfg(
        windows=args.viability.windows,
        classifier=args.classifier,
        d_min=args.viability.d_min,
        exempt_seconds=args.viability.exempt_seconds,
        impact_cap=args.viability.impact_cap,
    )
    w = cfg.windows.window_seconds

    rows = []
    for r in results:
        feats = r.features.get(w, r.features[2.0])
        v = evaluate_viability(feats, cfg)
        labels = feats.episode_labels(cfg.classifier)
        modal = int(np.bincount(labels, minlength=len(MODES)).argmax())
        label_ok = labels == modal
        rows.append(
            {
                "name": r.name,
                "source": r.source,
                "positive": r.positive,
                "intended_mode": r.intended_mode,
                "per_replica": float(v.viable.mean()),
                "label": MODES[modal],
                "label_agreement": float(np.mean(label_ok)),
                "median_displacement_m": float(np.median(feats.displacement)),
                "admitted": admission(
                    v.viable, label_ok, args.viable_min,
                    args.label_agreement_min, args.replicas,
                    label_over_viable_only=args.label_over_viable_only,
                ),
                "admitted_label_over_all": admission(
                    v.viable, label_ok, args.viable_min,
                    args.label_agreement_min, args.replicas,
                    label_over_viable_only=False,
                ),
                "_viable": v.viable,
                "_label_ok": label_ok,
            }
        )

    positives = [r for r in rows if r["positive"] and r["source"] != "scripted"]
    walk_positives = [r for r in positives if r["intended_mode"] == "walk"]
    negatives = [r for r in rows if not r["positive"]]
    rescued = [r for r in negatives if r["name"] in RESCUED_CRAWLS]
    other_negs = [r for r in negatives if r["name"] not in RESCUED_CRAWLS]

    admitted = [r for r in other_negs if r["admitted"] > 0.5]
    expected = float(sum(r["admitted"] for r in other_negs))

    report = {
        "gate": {
            "window_seconds": w,
            "stride_seconds": cfg.windows.stride_seconds,
            "d_min": cfg.d_min,
            "impact_cap": cfg.impact_cap,
            "viable_min": args.viable_min,
            "replicas": args.replicas,
            "label_agreement_min": args.label_agreement_min,
            "hop_air_min": cfg.classifier.hop_air_min,
        },
        "negatives": {
            "total": len(other_negs),
            "admitted_more_likely_than_not": [r["name"] for r in admitted],
            "expected_admitted": expected,
            "expected_admitted_fraction": expected / max(len(other_negs), 1),
            "worst_per_replica": max((r["per_replica"] for r in other_negs), default=0.0),
            "by_source": {
                src: sum(1 for r in other_negs if r["source"] == src)
                for src in sorted({r["source"] for r in other_negs})
            },
        },
        "rescued_crawls": [
            {k: r[k] for k in ("name", "per_replica", "label", "median_displacement_m", "admitted")}
            for r in rescued
        ],
        "walk_positives": [
            {
                k: r[k]
                for k in ("name", "per_replica", "label", "label_agreement",
                          "median_displacement_m", "admitted",
                          "admitted_label_over_all")
            }
            for r in walk_positives
        ],
        "walk_summary": {
            "count": len(walk_positives),
            "all_label_walk": all(r["label"] == "walk" for r in walk_positives),
            "min_label_agreement": min(
                (r["label_agreement"] for r in walk_positives), default=float("nan")
            ),
            "label_agreement_bar": args.label_agreement_min / args.replicas,
            "mean_per_replica": float(
                np.mean([r["per_replica"] for r in walk_positives])
            ),
            "mean_admitted": float(np.mean([r["admitted"] for r in walk_positives])),
            "mean_admitted_label_over_all": float(
                np.mean([r["admitted_label_over_all"] for r in walk_positives])
            ),
        },
        "strictness_sweep": [
            {
                "viable_min": k,
                "walk_admitted": float(
                    np.mean([
                        admission(r["_viable"], r["_label_ok"], k,
                                  args.label_agreement_min, args.replicas,
                                  label_over_viable_only=args.label_over_viable_only)
                        for r in walk_positives
                    ])
                ),
                "negatives_expected": float(
                    sum(
                        admission(r["_viable"], r["_label_ok"], k,
                                  args.label_agreement_min, args.replicas)
                        for r in other_negs
                    )
                ),
                "rescued_crawls_admitted": float(
                    sum(
                        admission(r["_viable"], r["_label_ok"], k,
                                  args.label_agreement_min, args.replicas)
                        for r in rescued
                    )
                ),
            }
            for k in range(4, args.replicas + 1)
        ],
    }

    for r in rows:
        r.pop("_viable", None)
        r.pop("_label_ok", None)
    _print(report)
    if args.out:
        write_json(Path(args.out) / "check_knowns.json", report)
        print(f"\nwrote {args.out}/check_knowns.json")


def _print(report: dict) -> None:
    g = report["gate"]
    print(
        f"\ngate: W={g['window_seconds']:g}s stride={g['stride_seconds']:g}s "
        f"d_min={g['d_min']:g}m cap={g['impact_cap']} "
        f"viable {g['viable_min']}-of-{g['replicas']} "
        f"label {g['label_agreement_min']}-of-{g['replicas']}"
    )

    n = report["negatives"]
    print(f"\n=== must fail: {n['total']} known negatives ===")
    print(f"   by source: {n['by_source']}")
    print(f"   worst per-replica pass rate: {n['worst_per_replica']:.3f}")
    print(
        f"   expected admitted by the gate: {n['expected_admitted']:.2f} "
        f"({n['expected_admitted_fraction'] * 100:.3f}%)"
    )
    print(f"   admitted more likely than not: {n['admitted_more_likely_than_not'] or 'none'}")

    print(f"\n=== the named exception: {len(report['rescued_crawls'])} v1 CPG crawls ===")
    for r in report["rescued_crawls"]:
        print(
            f"   {r['name']:14s} per-replica {r['per_replica']:.3f}  "
            f"label {r['label']:6s}  dx {r['median_displacement_m']:+.3f} m  "
            f"admitted {r['admitted']:.3f}"
        )

    w = report["walk_summary"]
    print(f"\n=== must pass and classify as walk: {w['count']} known walkers ===")
    print(f"   {'probe':16s} {'per-replica':>12s} {'label':>7s} {'agreement':>10s} "
          f"{'dx':>8s} {'admitted':>9s} {'(label/all)':>12s}")
    for r in report["walk_positives"]:
        print(
            f"   {r['name']:16s} {r['per_replica']:12.3f} {r['label']:>7s} "
            f"{r['label_agreement']:10.3f} {r['median_displacement_m']:8.3f} "
            f"{r['admitted']:9.3f} {r['admitted_label_over_all']:12.3f}"
        )
    print(
        f"   all labelled walk: {w['all_label_walk']}   "
        f"min label agreement {w['min_label_agreement']:.3f} "
        f"(bar {w['label_agreement_bar']:.3f})"
    )
    print(
        f"   mean per-replica {w['mean_per_replica']:.3f} -> "
        f"mean admitted {w['mean_admitted']:.3f} "
        f"(label asked of ALL replicas: {w['mean_admitted_label_over_all']:.3f})"
    )

    print("\n=== strictness sweep (the reading is revisable) ===")
    print(f"   {'k of 8':8s} {'walk admitted':>14s} {'negatives expected':>19s} "
          f"{'rescued crawls':>15s}")
    for row in report["strictness_sweep"]:
        print(
            f"   {row['viable_min']:<8d} {row['walk_admitted']:14.3f} "
            f"{row['negatives_expected']:19.2f} {row['rescued_crawls_admitted']:15.2f}"
        )


if __name__ == "__main__":
    main()
