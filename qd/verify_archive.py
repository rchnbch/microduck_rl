"""Re-evaluate every elite N times and keep the ones that are actually real.

A survival-gated archive is gated on **one** rollout per candidate. On this
simulator that is not enough: `qd.check_repeatability` measures a walking
policy's displacement at sd 0.605 m across byte-identical worlds, so the gate
admits any marginal policy that happened to stay up once, and MAP-Elites then
keeps whichever of them got the luckiest distance. The archive that comes out
is therefore optimistic in *both* of its columns — the fitness and the survival.

This measures both and writes the honest archive:

* every elite is rolled out ``replicas`` times;
* ``survival_rate`` is the fraction of those rollouts it stayed upright for;
* fitness becomes the **median** displacement, not the archived best sample;
* elites below ``--min-survival`` are dropped.

What comes out is what re-evaluation on insertion would have produced, applied
after the fact. It cannot recover the elites a properly re-evaluating run would
have found instead — those were never generated — so treat the verified archive
as a lower bound on what the same budget could do, and the gap between the raw
and verified numbers as the cost of inserting on a single sample.

    uv run python -m qd.verify_archive \\
        --archive logs/qd/pga_me_v2/archive_final.npz --replicas 8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.replay import infer_kind, reevaluate


@dataclass
class Args:
    archive: Path
    replicas: int = 8
    """Rollouts per elite. 8 puts the standard error of a survival rate at
    ~0.18 for a coin-flip elite and ~0 for a robust one, which is enough to
    separate the two populations."""

    min_survival: float = 0.875
    """Keep elites that survive at least this fraction of their replicas.

    0.875 is 7 of 8: one bad draw forgiven, two not. Chosen because the
    distilled seeds sit at 0.99-1.00 and a lucky marginal policy sits far
    below, so the threshold lands in the empty middle rather than cutting
    through a population."""

    out: Path | None = None
    """Where to write the verified archive; defaults to
    ``<archive stem>_verified.npz`` beside the input."""

    device: str = "cuda:0"
    max_envs: int = 512
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_archive(args.archive)
    solutions, archived = data["solution"], data["objective"]
    dims = tuple(int(x) for x in data["grid_dims"])
    total_cells = dims[0] * dims[1]
    n, reps = len(archived), max(1, args.replicas)
    kind = infer_kind(solutions)

    print(
        f"re-evaluating all {n} elites of {args.archive} ({kind}) x {reps} "
        f"replicas = {n * reps} rollouts",
        flush=True,
    )
    # Elite-major, so a reshape recovers the per-elite axis after chunking.
    _, measures, info, control_dt = reevaluate(
        np.repeat(solutions, reps, axis=0),
        kind,
        args.fitness,
        args.device,
        args.max_envs,
    )
    displacement = info["displacement"].reshape(n, reps)
    upright = ~info["fell"].reshape(n, reps)
    measures = measures.reshape(n, reps, -1)

    survival_rate = upright.mean(axis=1)
    median_displacement = np.median(displacement, axis=1)
    keep = survival_rate >= args.min_survival

    def cells(mask):
        return int(len(np.unique(data["index"][mask]))) if mask.any() else 0

    summary = {
        "archive": str(args.archive),
        "replicas": reps,
        "min_survival": args.min_survival,
        "elites_raw": n,
        "coverage_raw": n / total_cells,
        "archived_best_m": float(archived.max()),
        # --- what survives verification ---
        "elites_verified": int(keep.sum()),
        "coverage_verified": float(keep.sum()) / total_cells,
        "verified_fraction": float(keep.mean()),
        "mean_survival_rate": float(survival_rate.mean()),
        "elites_never_survived": int((survival_rate == 0).sum()),
        "elites_always_survived": int((survival_rate == 1).sum()),
        "best_verified_median_displacement_m": (
            float(median_displacement[keep].max()) if keep.any() else None
        ),
        "verified_cells_over_0.25m": cells(keep & (median_displacement >= 0.25)),
        "verified_cells_over_0.50m": cells(keep & (median_displacement >= 0.50)),
        "verified_cells_over_1.00m": cells(keep & (median_displacement >= 1.00)),
        # --- the cost of inserting on a single sample ---
        "archive_optimism_mean_m": float(np.mean(archived - median_displacement)),
        "archive_optimism_median_m": float(np.median(archived - median_displacement)),
        "within_elite_displacement_sd_m": float(displacement.std(axis=1).mean()),
        "within_elite_duty_sd": [
            float(measures[:, :, 0].std(axis=1).mean()),
            float(measures[:, :, 1].std(axis=1).mean()),
        ],
    }

    hist = np.histogram(survival_rate, bins=np.linspace(0, 1, reps + 2))[0]
    print(f"\nsurvival rate over {reps} replicas, per elite:")
    edges = np.linspace(0, 1, reps + 2)
    for lo, hi, count in zip(edges[:-1], edges[1:], hist):
        if count:
            print(f"  {lo:.2f}-{hi:.2f}: {'#' * min(count, 60)} {count}")
    print(
        f"\nraw archive      {n:4d} elites, {n / total_cells * 100:.1f}% coverage, "
        f"best archived {archived.max():+.3f} m\n"
        f"verified archive {int(keep.sum()):4d} elites "
        f"({keep.mean() * 100:.0f}% survive >= {args.min_survival:.0%} of replicas), "
        f"{keep.sum() / total_cells * 100:.1f}% coverage, "
        f"best median {summary['best_verified_median_displacement_m'] or float('nan'):+.3f} m\n"
        f"cells with a verified elite over 0.25 m: "
        f"{summary['verified_cells_over_0.25m']}, "
        f"over 0.50 m: {summary['verified_cells_over_0.50m']}, "
        f"over 1.00 m: {summary['verified_cells_over_1.00m']}\n"
        f"archive optimism (archived - verified median): "
        f"{summary['archive_optimism_mean_m']:+.3f} m mean, "
        f"{summary['archive_optimism_median_m']:+.3f} m median",
        flush=True,
    )

    out = args.out or args.archive.with_name(args.archive.stem + "_verified.npz")
    np.savez_compressed(
        out,
        solution=solutions[keep],
        objective=median_displacement[keep],
        measures=np.median(measures, axis=1)[keep],
        index=data["index"][keep],
        grid_dims=np.asarray(dims),
        measure_ranges=data["measure_ranges"],
        survival_rate=survival_rate[keep],
        meta_json=np.array(
            __import__("json").dumps(
                {**data["meta"], "verified": True, "replicas": reps,
                 "min_survival": args.min_survival}
            )
        ),
    )
    write_json(out.with_suffix(".json"), summary)
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
