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
from qd.descriptors import DescriptorCfg, grid_indices
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
    # Verify on the axes the archive was BUILT on. Defaulting to duty factor
    # here would silently re-bin a v3 archive and every cell number below would
    # be about a descriptor the run never used.
    descriptor = DescriptorCfg.from_meta(data.get("meta"))
    total_cells = dims[0] * dims[1]
    n, reps = len(archived), max(1, args.replicas)
    kind = infer_kind(solutions)

    print(
        f"re-evaluating all {n} elites of {args.archive} ({kind}) x {reps} "
        f"replicas = {n * reps} rollouts",
        flush=True,
    )
    # Elite-major, so a reshape recovers the per-elite axis after chunking.
    _, measures, info, _dt = reevaluate(
        np.repeat(solutions, reps, axis=0),
        kind,
        args.fitness,
        args.device,
        args.max_envs,
        descriptor=descriptor,
    )
    displacement = info["displacement"].reshape(n, reps)
    upright = ~info["fell"].reshape(n, reps)
    measures = measures.reshape(n, reps, -1)

    survival_rate = upright.mean(axis=1)
    median_displacement = np.median(displacement, axis=1)
    keep = survival_rate >= args.min_survival

    # --- is the archive's geography real? ---------------------------------- #
    # A cell means something only if an elite re-measured from fresh replicas
    # still lands in it. On v3's axes a 20-bin grid over a 5 mm span has 0.26 mm
    # bins against a 0.32 mm single-rollout sd, so this is not a formality: it
    # is the number that says whether "cell (7, 12)" names a behaviour or a
    # rounding of noise. Reported at the same median-over-replicas the archive
    # inserted on, so it measures the archive's rule rather than a stricter one.
    median_measures = np.median(measures, axis=1)
    verified_cell = grid_indices(median_measures, dims, descriptor.ranges)
    archived_cell = np.stack(np.unravel_index(data["index"].astype(int), dims), axis=-1)
    cell_offset = np.abs(verified_cell - archived_cell).max(axis=1)
    per_axis_offset = np.abs(verified_cell - archived_cell)
    stable = cell_offset == 0
    near = cell_offset <= 1

    # A grid finer than the descriptor's own reproducibility does not measure
    # more behaviour, it measures more noise. Coarsening by k and asking again
    # says what the archive's *effective* resolution is: the k at which an
    # elite reliably returns to its own cell is the resolution its geography
    # can actually support, whatever the 20x20 the run was binned at.
    # How many cells does the descriptor's own reproducibility actually
    # support? An archived measure is a median over `reps` replicas, so its
    # standard error is the within-elite sd over root-reps. Two elites one bin
    # apart differ by a quantity whose sd is root-2 times that, so bins
    # 2*root-2 sigma wide are the finest at which *adjacent* cells are
    # separated at about two sigma. That width, not the grid the run was binned
    # at, is what a coverage number should be read against — and it is capped
    # at the grid's own resolution, since the archive cannot hold more cells
    # than it has.
    sigma_median = np.array(
        [measures[:, :, k].std(axis=1).mean() / np.sqrt(reps) for k in range(2)]
    )
    axis_span = np.array([hi - lo for lo, hi in descriptor.ranges])
    resolvable_dims = tuple(
        int(np.clip(np.floor(axis_span[k] / (2 * np.sqrt(2) * sigma_median[k])), 1, dims[k]))
        for k in range(2)
    )
    resolvable_cell = grid_indices(
        median_measures, resolvable_dims, descriptor.ranges
    )

    def resolvable_cells(mask):
        if not mask.any():
            return 0
        return int(len(np.unique(resolvable_cell[mask], axis=0)))

    resolution_sweep = [
        {
            "coarsen": int(k),
            "grid": [dims[0] // k, dims[1] // k],
            "same_cell_fraction": float(
                ((verified_cell // k) == (archived_cell // k)).all(axis=1).mean()
            ),
            "same_cell_fraction_verified": (
                float(
                    ((verified_cell[keep] // k) == (archived_cell[keep] // k))
                    .all(axis=1)
                    .mean()
                )
                if keep.any()
                else None
            ),
        }
        for k in (1, 2, 4, 5)
        if dims[0] % k == 0 and dims[1] % k == 0
    ]

    def cells(mask):
        return len(np.unique(data["index"][mask])) if mask.any() else 0

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
        # The same counts re-binned at the resolution the descriptor supports.
        # Quote these first: a cell count on a grid finer than the measurement
        # is a count of quantization, not of behaviours.
        "resolvable_grid": list(resolvable_dims),
        "descriptor_median_se": sigma_median.tolist(),
        "resolvable_cells_over_0.25m": resolvable_cells(
            keep & (median_displacement >= 0.25)
        ),
        "resolvable_cells_over_0.50m": resolvable_cells(
            keep & (median_displacement >= 0.50)
        ),
        "resolvable_cells_over_1.00m": resolvable_cells(
            keep & (median_displacement >= 1.00)
        ),
        # --- the cost of inserting on a single sample ---
        "archive_optimism_mean_m": float(np.mean(archived - median_displacement)),
        "archive_optimism_median_m": float(np.median(archived - median_displacement)),
        "within_elite_displacement_sd_m": float(displacement.std(axis=1).mean()),
        "descriptor_axes": list(descriptor.names),
        "within_elite_measure_sd": [
            float(measures[:, :, 0].std(axis=1).mean()),
            float(measures[:, :, 1].std(axis=1).mean()),
        ],
        # --- does an elite stay in the cell it was filed under? ---
        "cell_stable_fraction": float(stable.mean()),
        "cell_within_one_fraction": float(near.mean()),
        "cell_stable_fraction_verified": (
            float(stable[keep].mean()) if keep.any() else None
        ),
        "cell_within_one_fraction_verified": (
            float(near[keep].mean()) if keep.any() else None
        ),
        "mean_cell_offset": float(cell_offset.mean()),
        "mean_cell_offset_per_axis": per_axis_offset.mean(axis=0).tolist(),
        "cell_resolution_sweep": resolution_sweep,
    }

    # A single pass/fail threshold hides how the archive is distributed, and
    # picking one after seeing the numbers is how a report flatters itself. The
    # sweep is reported whole, and it is the same sweep for every archive.
    summary["threshold_sweep"] = [
        {
            "min_survival": float(t),
            "elites": int((survival_rate >= t).sum()),
            "cells_over_0.25m": cells((survival_rate >= t) & (median_displacement >= 0.25)),
            "cells_over_0.50m": cells((survival_rate >= t) & (median_displacement >= 0.50)),
            "resolvable_cells_over_0.25m": resolvable_cells(
                (survival_rate >= t) & (median_displacement >= 0.25)
            ),
            "resolvable_cells_over_0.50m": resolvable_cells(
                (survival_rate >= t) & (median_displacement >= 0.50)
            ),
            "best_median_m": (
                float(median_displacement[survival_rate >= t].max())
                if (survival_rate >= t).any()
                else None
            ),
        }
        for t in (0.5, 0.625, 0.75, 0.875, 1.0)
    ]
    print(f"\nsurvivors by verification strictness (of {n} raw elites):")
    print(
        f"  cells are reported as resolvable ({resolvable_dims[0]}x"
        f"{resolvable_dims[1]}) / raw ({dims[0]}x{dims[1]})"
    )
    print(f"  {'needs':>12} {'elites':>7} {'>=0.25m':>11} {'>=0.50m':>11} {'best m':>8}")
    for row in summary["threshold_sweep"]:
        best = row["best_median_m"]
        print(
            f"  {row['min_survival'] * reps:>5.0f}/{reps} replicas "
            f"{row['elites']:>7d} "
            f"{str(row['resolvable_cells_over_0.25m']) + '/' + str(row['cells_over_0.25m']):>11} "
            f"{str(row['resolvable_cells_over_0.50m']) + '/' + str(row['cells_over_0.50m']):>11} "
            + (f"{best:>+8.3f}" if best is not None else f"{'—':>8}")
        )

    print(
        f"\ncell stability on ({descriptor.axis_x}, {descriptor.axis_y}), "
        f"median of {reps} fresh replicas vs the archived cell:\n"
        f"  all {n} elites:      {stable.mean():5.1%} in their own cell, "
        f"{near.mean():5.1%} within one, mean offset {cell_offset.mean():.2f} bins"
    )
    if keep.any():
        print(
            f"  {int(keep.sum())} verified:      {stable[keep].mean():5.1%} in their own "
            f"cell, {near[keep].mean():5.1%} within one"
        )
    print(
        f"  mean offset per axis: {per_axis_offset.mean(axis=0)[0]:.2f} bins on "
        f"{descriptor.axis_x}, {per_axis_offset.mean(axis=0)[1]:.2f} on "
        f"{descriptor.axis_y}"
    )
    print("  effective resolution — same cell after coarsening the grid by k:")
    for row in resolution_sweep:
        v = row["same_cell_fraction_verified"]
        print(
            f"    k={row['coarsen']}  {row['grid'][0]}x{row['grid'][1]}  "
            f"all {row['same_cell_fraction']:5.1%}"
            + (f"  verified {v:5.1%}" if v is not None else "")
        )

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
        f"cells with a verified elite, resolvable ({resolvable_dims[0]}x"
        f"{resolvable_dims[1]}) / raw ({dims[0]}x{dims[1]}) — "
        f"over 0.25 m: {summary['resolvable_cells_over_0.25m']}/"
        f"{summary['verified_cells_over_0.25m']}, "
        f"over 0.50 m: {summary['resolvable_cells_over_0.50m']}/"
        f"{summary['verified_cells_over_0.50m']}, "
        f"over 1.00 m: {summary['resolvable_cells_over_1.00m']}/"
        f"{summary['verified_cells_over_1.00m']}\n"
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
        measures=median_measures[keep],
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
