"""Side-by-side comparison of two archives (vanilla CPG vs PGA-ME).

**Leads with replay numbers, not archived ones.** MAP-Elites keeps the luckiest
sample per cell and this simulator is not bit-reproducible, so archived fitness
is biased upward — and by a genome-dependent amount (see :mod:`qd.replay`).
Ranking two archives from different genome classes on their archived values is
therefore invalid, so every elite of both archives is re-evaluated here and the
table leads with what came back. Archived values are shown underneath, with the
optimism gap made explicit.

**And it leads with *surviving-elite* coverage, not raw coverage.** Raw coverage
counts every filled cell, whatever is in it. Under v1's fall-penalty objective
almost every cell held a policy that falls — the CPG archive covered 85% of the
grid and, re-measured, not one of its top 64 elites survived a full episode — so
raw coverage compared between a penalty archive and a survival-gated one is not
a comparison of anything. Surviving-elite coverage counts only cells whose elite
is a **replay-verified** full-episode survivor, which is like-for-like: it asks
both pipelines the same question, "how much of the behaviour space can this
robot reach without falling over". Raw coverage is still reported, underneath,
with the caveat attached.

    uv run python -m qd.compare_archives \\
        --a logs/qd/map_elites/archive_final.npz --a-label "MAP-Elites (CPG)" \\
        --b logs/qd/pga_me_matched/archive_final.npz --b-label "PGA-ME (MLP)" \\
        --out logs/qd/comparison

Pass ``--no-replay`` to skip re-evaluation and compare archived values only —
valid when both archives use the *same* genome, and misleading otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.common import MEASURE_NAMES, FitnessCfg, load_archive, write_json
from qd.replay import infer_kind, reevaluate


@dataclass
class Args:
    a: Path
    b: Path
    out: Path = Path("logs/qd/comparison")
    a_label: str = "MAP-Elites (CPG)"
    b_label: str = "PGA-ME (MLP)"

    replay: bool = True
    """Re-evaluate every elite. Off compares archived values only."""

    qd_score_offset: float = -5.0
    """Must match the runs' offset, or the QD-scores are not comparable."""

    device: str = "cuda:0"
    max_envs: int = 512
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def _grid(dims: tuple[int, int], index: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.full(dims, np.nan)
    rows, cols = np.unravel_index(index.astype(int), dims)
    grid[rows, cols] = values
    return grid


def _measure(data: dict, args: Args, label: str) -> dict:
    """Archived + (optionally) replayed statistics for one archive."""
    dims = tuple(int(x) for x in data["grid_dims"])
    archived = data["objective"]
    total_cells = dims[0] * dims[1]

    out = {
        "label": label,
        "genome": infer_kind(data["solution"]),
        "total_cells": total_cells,
        "elites": len(archived),
        "coverage": len(archived) / total_cells,
        "archived_qd_score": float(np.sum(archived - args.qd_score_offset)),
        "archived_best": float(archived.max()),
        "archived_mean": float(archived.mean()),
    }

    if not args.replay:
        out["grid"] = _grid(dims, data["index"], archived)
        return out

    print(f"  re-evaluating {len(archived)} elites of {label}...", flush=True)
    fitness, measures, info, control_dt = reevaluate(
        data["solution"], out["genome"], args.fitness, args.device, args.max_envs
    )
    del measures
    upright_s = info["alive_steps"] * control_dt
    survived = ~info["fell"]

    out.update(
        {
            # The headline structural number: cells whose elite is a
            # replay-verified survivor. Every elite occupies a distinct cell
            # (that is what an archive is), so this counts cells.
            "surviving_elites": int(survived.sum()),
            "surviving_coverage": float(survived.sum()) / total_cells,
            # ...and how many of those actually go somewhere. The spread
            # criterion for walking-v2, at the threshold it was set at.
            "surviving_cells_over_0.25m": int(
                np.sum(survived & (info["displacement"] >= 0.25))
            ),
            "surviving_cells_over_0.50m": int(
                np.sum(survived & (info["displacement"] >= 0.50))
            ),
            "replay_qd_score": float(np.sum(fitness - args.qd_score_offset)),
            "replay_best": float(fitness.max()),
            "replay_mean": float(fitness.mean()),
            "replay_positive_elites": int((fitness > 0).sum()),
            "survived_full_episode": int(survived.sum()),
            "max_upright_s": float(upright_s.max()),
            "median_upright_s": float(np.median(upright_s)),
            "max_displacement_m": float(info["displacement"].max()),
            # Distance covered by the elites that actually stayed up — the test
            # of "balances but barely locomotes".
            "max_displacement_of_survivors_m": (
                float(info["displacement"][survived].max()) if survived.any() else None
            ),
            "median_displacement_of_survivors_m": (
                float(np.median(info["displacement"][survived])) if survived.any() else None
            ),
            "archive_optimism_mean": float(np.mean(archived - fitness)),
            "archive_optimism_median": float(np.median(archived - fitness)),
            "grid": _grid(dims, data["index"], fitness),
        }
    )
    return out


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_data, b_data = load_archive(args.a), load_archive(args.b)
    if list(a_data["grid_dims"]) != list(b_data["grid_dims"]):
        raise ValueError("archives use different grids; they are not comparable")

    a = _measure(a_data, args, args.a_label)
    b = _measure(b_data, args, args.b_label)
    grid_a, grid_b = a.pop("grid"), b.pop("grid")
    basis = "replay" if args.replay else "archived"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vmin = float(np.nanmin([np.nanmin(grid_a), np.nanmin(grid_b)]))
    vmax = float(np.nanmax([np.nanmax(grid_a), np.nanmax(grid_b)]))
    extent = (0.0, 1.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=140)
    for ax, grid, s in ((axes[0], grid_a, a), (axes[1], grid_b, b)):
        im = ax.imshow(grid.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                       cmap="viridis", aspect="equal")
        surviving = (
            f"  survivors {s['surviving_elites']} "
            f"({s['surviving_coverage'] * 100:.1f}% of the grid)"
            if "surviving_elites" in s
            else ""
        )
        ax.set_title(
            f"{s['label']} — {basis} fitness\n"
            f"raw coverage {s['coverage'] * 100:.1f}%{surviving}\n"
            f"QD {s[basis + '_qd_score']:.0f}  best {s[basis + '_best']:+.3f} m"
        )
        ax.set_xlabel(MEASURE_NAMES[0].replace("_", " "))
        ax.set_ylabel(MEASURE_NAMES[1].replace("_", " "))
        fig.colorbar(im, ax=ax, label="fitness [m]")

    filled_a, filled_b = ~np.isnan(grid_a), ~np.isnan(grid_b)
    diff = np.where(filled_b, np.nan_to_num(grid_b), np.nan) - np.where(
        filled_a, np.nan_to_num(grid_a), 0.0
    )
    lim = float(np.nanmax(np.abs(diff))) if np.any(~np.isnan(diff)) else 1.0
    im = axes[2].imshow(diff.T, origin="lower", extent=extent, vmin=-lim, vmax=lim,
                        cmap="RdBu_r", aspect="equal")
    only_b = int(np.sum(filled_b & ~filled_a))
    only_a = int(np.sum(filled_a & ~filled_b))
    axes[2].set_title(f"{b['label']} − {a['label']}\n"
                      f"{only_b} cells only in B, {only_a} only in A")
    axes[2].set_xlabel(MEASURE_NAMES[0].replace("_", " "))
    axes[2].set_ylabel(MEASURE_NAMES[1].replace("_", " "))
    fig.colorbar(im, ax=axes[2], label="fitness delta [m]")
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)

    both = filled_a & filled_b
    payload = {
        "basis": basis,
        a["label"]: a,
        b["label"]: b,
        "shared_cells": int(both.sum()),
        "cells_only_in_b": only_b,
        "cells_only_in_a": only_a,
        "mean_delta_on_shared_cells": (
            float(np.mean(grid_b[both] - grid_a[both])) if both.any() else None
        ),
    }
    write_json(out / "comparison.json", payload)

    rows = [
        ("LOCOMOTION — replay-verified survivors only", None, None),
        ("surviving elites", "surviving_elites", "{:d}"),
        ("surviving-elite coverage", "surviving_coverage", "{:.1%}"),
        ("furthest by a survivor", "max_displacement_of_survivors_m", "{:+.3f} m"),
        ("median travel of survivors", "median_displacement_of_survivors_m", "{:+.3f} m"),
        ("cells with a survivor >= 0.25 m", "surviving_cells_over_0.25m", "{:d}"),
        ("cells with a survivor >= 0.50 m", "surviving_cells_over_0.50m", "{:d}"),
        ("HONEST — every elite re-evaluated", None, None),
        ("best-cell fitness", "replay_best", "{:+.4f} m"),
        ("QD-score", "replay_qd_score", "{:.1f}"),
        ("mean fitness", "replay_mean", "{:+.4f} m"),
        ("positive-fitness elites", "replay_positive_elites", "{:d}"),
        ("survived the full episode", "survived_full_episode", "{:d}"),
        ("longest upright", "max_upright_s", "{:.2f} s"),
        ("median upright", "median_upright_s", "{:.2f} s"),
        ("furthest travelled", "max_displacement_m", "{:+.3f} m"),
        ("ARCHIVED — optimistic, shown for reference", None, None),
        ("archived best-cell fitness", "archived_best", "{:+.4f} m"),
        ("archived QD-score", "archived_qd_score", "{:.1f}"),
        ("archive optimism (mean)", "archive_optimism_mean", "{:+.4f} m"),
        ("STRUCTURE — raw, counts fallen elites too", None, None),
        ("elites", "elites", "{:d}"),
        ("raw coverage", "coverage", "{:.1%}"),
    ]
    print(f"\n| metric | {a['label']} | {b['label']} |")
    print("| --- | --- | --- |")
    for name, key, fmt in rows:
        if key is None:
            print(f"| **{name}** | | |")
            continue
        cells = ["—" if s.get(key) is None else fmt.format(s[key]) for s in (a, b)]
        print(f"| {name} | {cells[0]} | {cells[1]} |")
    print(f"\ncells only in {b['label']}: {only_b} | only in {a['label']}: {only_a} "
          f"| shared: {int(both.sum())}")
    print(f"wrote {out / 'comparison.png'} and {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
