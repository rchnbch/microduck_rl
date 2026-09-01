"""Side-by-side comparison of two archives (vanilla CPG vs PGA-ME).

Both pipelines write the same ``archive_*.npz`` format over the same 20x20
duty-factor grid with the same objective, so they are directly comparable on
the three numbers that matter for QD: **coverage** (how much of the behaviour
space was filled), **QD-score** (total quality summed over filled cells), and
**best-cell fitness** (peak quality). Produces one figure with both heatmaps on
a shared colour scale plus a difference map, and a markdown table.

    uv run python -m qd.compare_archives \\
        --a logs/qd/map_elites/archive_final.npz --a-label "MAP-Elites (CPG)" \\
        --b logs/qd/pga_me/archive_final.npz     --b-label "PGA-ME (MLP)" \\
        --out logs/qd/comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from qd.common import MEASURE_NAMES, load_archive, write_json


@dataclass
class Args:
    a: Path
    b: Path
    out: Path = Path("logs/qd/comparison")
    a_label: str = "MAP-Elites (CPG)"
    b_label: str = "PGA-ME (MLP)"
    qd_score_offset: float = -5.0
    """Must match the runs' offset, or the QD-scores are not comparable."""


def _grid(data: dict) -> np.ndarray:
    """Dense ``(rows, cols)`` fitness grid; empty cells are NaN."""
    dims = tuple(int(x) for x in data["grid_dims"])
    grid = np.full(dims, np.nan)
    rows, cols = np.unravel_index(data["index"].astype(int), dims)
    grid[rows, cols] = data["objective"]
    return grid


def summarize(data: dict, offset: float) -> dict:
    grid = _grid(data)
    obj = data["objective"]
    total_cells = grid.size
    return {
        "cells": total_cells,
        "elites": len(obj),
        "coverage": float(len(obj) / total_cells),
        "qd_score": float(np.sum(obj - offset)),
        "best_fitness": float(obj.max()),
        "mean_fitness": float(obj.mean()),
        "positive_fitness_elites": int((obj > 0).sum()),
    }


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = load_archive(args.a), load_archive(args.b)
    grid_a, grid_b = _grid(a), _grid(b)
    if grid_a.shape != grid_b.shape:
        raise ValueError(
            f"archives use different grids ({grid_a.shape} vs {grid_b.shape}); "
            "they are not comparable"
        )

    stats = {
        args.a_label: summarize(a, args.qd_score_offset),
        args.b_label: summarize(b, args.qd_score_offset),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vmin = float(np.nanmin([np.nanmin(grid_a), np.nanmin(grid_b)]))
    vmax = float(np.nanmax([np.nanmax(grid_a), np.nanmax(grid_b)]))
    extent = (0.0, 1.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=140)
    for ax, grid, label in (
        (axes[0], grid_a, args.a_label),
        (axes[1], grid_b, args.b_label),
    ):
        im = ax.imshow(
            grid.T, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
            cmap="viridis", aspect="equal",
        )
        s = stats[label]
        ax.set_title(
            f"{label}\ncoverage {s['coverage'] * 100:.1f}%  "
            f"QD {s['qd_score']:.0f}  best {s['best_fitness']:+.3f} m"
        )
        ax.set_xlabel(MEASURE_NAMES[0].replace("_", " "))
        ax.set_ylabel(MEASURE_NAMES[1].replace("_", " "))
        fig.colorbar(im, ax=ax, label="fitness [m]")

    # Difference map: where does B beat A? NaN-aware, so a cell only one
    # pipeline filled still shows up.
    filled_a, filled_b = ~np.isnan(grid_a), ~np.isnan(grid_b)
    diff = np.where(filled_b, np.nan_to_num(grid_b, nan=0.0), np.nan) - np.where(
        filled_a, np.nan_to_num(grid_a, nan=0.0), 0.0
    )
    lim = float(np.nanmax(np.abs(diff))) if np.any(~np.isnan(diff)) else 1.0
    im = axes[2].imshow(
        diff.T, origin="lower", extent=extent, vmin=-lim, vmax=lim,
        cmap="RdBu_r", aspect="equal",
    )
    only_b = int(np.sum(filled_b & ~filled_a))
    only_a = int(np.sum(filled_a & ~filled_b))
    axes[2].set_title(
        f"{args.b_label} - {args.a_label}\n"
        f"{only_b} cells only in B, {only_a} only in A"
    )
    axes[2].set_xlabel(MEASURE_NAMES[0].replace("_", " "))
    axes[2].set_ylabel(MEASURE_NAMES[1].replace("_", " "))
    fig.colorbar(im, ax=axes[2], label="fitness delta [m]")

    fig.tight_layout()
    fig_path = out / "comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)

    both = filled_a & filled_b
    stats["shared_cells"] = int(both.sum())
    stats["cells_only_in_b"] = only_b
    stats["cells_only_in_a"] = only_a
    stats["mean_delta_on_shared_cells"] = (
        float(np.mean(grid_b[both] - grid_a[both])) if both.any() else None
    )
    write_json(out / "comparison.json", stats)

    rows = [args.a_label, args.b_label]
    print(f"\n| metric | {rows[0]} | {rows[1]} |")
    print("| --- | --- | --- |")
    for key, fmt in (
        ("elites", "{:d}"), ("coverage", "{:.1%}"), ("qd_score", "{:.1f}"),
        ("best_fitness", "{:+.4f} m"), ("mean_fitness", "{:+.4f} m"),
        ("positive_fitness_elites", "{:d}"),
    ):
        print(
            f"| {key.replace('_', ' ')} | "
            + " | ".join(fmt.format(stats[r][key]) for r in rows)
            + " |"
        )
    print(f"\ncells only in {rows[1]}: {only_b} | only in {rows[0]}: {only_a} "
          f"| shared: {int(both.sum())}")
    if stats["mean_delta_on_shared_cells"] is not None:
        print(f"mean fitness delta on shared cells: "
              f"{stats['mean_delta_on_shared_cells']:+.4f} m")
    print(f"\nwrote {fig_path} and {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
