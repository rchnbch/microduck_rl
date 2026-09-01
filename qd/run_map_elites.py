"""Vanilla MAP-Elites over the open-loop CPG genome.

``GridArchive`` + a plain ``GaussianEmitter`` (isotropic-per-dimension Gaussian
mutation of a uniformly drawn elite) — no CMA-ES, no gradients.  Run it with::

    uv run python -m qd.run_map_elites --generations 200 --batch-size 256

``qd/`` is intentionally *not* part of the installed distribution (see
``qd/README.md``), so there is no console script; ``-m`` puts the repo root on
``sys.path``, which is what the imports need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd import cpg_genome
from qd.common import (
    DEFAULT_GRID_DIMS,
    FitnessCfg,
    archive_stats,
    make_archive,
    plot_archive,
    save_archive,
    write_json,
)
from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness


@dataclass
class Args:
    """CLI for the Phase-2 MAP-Elites run."""

    out_dir: Path = Path("logs/qd/map_elites")
    """Archive checkpoints, heatmaps and the metrics log land here."""

    generations: int = 200
    """Number of ask/tell iterations after the random seeding population."""

    batch_size: int = 100
    """Solutions the emitter proposes per generation."""

    initial_solutions: int = 200
    """Uniform random genomes evaluated to seed the archive."""

    num_envs: int | None = None
    """Parallel worlds; defaults to ``batch_size`` so one generation is exactly
    one batched rollout. Fixed once the sim is built (the CUDA graph pins it),
    and batches are chunked/padded to it — setting it *above* ``batch_size``
    just simulates padding, so raise ``batch_size`` instead."""

    sigma_fraction: float = 0.1
    """Gaussian mutation sigma, as a fraction of each parameter's range."""

    grid_dims: tuple[int, int] = DEFAULT_GRID_DIMS
    """Archive resolution over (left duty factor, right duty factor)."""

    seed: int = 0
    device: str = "cuda:0"

    fitness: FitnessCfg = field(default_factory=FitnessCfg)

    checkpoint_every: int = 20
    """Generations between archive checkpoints and heatmaps (0 disables)."""

    qd_score_offset: float | None = None
    """Objective baseline for the QD-score. Defaults to ``fitness.min_fitness``
    so every insertable elite contributes a non-negative amount."""


def _log_row(gen: int, stats: dict[str, float], evals: int, elapsed: float) -> str:
    return (
        f"gen {gen:4d} | evals {evals:6d} | elites {int(stats['num_elites']):4d} "
        f"| coverage {stats['coverage'] * 100:5.1f}% | QD {stats['qd_score']:9.2f} "
        f"| best {stats['obj_max']:+.3f} m | {elapsed:6.1f}s"
    )


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    space = cpg_genome.genome_space()
    offset = (
        args.qd_score_offset
        if args.qd_score_offset is not None
        else args.fitness.min_fitness
    )
    archive = make_archive(
        solution_dim=space.dim,
        grid_dims=args.grid_dims,
        qd_score_offset=offset,
        seed=args.seed,
    )

    num_envs = args.num_envs if args.num_envs is not None else args.batch_size
    harness = MicroduckRolloutHarness(
        HarnessCfg(num_envs=num_envs, device=args.device), args.fitness
    )
    evaluator = CpgEvaluator(harness)

    history: list[dict] = []
    t_start = time.perf_counter()

    # --- seeding: uniform random genomes ---------------------------------- #
    init = space.sample(args.initial_solutions, rng)
    fitness, measures, info = evaluator.evaluate(init)
    archive.add(init, fitness, measures)
    evals = len(init)
    stats = archive_stats(archive)
    print(_log_row(0, stats, evals, time.perf_counter() - t_start), flush=True)
    history.append(
        {
            "generation": 0,
            "evaluations": evals,
            "fell_fraction": float(np.mean(info["fell"])),
            "elapsed_s": time.perf_counter() - t_start,
            **stats,
        }
    )

    # --- MAP-Elites loop --------------------------------------------------- #
    from ribs.emitters import GaussianEmitter
    from ribs.schedulers import Scheduler

    emitter = GaussianEmitter(
        archive,
        sigma=space.sigma(args.sigma_fraction),
        # Only used if the archive is somehow still empty; the seeding above
        # normally fills it first.
        x0=0.5 * (space.lower + space.upper),
        bounds=space.bounds,  # bound enforcement #1 (#2 is in CpgEvaluator)
        batch_size=args.batch_size,
        seed=args.seed,
    )
    scheduler = Scheduler(archive, [emitter])

    for gen in range(1, args.generations + 1):
        solutions = scheduler.ask()
        fitness, measures, info = evaluator.evaluate(solutions)
        scheduler.tell(fitness, measures)
        evals += len(solutions)

        stats = archive_stats(archive)
        elapsed = time.perf_counter() - t_start
        history.append(
            {
                "generation": gen,
                "evaluations": evals,
                "fell_fraction": float(np.mean(info["fell"])),
                "elapsed_s": elapsed,
                **stats,
            }
        )
        print(_log_row(gen, stats, evals, elapsed), flush=True)

        if args.checkpoint_every and gen % args.checkpoint_every == 0:
            save_archive(archive, out / f"archive_gen{gen:04d}.npz", _meta(args, gen, evals))
            plot_archive(archive, out / f"heatmap_gen{gen:04d}.png", f"CPG MAP-Elites (gen {gen})")
            write_json(out / "history.json", history)

    save_archive(archive, out / "archive_final.npz", _meta(args, args.generations, evals))
    plot_archive(archive, out / "heatmap_final.png", "CPG MAP-Elites (final)")
    write_json(out / "history.json", history)
    write_json(
        out / "summary.json",
        {
            "algorithm": "MAP-Elites (GaussianEmitter, open-loop CPG genome)",
            "solution_dim": space.dim,
            "evaluations": evals,
            "wall_clock_s": time.perf_counter() - t_start,
            "args": args,
            **archive_stats(archive),
        },
    )
    print(f"\nwrote {out}/archive_final.npz and {out}/heatmap_final.png", flush=True)


def _meta(args: Args, gen: int, evals: int) -> dict:
    return {
        "algorithm": "map_elites_cpg",
        "generation": gen,
        "evaluations": evals,
        "genome": "cpg31",
        "joint_names": list(cpg_genome.LEG_JOINT_NAMES),
        "episode_seconds": args.fitness.episode_seconds,
        "settle_seconds": args.fitness.settle_seconds,
        "fall_penalty": args.fitness.fall_penalty,
    }


if __name__ == "__main__":
    main()
