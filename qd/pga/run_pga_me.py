"""PGA-MAP-Elites over a closed-loop MLP genome.

Same 20x20 duty-factor archive, same objective, same behaviour descriptor as
Phase 2 — so the two ``summary.json`` files are directly comparable. Run it::

    uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 100

Each iteration:

1. train the TD3 critic (and the greedy actor) on the shared replay buffer;
2. build a batch of offspring — half by iso+lineDD variation between two random
   elites, half by taking policy-gradient steps on copies of random elites;
3. evaluate every offspring **plus the greedy actor** in one batched rollout,
   collecting transitions into the buffer;
4. insert everything into the archive, recording the insertion rate **per
   operator**, because a PG insertion rate near zero means the critic or the
   reward wiring is broken rather than that PG variation "did not help".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import (
    DEFAULT_GRID_DIMS,
    FitnessCfg,
    archive_stats,
    make_archive,
    plot_archive,
    save_archive,
    survival_summary,
    write_json,
)
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.td3 import Td3Cfg, Td3Trainer
from qd.pga.variation import (
    ISO_SIGMA,
    LINE_SIGMA,
    isoline_variation,
    pg_variation,
    sample_parents,
)


@dataclass
class Args:
    out_dir: Path = Path("logs/qd/pga_me")

    iterations: int = 200
    batch_size: int = 100
    """Offspring per iteration. One extra world holds the greedy actor, so the
    simulation runs ``batch_size + 1`` environments."""

    initial_solutions: int = 200
    """Randomly initialised MLPs evaluated to seed the archive and the buffer."""

    proportion_mutation_ga: float = 0.5
    """Share of offspring made by GA variation; the rest get PG variation."""

    iso_sigma: float = ISO_SIGMA
    line_sigma: float = LINE_SIGMA

    grid_dims: tuple[int, int] = DEFAULT_GRID_DIMS
    seed: int = 0
    device: str = "cuda:0"

    fitness: FitnessCfg = field(default_factory=FitnessCfg)
    td3: Td3Cfg = field(default_factory=Td3Cfg)

    checkpoint_every: int = 25
    qd_score_offset: float | None = None


def _log_row(
    it: int, stats: dict, evals: int, rates: dict, surv: dict, elapsed: float
) -> str:
    return (
        f"it {it:4d} | evals {evals:6d} | elites {int(stats['num_elites']):4d} "
        f"| cov {stats['coverage'] * 100:5.1f}% | QD {stats['qd_score']:9.2f} "
        f"| best {stats['obj_max']:+.3f} m | GA {rates['ga'] * 100:4.1f}% "
        f"PG {rates['pg'] * 100:4.1f}% greedy {rates['greedy']:.0f} "
        f"| upright max {surv['max_upright_s']:5.2f}s survived {surv['survived_full_episode']:3d} "
        f"| {elapsed:6.1f}s"
    )


def _insert(archive, genomes: torch.Tensor, fitness, measures) -> float:
    """Insert a block and return its archive-insertion rate."""
    if genomes.shape[0] == 0:
        return float("nan")
    status = archive.add(genomes.cpu().numpy(), fitness, measures)["status"]
    return float(np.mean(np.asarray(status) > 0))


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = DEFAULT_SPEC
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    offset = (
        args.qd_score_offset
        if args.qd_score_offset is not None
        else args.fitness.min_fitness
    )
    archive = make_archive(
        solution_dim=spec.genome_dim,
        grid_dims=args.grid_dims,
        qd_score_offset=offset,
        seed=args.seed,
    )

    # +1 world for the greedy actor, evaluated alongside the offspring.
    num_envs = args.batch_size + 1
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(num_envs=num_envs, device=args.device), args.fitness, spec
    )
    trainer = Td3Trainer(args.td3, args.device, seed=args.seed, spec=spec)

    history: list[dict] = []
    t_start = time.perf_counter()
    evals = 0

    # --- seeding: randomly initialised MLPs -------------------------------- #
    remaining = args.initial_solutions
    surv = {}
    while remaining > 0:
        block = spec.initial_population(num_envs, generator, args.device)
        fitness, measures, info, transitions = harness.rollout(block)
        trainer.buffer.add(transitions)
        keep = min(remaining, num_envs)
        _insert(archive, block[:keep], fitness[:keep], measures[:keep])
        surv = survival_summary(info, harness.control_dt)
        evals += keep
        remaining -= keep

    stats = archive_stats(archive)
    rates = {"ga": float("nan"), "pg": float("nan"), "greedy": 0.0}
    print(_log_row(0, stats, evals, rates, surv, time.perf_counter() - t_start), flush=True)
    history.append({"iteration": 0, "evaluations": evals, "buffer": len(trainer.buffer),
                    "elapsed_s": time.perf_counter() - t_start, **surv, **stats})

    # --- PGA-ME loop -------------------------------------------------------- #
    num_ga = round(args.proportion_mutation_ga * args.batch_size)
    num_pg = args.batch_size - num_ga
    print(f"offspring split: {num_ga} GA + {num_pg} PG + 1 greedy actor", flush=True)

    for it in range(1, args.iterations + 1):
        losses = trainer.train()

        ga = isoline_variation(
            sample_parents(archive, num_ga, generator, args.device),
            sample_parents(archive, num_ga, generator, args.device),
            generator,
            iso_sigma=args.iso_sigma,
            line_sigma=args.line_sigma,
        )
        pg = pg_variation(
            sample_parents(archive, num_pg, generator, args.device), trainer, spec=spec
        )
        greedy = trainer.greedy_genome()
        population = torch.cat([ga, pg, greedy])

        fitness, measures, info, transitions = harness.rollout(population)
        trainer.buffer.add(transitions)
        evals += population.shape[0]

        rates = {
            "ga": _insert(archive, ga, fitness[:num_ga], measures[:num_ga]),
            "pg": _insert(
                archive, pg, fitness[num_ga : num_ga + num_pg],
                measures[num_ga : num_ga + num_pg],
            ),
            "greedy": _insert(archive, greedy, fitness[-1:], measures[-1:]),
        }

        stats = archive_stats(archive)
        surv = survival_summary(info, harness.control_dt)
        elapsed = time.perf_counter() - t_start
        history.append(
            {
                "iteration": it,
                "evaluations": evals,
                "buffer": len(trainer.buffer),
                "ga_insert_rate": rates["ga"],
                "pg_insert_rate": rates["pg"],
                "greedy_inserted": rates["greedy"],
                "greedy_fitness": float(fitness[-1]),
                "greedy_survival_fraction": float(info["survival_fraction"][-1]),
                "elapsed_s": elapsed,
                **surv,
                **losses,
                **stats,
            }
        )
        print(_log_row(it, stats, evals, rates, surv, elapsed), flush=True)

        if args.checkpoint_every and it % args.checkpoint_every == 0:
            save_archive(archive, out / f"archive_it{it:04d}.npz", _meta(args, it, evals))
            plot_archive(archive, out / f"heatmap_it{it:04d}.png", f"PGA-ME (it {it})")
            write_json(out / "history.json", history)

    save_archive(archive, out / "archive_final.npz", _meta(args, args.iterations, evals))
    plot_archive(archive, out / "heatmap_final.png", "PGA-MAP-Elites (final)")
    write_json(out / "history.json", history)

    ga_rates = [h["ga_insert_rate"] for h in history if "ga_insert_rate" in h]
    pg_rates = [h["pg_insert_rate"] for h in history if "pg_insert_rate" in h]
    write_json(
        out / "summary.json",
        {
            "algorithm": "PGA-MAP-Elites (iso+lineDD GA + TD3 policy-gradient variation)",
            "solution_dim": spec.genome_dim,
            "evaluations": evals,
            "wall_clock_s": time.perf_counter() - t_start,
            "mean_ga_insert_rate": float(np.mean(ga_rates)) if ga_rates else None,
            "mean_pg_insert_rate": float(np.mean(pg_rates)) if pg_rates else None,
            "greedy_insertions": int(sum(h["greedy_inserted"] for h in history[1:])),
            # The headline number on this robot: a passive HOME hold topples at
            # ~1.34 s, so "did anything stay up for the whole episode" matters
            # more than QD-score alone.
            "best_upright_s": max(h["max_upright_s"] for h in history),
            "total_full_episode_survivors": int(
                sum(h["survived_full_episode"] for h in history)
            ),
            "args": args,
            **archive_stats(archive),
        },
    )
    print(f"\nwrote {out}/archive_final.npz and {out}/heatmap_final.png", flush=True)


def _meta(args: Args, it: int, evals: int) -> dict:
    return {
        "algorithm": "pga_me_mlp",
        "iteration": it,
        "evaluations": evals,
        "genome": f"mlp{DEFAULT_SPEC.obs_dim}-"
                  + "-".join(str(h) for h in DEFAULT_SPEC.hidden)
                  + f"-{DEFAULT_SPEC.action_dim}",
        "episode_seconds": args.fitness.episode_seconds,
        "settle_seconds": args.fitness.settle_seconds,
        "fall_penalty": args.fitness.fall_penalty,
    }


if __name__ == "__main__":
    main()
