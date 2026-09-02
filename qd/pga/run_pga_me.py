"""PGA-MAP-Elites over a closed-loop MLP genome — walking-v2.

Same 20x20 duty-factor archive, same objective (+x displacement), same
behaviour descriptor as Phase 2, so the ``summary.json`` files stay comparable.
Three things changed for v2, and they are the whole job::

    uv run python -m qd.pga.run_pga_me --iterations 200 --batch-size 1024 \
        --seed-genome logs/qd/seeds/ppo_seed.npz

1. **Honest physics.** Every shell collides with the ground and the rollout
   stops at the fall, so no post-fall frame reaches fitness, descriptor,
   replay buffer or video. See :class:`qd.evaluate.HarnessCfg.full_collision`.

2. **A survival gate.** Only solutions upright for the *whole* episode are
   inserted. Not-falling is a feasibility constraint, not a penalty term — v1
   made it a penalty and the archive filled with optimal divers, because
   "dive well" really was the optimum of ``displacement - 0.25 * time_down``.
   Inside a gated archive ``fallen_fraction`` is 0 for every member, so the
   pro-rata penalty is arithmetically inert and the archived objective *is*
   forward displacement — with v1's formula left untouched, which is what
   keeps the two archives comparable at all.

3. **A seeded start.** ``--seed-genome`` inserts the PPO walker, distilled into
   the genome architecture (see :mod:`qd.seed`), plus a cloud of jittered
   variants, so iteration 1 begins inside the feasible set instead of hunting
   for it.

Each iteration:

1. train the TD3 critic (and the greedy actor) on the shared replay buffer;
2. build a batch of offspring — half by iso+lineDD variation between two random
   elites, half by taking policy-gradient steps on copies of random elites;
3. evaluate every offspring **plus the greedy actor** in one batched rollout,
   collecting transitions into the buffer;
4. insert the survivors, logging **per operator** both the feasibility rate
   (what fraction stayed up) and the insertion rate. A PG insertion rate near
   zero means the critic or the reward wiring is broken rather than that PG
   variation "did not help"; a feasibility rate collapsing to zero means the
   gate has closed on the search and the run needs the episode-length ramp.
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
from qd.pga.evaluate import (
    PolicyHarnessCfg,
    PolicyRolloutHarness,
    ShapedRewardCfg,
)
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.td3 import Td3Cfg, Td3Trainer
from qd.pga.variation import (
    ISO_SIGMA,
    LINE_SIGMA,
    isoline_variation,
    pg_variation,
    sample_parents,
)
from qd.seed import SeedCfg, seed_family


@dataclass
class Args:
    out_dir: Path = Path("logs/qd/pga_me")

    iterations: int = 200
    batch_size: int = 100
    """Offspring per iteration. One extra world holds the greedy actor, so the
    simulation runs ``batch_size + 1`` environments."""

    initial_solutions: int = 200
    """Randomly initialised MLPs evaluated to seed the archive and the buffer.

    With ``--seed-genome`` these are still evaluated — they fill the replay
    buffer with the failure modes the critic has to learn to avoid — but under
    the survival gate essentially none of them will be *inserted*."""

    seed_genome: Path | None = None
    """``.npz`` written by :mod:`qd.seed`: the PPO walker as one genome.

    Inserted at iteration 0 together with ``seeding.jitter_count`` mutated
    copies, so the archive opens with a feasible, actually-walking population."""

    seeding: SeedCfg = field(default_factory=SeedCfg)

    proportion_mutation_ga: float = 0.5
    """Share of offspring made by GA variation; the rest get PG variation."""

    iso_sigma: float = ISO_SIGMA
    line_sigma: float = LINE_SIGMA

    grid_dims: tuple[int, int] = DEFAULT_GRID_DIMS
    seed: int = 0
    device: str = "cuda:0"

    full_collision: bool = True
    """Walking-v2 honest physics; see :class:`qd.evaluate.HarnessCfg`."""

    survival_gate: bool = True
    """Insert only solutions upright for the whole episode.

    ``False`` reproduces v1's penalty-only archive."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)
    reward: ShapedRewardCfg = field(default_factory=ShapedRewardCfg)
    td3: Td3Cfg = field(default_factory=Td3Cfg)

    checkpoint_every: int = 25
    qd_score_offset: float | None = None


def _log_row(
    it: int, stats: dict, evals: int, rates: dict, surv: dict, elapsed: float
) -> str:
    return (
        f"it {it:4d} | evals {evals:6d} | elites {int(stats['num_elites']):4d} "
        f"| cov {stats['coverage'] * 100:5.1f}% | QD {stats['qd_score']:9.2f} "
        f"| best {stats['obj_max']:+.3f} m "
        f"| feas {surv['feasible_fraction'] * 100:5.1f}% "
        f"| GA {rates['ga'] * 100:4.1f}% PG {rates['pg'] * 100:4.1f}% "
        f"greedy {rates['greedy']:.0f} "
        f"| upright max {surv['max_upright_s']:5.2f}s survived "
        f"{surv['survived_full_episode']:4d} | {elapsed:6.1f}s"
    )


def _insert(
    archive,
    genomes: torch.Tensor,
    fitness,
    measures,
    survived,
    gate: bool = True,
) -> tuple[float, float]:
    """Insert a block; return ``(insertion_rate, feasible_rate)``.

    Under the survival gate only ``survived`` rows are offered to the archive,
    and the insertion rate is still measured over the **whole** block — the
    denominator v1 used, so the per-operator rates stay comparable across the
    two runs. ``feasible_rate`` is the new number: the share of this operator's
    offspring that stayed upright for the full episode, which is the curve that
    says whether the search is living inside the constraint or bouncing off it.
    """
    attempted = int(genomes.shape[0])
    if attempted == 0:
        return float("nan"), float("nan")
    survived = np.asarray(survived, dtype=bool)
    feasible_rate = float(survived.mean())

    keep = survived if gate else np.ones(attempted, dtype=bool)
    if not keep.any():
        return 0.0, feasible_rate
    status = archive.add(
        genomes.detach().cpu().numpy()[keep],
        np.asarray(fitness)[keep],
        np.asarray(measures)[keep],
    )["status"]
    return float(np.sum(np.asarray(status) > 0) / attempted), feasible_rate


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
        PolicyHarnessCfg(
            num_envs=num_envs,
            device=args.device,
            full_collision=args.full_collision,
        ),
        args.fitness,
        spec,
        args.reward,
    )
    trainer = Td3Trainer(args.td3, args.device, seed=args.seed, spec=spec)

    history: list[dict] = []
    t_start = time.perf_counter()
    evals = 0
    gate = args.survival_gate

    def evaluate_and_insert(block: torch.Tensor) -> tuple[dict, float, float]:
        """Roll a block out, bank its transitions, offer the survivors."""
        fitness, measures, info, transitions = harness.rollout(block)
        trainer.buffer.add(transitions)
        rate, feasible = _insert(
            archive, block, fitness, measures, ~info["fell"], gate=gate
        )
        return info, rate, feasible

    # --- iteration 0: the PPO seed, then random MLPs ------------------------ #
    # Random MLPs are still evaluated with a seed present: under the gate they
    # will not be inserted, but they are precisely the falling-over experience
    # the critic needs in its buffer to learn what NOT to do.
    seed_info: dict = {}
    if args.seed_genome is not None:
        with np.load(args.seed_genome) as f:
            seeds = torch.as_tensor(
                f["genome"], dtype=torch.float32, device=args.device
            ).reshape(-1, spec.genome_dim)
        # `qd.seed` can distil several twist commands into several genomes; the
        # greedy actor gets the first, and each seed contributes its own
        # jittered neighbourhood so the archive opens across the descriptor.
        trainer.set_greedy(seeds[:1])
        family = seed_family(
            seeds,
            args.seeding.jitter_count,
            args.seeding.jitter_sigmas or args.seeding.jitter_sigma,
            generator,
        )
        # The seed family fills one rollout; whatever is left over goes to
        # random MLPs, which cost nothing extra (the world count is fixed) and
        # give the critic its first look at falling.
        block = torch.cat(
            [
                family,
                spec.initial_population(
                    max(0, num_envs - len(family)), generator, args.device
                ),
            ]
        )[:num_envs]
        info, _, _ = evaluate_and_insert(block)
        evals += num_envs
        n_seeds = len(seeds)
        seed_info = {
            "seeds": n_seeds,
            "seeds_survived": int((~info["fell"][:n_seeds]).sum()),
            "seed_displacements_m": info["displacement"][:n_seeds].tolist(),
            "seed_family_size": len(family),
            "seed_family_feasible": int((~info["fell"][: len(family)]).sum()),
            "seed_family_max_displacement_m": float(
                info["displacement"][: len(family)].max()
            ),
        }
        print(
            f"seeds: {seed_info['seeds_survived']}/{n_seeds} survived, "
            f"displacements "
            + " ".join(f"{d:+.2f}" for d in seed_info["seed_displacements_m"])
            + f" m | {seed_info['seed_family_feasible']}/{len(family)} of the "
            f"seed family feasible, furthest "
            f"{seed_info['seed_family_max_displacement_m']:+.3f} m",
            flush=True,
        )

    remaining = args.initial_solutions
    surv: dict = {}
    feasible = float("nan")
    while remaining > 0:
        block = spec.initial_population(num_envs, generator, args.device)
        info, _, feasible = evaluate_and_insert(block)
        surv = survival_summary(info, harness.control_dt)
        evals += min(remaining, num_envs)
        remaining -= num_envs

    surv = surv or survival_summary(info, harness.control_dt)
    surv["feasible_fraction"] = feasible
    stats = archive_stats(archive)
    rates = {"ga": float("nan"), "pg": float("nan"), "greedy": 0.0}
    print(_log_row(0, stats, evals, rates, surv, time.perf_counter() - t_start), flush=True)
    history.append({"iteration": 0, "evaluations": evals, "buffer": len(trainer.buffer),
                    "elapsed_s": time.perf_counter() - t_start,
                    **seed_info, **surv, **stats})
    if int(stats["num_elites"]) == 0:
        print(
            "WARNING: the archive is empty after iteration 0. Under the "
            "survival gate the search has no feasible parent to vary and GA "
            "variation will fall back to random MLPs. Seed it "
            "(`--seed-genome`) or ramp the episode length.",
            flush=True,
        )

    # --- PGA-ME loop -------------------------------------------------------- #
    num_ga = round(args.proportion_mutation_ga * args.batch_size)
    num_pg = args.batch_size - num_ga
    print(f"offspring split: {num_ga} GA + {num_pg} PG + 1 greedy actor", flush=True)

    for it in range(1, args.iterations + 1):
        losses = trainer.train()

        ga = isoline_variation(
            sample_parents(archive, num_ga, generator, args.device, spec=spec),
            sample_parents(archive, num_ga, generator, args.device, spec=spec),
            generator,
            iso_sigma=args.iso_sigma,
            line_sigma=args.line_sigma,
        )
        pg = pg_variation(
            sample_parents(archive, num_pg, generator, args.device, spec=spec),
            trainer,
            spec=spec,
        )
        greedy = trainer.greedy_genome()
        population = torch.cat([ga, pg, greedy])

        fitness, measures, info, transitions = harness.rollout(population)
        trainer.buffer.add(transitions)
        evals += population.shape[0]
        survived = ~info["fell"]

        blocks = {
            "ga": (ga, slice(0, num_ga)),
            "pg": (pg, slice(num_ga, num_ga + num_pg)),
            "greedy": (greedy, slice(len(population) - 1, len(population))),
        }
        rates, feasible_rates = {}, {}
        for name, (block, sl) in blocks.items():
            rates[name], feasible_rates[name] = _insert(
                archive, block, fitness[sl], measures[sl], survived[sl], gate=gate
            )

        stats = archive_stats(archive)
        surv = survival_summary(info, harness.control_dt)
        surv["feasible_fraction"] = float(survived.mean())
        elapsed = time.perf_counter() - t_start
        history.append(
            {
                "iteration": it,
                "evaluations": evals,
                "buffer": len(trainer.buffer),
                "ga_insert_rate": rates["ga"],
                "pg_insert_rate": rates["pg"],
                "ga_feasible_rate": feasible_rates["ga"],
                "pg_feasible_rate": feasible_rates["pg"],
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

    mean = lambda key: (
        float(np.mean([h[key] for h in history if key in h]))
        if any(key in h for h in history)
        else None
    )
    write_json(
        out / "summary.json",
        {
            "algorithm": "PGA-MAP-Elites (iso+lineDD GA + TD3 policy-gradient variation)",
            "solution_dim": spec.genome_dim,
            "evaluations": evals,
            "wall_clock_s": time.perf_counter() - t_start,
            "survival_gate": gate,
            "full_collision": args.full_collision,
            "seed_genome": str(args.seed_genome) if args.seed_genome else None,
            "mean_ga_insert_rate": mean("ga_insert_rate"),
            "mean_pg_insert_rate": mean("pg_insert_rate"),
            # The curve this run is about: what share of offspring cleared the
            # survival constraint. A gated run whose feasibility rate stays
            # near zero is a run whose archive cannot grow, whatever its
            # insertion rates say.
            "mean_ga_feasible_rate": mean("ga_feasible_rate"),
            "mean_pg_feasible_rate": mean("pg_feasible_rate"),
            "mean_feasible_fraction": mean("feasible_fraction"),
            "final_feasible_fraction": history[-1].get("feasible_fraction"),
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
        # Read by qd.compare_archives / qd.survival_report so a v1 archive and
        # a v2 archive are never silently reported as the same kind of thing.
        "survival_gate": args.survival_gate,
        "full_collision": args.full_collision,
        "seeded": args.seed_genome is not None,
    }


if __name__ == "__main__":
    main()
