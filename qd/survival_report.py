"""How long do the archive's best elites actually stay upright?

The archive stores fitness, not survival, and the two are easy to confuse: the
pro-rata fall penalty means a high fitness can come from covering ground fast
before falling *or* from staying up the whole episode. This re-evaluates the
top-N elites and reports displacement, survival fraction and whether they
finish the episode on their feet — the number that says whether an open-loop
CPG can actually walk this robot.

    uv run python -m qd.survival_report --archive logs/qd/map_elites/archive_final.npz
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
    top: int = 64
    """Re-evaluate the N highest-fitness elites."""
    device: str = "cuda:0"
    out: Path | None = None
    """Optional JSON destination for the per-elite table."""
    genome: str = "auto"
    """'cpg', 'mlp', or 'auto' to infer from the archive's solution width."""
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_archive(args.archive)
    solutions, objective = data["solution"], data["objective"]
    order = np.argsort(-objective)[: args.top]
    batch = solutions[order]

    kind = args.genome
    if kind == "auto":
        kind = infer_kind(batch)
    fitness, measures, info, control_dt = reevaluate(
        batch, kind, args.fitness, args.device, max_envs=len(batch)
    )

    total_steps = round(args.fitness.episode_seconds / control_dt)
    survival = info["survival_fraction"]
    alive_s = info["alive_steps"] * control_dt
    survived = ~info["fell"]

    print(f"\nre-evaluated the top {len(batch)} elites of {args.archive} ({kind})")
    print(f"episode {args.fitness.episode_seconds:.1f} s = {total_steps} control steps\n")
    print(f"{'rank':>4} {'archived':>9} {'replay':>9} {'displ_m':>9} "
          f"{'upright_s':>10} {'survived':>9} {'duty_L':>7} {'duty_R':>7}")
    for i in range(min(15, len(batch))):
        print(
            f"{i:>4} {objective[order[i]]:>+9.4f} {fitness[i]:>+9.4f} "
            f"{info['displacement'][i]:>+9.4f} {alive_s[i]:>10.2f} "
            f"{bool(survived[i])!s:>9} {measures[i, 0]:>7.3f} {measures[i, 1]:>7.3f}"
        )

    # Archive optimism: MAP-Elites keeps the BEST sample per cell, and this sim
    # is not bit-reproducible, so archived fitness is biased upward by whatever
    # luck that cell got. The size of the bias is a property of the genome, not
    # a constant — a closed-loop policy amplifies a contact-solve difference
    # through its feedback loop, an open-loop CPG replays the same joint
    # trajectory regardless — so it has to be measured per archive before two
    # archives can be compared on fitness at all.
    optimism = objective[order] - fitness
    summary = {
        "archive": str(args.archive),
        "genome": kind,
        "elites_evaluated": len(batch),
        "episode_seconds": args.fitness.episode_seconds,
        "survived_full_episode": int(survived.sum()),
        "max_upright_seconds": float(alive_s.max()),
        "median_upright_seconds": float(np.median(alive_s)),
        "max_displacement_m": float(info["displacement"].max()),
        "max_replay_fitness_m": float(fitness.max()),
        "mean_survival_fraction": float(survival.mean()),
        "archived_best_m": float(objective[order].max()),
        "mean_archive_optimism_m": float(optimism.mean()),
        "median_archive_optimism_m": float(np.median(optimism)),
        "max_archive_optimism_m": float(optimism.max()),
    }
    print(
        f"\nsurvived the full episode: {summary['survived_full_episode']}/{len(batch)}"
        f"  |  longest upright {summary['max_upright_seconds']:.2f} s"
        f"  |  furthest {summary['max_displacement_m']:+.3f} m"
    )
    print(
        f"archive optimism (archived - replay): mean {optimism.mean():+.4f} m, "
        f"median {np.median(optimism):+.4f} m, max {optimism.max():+.4f} m"
    )
    print(
        f"best fitness: {objective[order].max():+.4f} m archived vs "
        f"{fitness.max():+.4f} m on replay"
    )
    if args.out is not None:
        write_json(args.out, summary)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
