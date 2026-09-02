"""How long do the archive's best elites actually stay upright?

A v1 archive stores fitness, not survival, and the two are easy to confuse: the
pro-rata fall penalty means a high fitness can come from covering ground fast
before falling *or* from staying up the whole episode. This re-evaluates the
top-N elites and reports displacement, survival fraction and whether they
finish the episode on their feet — the number that says whether an open-loop
CPG can actually walk this robot.

A **walking-v2** archive is survival-gated, so every member was a survivor when
it was inserted and the interesting question changes: how many of them are
*still* survivors on a fresh rollout. That is what ``--sample`` is for — a
uniform draw, not the top of the archive — and ``replay_survival_rate`` in the
JSON is the answer. It will not be 1.0: this simulator's batched contact solve
is order-sensitive and a closed-loop policy amplifies the difference.

    uv run python -m qd.survival_report --archive logs/qd/map_elites/archive_final.npz
    uv run python -m qd.survival_report --archive logs/qd/pga_me_v2/archive_final.npz \\
        --sample 64 --out logs/qd/pga_me_v2/replay_sample.json
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
    sample: int = 0
    """Instead, re-evaluate N elites drawn uniformly at random.

    The top-N view answers "is the best of this archive real". A uniform sample
    answers "is the archive real", which is the question a survival-gated
    archive has to face: every member was a survivor when it was inserted, so
    the honest check is what fraction of a *representative* draw survives a
    fresh rollout. On this simulator that gap is not zero — the batched contact
    solve is order-sensitive and a closed-loop policy amplifies it."""
    sample_seed: int = 0
    replicas: int = 1
    """Evaluate each selected elite this many times and report the MEDIAN.

    On a walking biped one rollout is close to uninformative about
    displacement: `qd.check_repeatability` measures a standard deviation of
    0.6 m across 256 byte-identical copies of one genome, with the trajectories
    diverging past 100 mm inside 0.52 s. MAP-Elites inserts on a single sample
    and keeps the maximum, so an archived value is an estimate of an elite's
    best luck. A median over replicas is an estimate of the elite.

    Survival is far more repeatable than displacement (98.8% on that same
    genome), so `replay_survival_rate` is meaningful at 1 replica; the
    displacement columns are not."""
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
    if args.sample:
        rng = np.random.default_rng(args.sample_seed)
        order = rng.choice(
            len(objective), size=min(args.sample, len(objective)), replace=False
        )
        selection = f"a uniform sample of {len(order)}"
    else:
        order = np.argsort(-objective)[: args.top]
        selection = f"the top {len(order)}"
    batch = solutions[order]

    kind = args.genome
    if kind == "auto":
        kind = infer_kind(batch)

    reps = max(1, args.replicas)
    # Replicas are laid out elite-major (elite 0 x reps, elite 1 x reps, ...)
    # so a reshape recovers the per-elite axis whatever the chunking did.
    tiled = np.repeat(batch, reps, axis=0)
    fitness, measures, info, control_dt = reevaluate(
        tiled, kind, args.fitness, args.device, max_envs=min(512, len(tiled))
    )
    if reps > 1:
        fitness = np.median(fitness.reshape(len(batch), reps), axis=1)
        measures = np.median(measures.reshape(len(batch), reps, -1), axis=1)
        info = {
            # Survival is a rate over replicas, so an elite counts as fallen
            # unless it survived the majority of them.
            "fell": np.mean(info["fell"].reshape(len(batch), reps), axis=1) > 0.5,
            "displacement": np.median(
                info["displacement"].reshape(len(batch), reps), axis=1
            ),
            "alive_steps": np.median(
                info["alive_steps"].reshape(len(batch), reps), axis=1
            ),
            "survival_fraction": np.mean(
                info["survival_fraction"].reshape(len(batch), reps), axis=1
            ),
        }

    total_steps = round(args.fitness.episode_seconds / control_dt)
    survival = info["survival_fraction"]
    alive_s = info["alive_steps"] * control_dt
    survived = ~info["fell"]

    print(
        f"\nre-evaluated {selection} elites of {args.archive} ({kind})"
        + (f", median of {reps} replicas each" if reps > 1 else "")
    )
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
        "selection": selection,
        "replicas_per_elite": reps,
        "survived_full_episode": int(survived.sum()),
        # The replay-honesty number for a survival-gated archive: every one of
        # these was a survivor on insertion, so anything below 1.0 is this
        # simulator's closed-loop non-determinism, not a scoring bug.
        "replay_survival_rate": float(survived.mean()),
        "survivors_over_0.25m": int(
            np.sum(survived & (info["displacement"] >= 0.25))
        ),
        "survivors_over_0.50m": int(
            np.sum(survived & (info["displacement"] >= 0.50))
        ),
        "max_displacement_of_survivors_m": (
            float(info["displacement"][survived].max()) if survived.any() else None
        ),
        "median_displacement_of_survivors_m": (
            float(np.median(info["displacement"][survived])) if survived.any() else None
        ),
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
    if survived.any():
        print(
            f"survivors travel: median {summary['median_displacement_of_survivors_m']:+.3f} m, "
            f"furthest {summary['max_displacement_of_survivors_m']:+.3f} m  |  "
            f"{summary['survivors_over_0.25m']} over 0.25 m, "
            f"{summary['survivors_over_0.50m']} over 0.50 m"
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
