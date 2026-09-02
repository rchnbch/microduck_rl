"""What does honest physics cost? Foot-only vs full-body ground collision.

Walking-v2 moved every QD rollout from ``robot_walk.xml`` (ground contact on
the two foot soles) to ``robot_allcollisions.xml`` (every shell), because a
rollout that keeps simulating after the fall needs a floor the robot cannot
sink through. More contact pairs is more constraint-solver work, so the switch
is not free and the size of the bill decides whether it is affordable at the
budget this job has to match.

Measured on the **MLP** harness, which is the one the run uses, at the batch
size the run uses. Reported per genome, and with the fall-stop separately
toggled, because the two changes push the wall clock in opposite directions:
full collision makes each step dearer, stopping at the fall removes most of the
steps (a random MLP is down inside 1.5 s of a 7 s episode).

    uv run python -m qd.bench_collision --batch-size 1024
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import tyro

from qd.common import FitnessCfg, survival_summary
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC


@dataclass
class Args:
    batch_size: int = 1024
    device: str = "cuda:0"
    seed: int = 0
    repeats: int = 2
    collect: bool = True
    """Collect transitions, as a real PGA-ME iteration does."""

    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    spec = DEFAULT_SPEC
    rows = []

    for full_collision in (False, True):
        for fall_check in (0, 25):
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            harness = PolicyRolloutHarness(
                PolicyHarnessCfg(
                    num_envs=args.batch_size,
                    device=args.device,
                    full_collision=full_collision,
                    fall_check_every=fall_check,
                ),
                args.fitness,
                spec,
            )
            # Identical population in every configuration: same seed, same
            # generator state, so the four rows differ only in the physics.
            population = spec.initial_population(
                args.batch_size, generator, args.device
            )
            harness.rollout(population, collect=args.collect)  # warm the graphs

            best = float("inf")
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                _, _, info, _ = harness.rollout(population, collect=args.collect)
                best = min(best, time.perf_counter() - t0)
            surv = survival_summary(info, harness.control_dt)

            rows.append(
                {
                    "full_collision": full_collision,
                    "stops_at_fall": bool(fall_check),
                    "seconds": best,
                    "ms_per_genome": best / args.batch_size * 1000,
                    "survivors": surv["survived_full_episode"],
                    "mean_upright_s": surv["mean_upright_s"],
                }
            )
            print(
                f"collision={'full' if full_collision else 'feet':>4} "
                f"stop_at_fall={bool(fall_check)!s:>5} | "
                f"{best:6.2f} s/generation, {rows[-1]['ms_per_genome']:5.2f} "
                f"ms/genome | survivors {surv['survived_full_episode']:4d} "
                f"mean upright {surv['mean_upright_s']:.2f} s",
                flush=True,
            )
            del harness
            torch.cuda.empty_cache()

    by = {(r["full_collision"], r["stops_at_fall"]): r["seconds"] for r in rows}
    print(
        f"\nfull collision costs {by[(True, False)] / by[(False, False)]:.2f}x "
        f"per generation with the loop running to the end;\n"
        f"stopping at the fall gives {by[(True, False)] / by[(True, True)]:.2f}x "
        f"of that back on a random population.\n"
        f"net v1 -> v2 (feet, no stop) -> (full, stop): "
        f"{by[(True, True)] / by[(False, False)]:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
