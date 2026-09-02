"""How hard can PG variation push a *walker* before it stops walking?

QDax's PGA-ME defaults — 100 Adam steps at lr 1e-3 for offspring, 300 at 3e-4
for the greedy actor — are tuned for a search that starts from random policies,
where "the offspring is different now" is the whole point and there is nothing
to break. Walking-v2 starts from a distilled PPO walker and only inserts
full-episode survivors, and in that regime those defaults are destructive:
Adam's per-parameter step is bounded by the learning rate, so 100 steps at 1e-3
moves every one of the 9038 weights by up to 0.1, against an initial weight
scale of ~1/sqrt(61) = 0.13. That is a ~70% perturbation of the whole policy.
The v2 validation run measured the consequence directly — GA offspring feasible
40% of the time, PG offspring **0.0%**, iteration after iteration.

This script measures the trade-off instead of guessing at it: fill the buffer
from one real generation, train the critic exactly as an iteration would, then
apply PG variation to copies of the seed at a grid of (steps, learning rate)
and evaluate every variant in one batched rollout.

Two numbers per setting. **Feasibility** is the constraint — a setting whose
offspring never survive contributes nothing to a gated archive no matter how
good its Q. **Displacement** is what the archive ranks. A setting that keeps
feasibility and moves the genome measurably is a setting where the gradient
half of PGA-ME is doing work.

    uv run python -m qd.pga.tune_pg --seed-genome logs/qd/seeds/ppo_seed.npz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, write_json
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness, ShapedRewardCfg
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.td3 import Td3Cfg, Td3Trainer
from qd.pga.variation import pg_variation
from qd.seed import SeedCfg, jitter


@dataclass
class Args:
    seed_genome: Path = Path("logs/qd/seeds/ppo_seed.npz")
    out: Path = Path("logs/qd/pg_tuning.json")

    num_envs: int = 256
    per_setting: int = 20
    """Offspring evaluated per (steps, lr) cell."""

    steps_grid: tuple[int, ...] = (10, 30, 100)
    lr_grid: tuple[float, ...] = (3e-5, 1e-4, 3e-4, 1e-3)

    device: str = "cuda:0"
    seed: int = 0
    seeding: SeedCfg = field(default_factory=SeedCfg)
    fitness: FitnessCfg = field(default_factory=FitnessCfg)
    reward: ShapedRewardCfg = field(default_factory=ShapedRewardCfg)
    td3: Td3Cfg = field(default_factory=Td3Cfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    spec = DEFAULT_SPEC
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(num_envs=args.num_envs, device=args.device),
        args.fitness,
        spec,
        args.reward,
    )
    trainer = Td3Trainer(args.td3, args.device, seed=args.seed, spec=spec)

    with np.load(args.seed_genome) as f:
        seed = torch.as_tensor(
            f["genome"], dtype=torch.float32, device=args.device
        ).reshape(1, -1)
    trainer.set_greedy(seed)

    # One real generation of experience: the seed family plus random MLPs,
    # exactly what iteration 0 of a seeded run banks.
    family = torch.cat(
        [seed, jitter(seed, args.seeding.jitter_count, args.seeding.jitter_sigma,
                      generator)]
    )
    block = torch.cat(
        [family, spec.initial_population(args.num_envs - len(family), generator,
                                         args.device)]
    )[: args.num_envs]
    _, _, info, transitions = harness.rollout(block)
    trainer.buffer.add(transitions)
    print(
        f"buffer {len(trainer.buffer)} transitions from one generation "
        f"({int((~info['fell']).sum())} survivors); training the critic "
        f"{args.td3.num_critic_training_steps} steps",
        flush=True,
    )
    trainer.train()

    settings = [(s, lr) for s in args.steps_grid for lr in args.lr_grid]
    capacity = args.num_envs // args.per_setting
    if len(settings) > capacity:
        raise SystemExit(
            f"{len(settings)} settings x {args.per_setting} offspring exceeds "
            f"{args.num_envs} worlds; raise --num-envs or shrink the grid"
        )

    # Every setting mutates the SAME parents, so the rows differ only in the
    # step size — the parents' own spread cannot explain a difference.
    parents = jitter(seed, args.per_setting, args.seeding.jitter_sigma, generator)
    offspring, drift = [], []
    for steps, lr in settings:
        cfg = Td3Cfg(**{**trainer.cfg.__dict__, "policy_learning_rate": lr})
        scoped = Td3Trainer.__new__(Td3Trainer)
        scoped.__dict__.update(trainer.__dict__)
        scoped.cfg = cfg
        child = pg_variation(parents, scoped, steps=steps, spec=spec)
        offspring.append(child)
        drift.append(float((child - parents).abs().mean()))

    population = torch.cat(offspring)
    pad = args.num_envs - population.shape[0]
    if pad:
        population = torch.cat([population, parents[:1].expand(pad, -1)])
    fitness, _, info, _ = harness.rollout(population, collect=False)

    # The unmutated parents, under the same physics, as the baseline every row
    # has to be read against.
    base_f, _, base_info, _ = harness.rollout(
        torch.cat([parents, parents[:1].expand(args.num_envs - len(parents), -1)]),
        collect=False,
    )
    base_feasible = float((~base_info["fell"][: len(parents)]).mean())
    base_displ = float(base_info["displacement"][: len(parents)].mean())
    print(
        f"\nparents (no PG): feasible {base_feasible * 100:.0f}%, "
        f"mean displacement {base_displ:+.3f} m\n"
    )
    print(f"{'steps':>5} {'lr':>8} {'feasible':>9} {'mean_displ':>11} "
          f"{'best_displ':>11} {'|dgenome|':>10}")

    rows = []
    for i, (steps, lr) in enumerate(settings):
        sl = slice(i * args.per_setting, (i + 1) * args.per_setting)
        feasible = ~info["fell"][sl]
        row = {
            "steps": steps,
            "learning_rate": lr,
            "feasible_rate": float(feasible.mean()),
            "mean_displacement_m": float(info["displacement"][sl].mean()),
            "best_displacement_m": float(info["displacement"][sl].max()),
            "mean_abs_genome_delta": drift[i],
            "mean_fitness_m": float(fitness[sl].mean()),
        }
        rows.append(row)
        print(
            f"{steps:>5} {lr:>8.0e} {row['feasible_rate'] * 100:>8.0f}% "
            f"{row['mean_displacement_m']:>+11.3f} "
            f"{row['best_displacement_m']:>+11.3f} "
            f"{drift[i]:>10.2e}"
        )

    write_json(
        args.out,
        {
            "baseline_parents": {
                "feasible_rate": base_feasible,
                "mean_displacement_m": base_displ,
                "mean_fitness_m": float(base_f[: len(parents)].mean()),
            },
            "critic_training_steps": args.td3.num_critic_training_steps,
            "buffer": len(trainer.buffer),
            "settings": rows,
        },
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
