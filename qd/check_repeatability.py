"""How repeatable is one genome's fitness? (Answer: for a walker, not at all.)

v1 measured this on an open-loop CPG and got a spread of ~4 mm, then noted that
a closed-loop MLP amplifies it ~60x. Walking-v2's seeds are *walkers*, and a
walking biped is a marginally stable closed-loop system integrating over 350
control steps: the amplification is not 60x, it is enough to make a single
evaluation nearly uninformative about displacement.

The measurement is deliberately blunt. Every domain-randomization knob is off,
the spawn is pinned, the actuator is deterministic, and every world is reset to
byte-identical state — so **N copies of one genome in one batched rollout differ
only by MuJoCo-Warp's contact/constraint solve order**, which is not
deterministic across worlds. Whatever spread comes back is the noise floor of a
single evaluation.

Why it matters more than it looks: MAP-Elites inserts on a *single* sample and
keeps the maximum per cell. If displacement has a ~1 m spread, a cell fills with
a lucky draw that later offspring cannot beat on merit, insertion rates collapse
towards zero, and the archived value is not an estimate of the elite's ability —
it is an estimate of its best luck. Report replayed numbers, and prefer medians
over many replicas to any single rollout.

    uv run python -m qd.check_repeatability --genomes logs/qd/seeds/ppo_seeds.npz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC


@dataclass
class Args:
    genomes: Path | None = None
    """``.npz`` from :mod:`qd.seed` (key ``genome``), or an archive."""

    rank: int = 0
    """Which genome, if the file holds several."""

    replicas: int = 256
    """Identical copies evaluated in one batch."""

    divergence_probe: bool = True
    """Also report when the trajectories start to differ, step by step."""

    device: str = "cuda:0"
    out: Path | None = None
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    spec = DEFAULT_SPEC

    if args.genomes is None:
        raise SystemExit("pass --genomes")
    with np.load(args.genomes) as f:
        key = "genome" if "genome" in f.files else "solution"
        block = f[key]
    if key == "solution":
        block = block[np.argsort(-load_archive(args.genomes)["objective"])]
    genome = torch.as_tensor(
        block.reshape(-1, spec.genome_dim)[args.rank],
        dtype=torch.float32,
        device=args.device,
    )

    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(num_envs=args.replicas, device=args.device),
        args.fitness,
        spec,
    )
    population = genome.reshape(1, -1).expand(args.replicas, -1).contiguous()

    spread_log: list[dict] = []
    origin: dict[str, torch.Tensor] = {}
    if args.divergence_probe:
        def recorder(phase, step, _alive):
            if phase != "episode":
                return
            # Worlds are laid out at different env origins, so the raw world-x
            # is not comparable across them. Compare each world's *travel* from
            # its own position at the first scored step.
            x = harness.base_pos()[:, 0]
            if "x0" not in origin:
                origin["x0"] = x.clone()
            travel = x - origin["x0"]
            spread_log.append(
                {
                    "step": step,
                    "max_abs_delta_m": float((travel - travel[0]).abs().max()),
                }
            )
    else:
        recorder = None

    _, measures, info, _ = harness.rollout(
        population, collect=False, recorder=recorder
    )
    displacement = info["displacement"]
    survived = ~info["fell"]

    payload = {
        "genomes": str(args.genomes),
        "rank": args.rank,
        "replicas": args.replicas,
        "survival_rate": float(survived.mean()),
        "displacement_mean_m": float(displacement.mean()),
        "displacement_median_m": float(np.median(displacement)),
        "displacement_std_m": float(displacement.std()),
        "displacement_min_m": float(displacement.min()),
        "displacement_max_m": float(displacement.max()),
        "displacement_p5_m": float(np.percentile(displacement, 5)),
        "displacement_p95_m": float(np.percentile(displacement, 95)),
        # What MAP-Elites would have archived (the best single sample) against
        # what the genome is actually worth (the median).
        "map_elites_optimism_m": float(displacement.max() - np.median(displacement)),
        "duty_left_std": float(measures[:, 0].std()),
        "duty_right_std": float(measures[:, 1].std()),
    }
    print(
        f"\n{args.replicas} identical copies of one genome, every DR knob off:\n"
        f"  survival        {payload['survival_rate'] * 100:.1f}%\n"
        f"  displacement    median {payload['displacement_median_m']:+.3f} m, "
        f"sd {payload['displacement_std_m']:.3f} m, "
        f"5-95% {payload['displacement_p5_m']:+.3f} .. {payload['displacement_p95_m']:+.3f} m\n"
        f"  full range      {payload['displacement_min_m']:+.3f} .. "
        f"{payload['displacement_max_m']:+.3f} m\n"
        f"  duty factor sd  ({payload['duty_left_std']:.3f}, {payload['duty_right_std']:.3f})\n"
        f"  what MAP-Elites would archive, minus what the genome is worth: "
        f"{payload['map_elites_optimism_m']:+.3f} m",
        flush=True,
    )

    if spread_log:
        payload["divergence"] = spread_log
        marks = [0.001, 0.01, 0.1]
        for mark in marks:
            hit = next((r for r in spread_log if r["max_abs_delta_m"] > mark), None)
            when = f"step {hit['step']} ({hit['step'] * 0.02:.2f} s)" if hit else "never"
            print(f"  trunk-x spread first exceeds {mark * 1000:5.0f} mm at {when}")

    if args.out is not None:
        write_json(args.out, payload)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
