"""Throughput and evaluation-noise benchmark — run it before choosing a batch size.

Two numbers decide how a MAP-Elites run should be configured, and both are
machine-specific:

* **ms per genome.** The rollout loop is 350 control steps of Python, so a
  generation costs about the same wall-clock whether it holds 8 worlds or 2048.
  Bigger batches are nearly free; the benchmark says where that stops being
  true on this GPU.
* **fitness noise.** MuJoCo-Warp is *not* bit-reproducible across runs — the
  batched contact/constraint solve is order-sensitive — so the same genome
  scores slightly differently each time. MAP-Elites keeps the luckiest sample
  per cell, so knowing this spread is knowing how optimistic the archive is.

    uv run python -m qd.bench --archive logs/qd/map_elites/archive_final.npz
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd import cpg_genome
from qd.common import FitnessCfg, load_archive
from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness


@dataclass
class Args:
    archive: Path | None = None
    """Archive to take the reference genome from; random if omitted."""

    batch_sizes: tuple[int, ...] = (64, 256, 1024, 2048)
    repeats: int = 2
    """Re-evaluations of the reference genome per batch size."""

    device: str = "cuda:0"
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    space = cpg_genome.genome_space()

    if args.archive is not None:
        data = load_archive(args.archive)
        reference = data["solution"][int(np.argmax(data["objective"]))]
        print(f"reference = best elite of {args.archive} "
              f"(archived fitness {data['objective'].max():+.6f} m)")
    else:
        reference = space.sample(1, np.random.default_rng(0))[0]
        print("reference = a random genome")

    for num_envs in args.batch_sizes:
        harness = MicroduckRolloutHarness(
            HarnessCfg(num_envs=num_envs, device=args.device), args.fitness
        )
        evaluator = CpgEvaluator(harness)

        scores = [float(evaluator.replay(reference)[0]) for _ in range(args.repeats)]
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        evaluator.evaluate(space.sample(num_envs, rng))
        dt = time.perf_counter() - t0

        spread = max(scores) - min(scores)
        print(
            f"num_envs={num_envs:5d} | {dt:6.2f} s/generation, "
            f"{dt / num_envs * 1000:6.1f} ms/genome | reference fitness "
            f"{' '.join(f'{s:+.5f}' for s in scores)} (spread {spread:.1e} m)"
        )
        harness.close()


if __name__ == "__main__":
    main()
