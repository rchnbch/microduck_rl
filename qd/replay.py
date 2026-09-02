"""Re-evaluate archived genomes — the honest-number path shared by the reports.

MAP-Elites stores the *luckiest* sample per cell, and MuJoCo-Warp's batched
contact solve is not bit-reproducible, so archived fitness is biased upward.
The size of that bias is a property of the genome class rather than a constant:
measured over the top 64 of each archive, an open-loop CPG comes back within
+0.003 m of its archived value while a closed-loop MLP is +0.166 m optimistic,
because the closed-loop policy amplifies a tiny contact difference through its
feedback loop and the CPG replays the same joint trajectory regardless.

Consequence: **two archives from different genome classes cannot be compared on
archived fitness.** Both :mod:`qd.survival_report` and :mod:`qd.compare_archives`
go through this module so the numbers they print are re-measured, not restored.
"""

from __future__ import annotations

import numpy as np

from qd.common import FitnessCfg


def infer_kind(solutions: np.ndarray) -> str:
    """``'cpg'`` or ``'mlp'`` from the solution width (31 vs 9038)."""
    return "cpg" if solutions.shape[1] < 100 else "mlp"


def reevaluate(
    solutions: np.ndarray,
    kind: str,
    fitness: FitnessCfg,
    device: str = "cuda:0",
    max_envs: int = 512,
    full_collision: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """Roll ``solutions`` out again.

    Returns ``(fitness, measures, info, control_dt)``. The batch is chunked to
    ``max_envs``; each chunk builds its own harness, because the CUDA graph pins
    the world count.

    ``full_collision`` defaults to walking-v2's honest physics. Re-evaluating a
    v1 archive with it on is a *fair* thing to do and an informative one — it
    answers "was that survivor real, or was it standing on a floor it could
    sink into" — but it is not the physics that archive was searched under, so
    label such numbers as a re-measurement rather than as v1's result.
    """
    n = len(solutions)
    parts: list[tuple[np.ndarray, np.ndarray, dict]] = []
    control_dt = 0.02

    for start in range(0, n, max_envs):
        block = solutions[start : start + max_envs]
        if kind == "cpg":
            from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness

            harness = MicroduckRolloutHarness(
                HarnessCfg(
                    num_envs=len(block),
                    device=device,
                    full_collision=full_collision,
                ),
                fitness,
            )
            f, m, info = CpgEvaluator(harness).evaluate(block)
            control_dt = harness.control_dt
        else:
            import torch

            from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness

            harness = PolicyRolloutHarness(
                PolicyHarnessCfg(
                    num_envs=len(block),
                    device=device,
                    full_collision=full_collision,
                ),
                fitness,
            )
            genomes = torch.as_tensor(block, dtype=torch.float32, device=device)
            f, m, info, _ = harness.rollout(genomes, collect=False)
            control_dt = harness.control_dt
        parts.append((f, m, info))

    fitness_out = np.concatenate([p[0] for p in parts])
    measures_out = np.concatenate([p[1] for p in parts])
    info_out = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    return fitness_out, measures_out, info_out, control_dt
