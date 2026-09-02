"""The two ways PGA-MAP-Elites makes offspring.

**GA variation — iso+lineDD** (Vassiliades & Mouret, "Discovering the elite
hypervolume by leveraging interspecies correlation")::

    child = parent_a + iso_sigma * N(0, I) + line_sigma * N(0, 1) * (parent_b - parent_a)

This replaces the plain per-dimension Gaussian that Phase 2 uses. On a 31-D CPG
genome isotropic noise is fine; on ~9k MLP weights it is hopeless — a fixed
per-dimension sigma either barely moves the policy or destroys it. The *line*
term fixes that by mutating along the direction between two existing elites,
which is a direction the archive has already shown to be productive, and it is
scale-free: the step size adapts to how far apart the parents are.

**PG variation** copies an elite and takes ``num_pg_training_steps`` Adam steps
on it to maximise the TD3 critic's Q — gradient ascent on the same objective
the archive ranks by. All offspring are stepped *simultaneously*, each on its
own independently sampled transition batch, mirroring QDax's vmapped emitter.

Defaults follow QDax: ``iso_sigma`` 0.005, ``line_sigma`` 0.05,
``proportion_mutation_ga`` 0.5, ``num_pg_training_steps`` 100.
"""

from __future__ import annotations

import torch

from qd.pga.policy_genome import DEFAULT_SPEC, PolicySpec
from qd.pga.td3 import Td3Trainer

ISO_SIGMA = 0.005
LINE_SIGMA = 0.05


def isoline_variation(
    parents_a: torch.Tensor,
    parents_b: torch.Tensor,
    generator: torch.Generator,
    iso_sigma: float = ISO_SIGMA,
    line_sigma: float = LINE_SIGMA,
) -> torch.Tensor:
    """``(P, D)`` offspring from two ``(P, D)`` parent batches."""
    if parents_a.shape != parents_b.shape:
        raise ValueError(f"parent shapes differ: {parents_a.shape} vs {parents_b.shape}")
    iso = torch.randn(
        parents_a.shape, device=parents_a.device, generator=generator
    ) * iso_sigma
    line = torch.randn(
        (parents_a.shape[0], 1), device=parents_a.device, generator=generator
    ) * line_sigma
    return parents_a + iso + line * (parents_b - parents_a)


def pg_variation(
    parents: torch.Tensor,
    trainer: Td3Trainer,
    steps: int | None = None,
    spec: PolicySpec = DEFAULT_SPEC,
) -> torch.Tensor:
    """Gradient-ascend ``parents`` on the critic; returns ``(P, D)`` offspring.

    Each parent gets its own Adam state and its own transition sample per step,
    so the P policies are optimised independently even though they are stepped
    in one batched pass.
    """
    steps = trainer.cfg.num_pg_training_steps if steps is None else steps
    pop = parents.shape[0]
    if pop == 0 or len(trainer.buffer) < trainer.cfg.batch_size:
        return parents.detach().clone()

    genomes = parents.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([genomes], lr=trainer.cfg.policy_learning_rate)

    for _ in range(steps):
        batch = trainer.buffer.sample(
            pop, trainer.cfg.batch_size, generator=trainer.generator
        )
        actions = spec.forward(genomes, batch.obs)  # (P, N, A)
        # The critic is a plain MLP over (obs, action); flatten the policy axis
        # into the batch axis so one forward covers every offspring.
        q = trainer.critic.q1_value(
            batch.obs.reshape(-1, spec.obs_dim), actions.reshape(-1, spec.action_dim)
        )
        loss = -q.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return genomes.detach()


def sample_parents(
    archive,
    count: int,
    generator: torch.Generator,
    device: str,
    spec: PolicySpec = DEFAULT_SPEC,
) -> torch.Tensor:
    """``count`` elite genomes drawn uniformly with replacement, as a tensor.

    An **empty** archive returns fresh random MLPs instead of raising. Under
    walking-v2's survival gate an archive can legitimately be empty for a
    while — nothing has stayed upright yet — and the search should keep
    sampling the space rather than crash on iteration 1.
    """
    solutions = archive.data("solution")
    if len(solutions) == 0:
        return spec.initial_population(count, generator, device)
    idx = torch.randint(
        len(solutions), (count,), device=device, generator=generator
    )
    return torch.as_tensor(
        solutions[idx.cpu().numpy()], dtype=torch.float32, device=device
    )
