"""The Phase-3 genome: a closed-loop MLP flattened into one weight vector.

``61 -> 64 -> 64 -> 14``, tanh everywhere including the output.

* **61 in** is the repo's shared observation contract (48 proprioception + the
  13-D command block), so an evolved policy is drop-in compatible with the
  existing ONNX export and runtime — the whole point of not inventing a new
  observation layout here.
* **14 out** is the servo action, which mjlab's ``JointPositionActionCfg``
  turns into ``HOME + action`` (scale 1.0).
* **tanh on the output** bounds the action to ±1 rad around HOME. That is a
  sane range for a 25 cm robot (the widest joint is ±1.57 rad) and it is also
  what makes TD3 correct: target-policy smoothing assumes a bounded action
  space to clip the exploration noise against.

Everything is *batched over policies*: a population of ``P`` genomes is one
``(P, 9102)`` tensor and one forward pass, because MAP-Elites evaluates a whole
generation at once and PG variation trains ~50 offspring simultaneously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

OBS_DIM = 61
ACTION_DIM = 14
HIDDEN = (64, 64)


@dataclass(frozen=True)
class PolicySpec:
    """Layer shapes and the flat-vector layout derived from them."""

    obs_dim: int = OBS_DIM
    action_dim: int = ACTION_DIM
    hidden: tuple[int, ...] = HIDDEN

    @property
    def layer_shapes(self) -> list[tuple[int, int]]:
        """``[(out, in), ...]`` for each Linear layer."""
        dims = [self.obs_dim, *self.hidden, self.action_dim]
        return [(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]

    @property
    def genome_dim(self) -> int:
        return sum(o * i + o for o, i in self.layer_shapes)

    def unflatten(self, genomes: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(P, genome_dim)`` -> per-layer ``(W (P, out, in), b (P, out))``.

        Views into ``genomes``, not copies, so autograd flows straight back to
        the flat vector — which is what lets PG variation optimise the genome
        itself rather than a reconstructed module.
        """
        if genomes.shape[-1] != self.genome_dim:
            raise ValueError(
                f"expected genomes of width {self.genome_dim}, got {genomes.shape[-1]}"
            )
        params, offset = [], 0
        pop = genomes.shape[0]
        for out_dim, in_dim in self.layer_shapes:
            n_w = out_dim * in_dim
            weight = genomes[:, offset : offset + n_w].reshape(pop, out_dim, in_dim)
            offset += n_w
            bias = genomes[:, offset : offset + out_dim]
            offset += out_dim
            params.append((weight, bias))
        return params

    def forward(self, genomes: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Per-policy forward pass.

        Args:
            genomes: ``(P, genome_dim)``.
            obs: ``(P, obs_dim)`` — one observation per policy (rollout), or
                ``(P, N, obs_dim)`` — a batch of ``N`` transitions per policy
                (PG variation).

        Returns:
            ``(P, action_dim)`` or ``(P, N, action_dim)`` to match ``obs``.
        """
        squeeze = obs.dim() == 2
        x = obs.unsqueeze(1) if squeeze else obs  # (P, N, obs)
        for weight, bias in self.unflatten(genomes):
            x = torch.tanh(torch.einsum("poi,pni->pno", weight, x) + bias.unsqueeze(1))
        return x.squeeze(1) if squeeze else x

    def initial_population(
        self, pop: int, generator: torch.Generator, device: str | torch.device
    ) -> torch.Tensor:
        """``(pop, genome_dim)`` of independently He/LeCun-initialised MLPs.

        Uses PyTorch's ``nn.Linear`` convention (uniform ``±1/sqrt(fan_in)``)
        per layer rather than one global scale: a single Gaussian over 9k
        parameters gives layers wildly wrong scales and a population that all
        saturate tanh on the first step.
        """
        parts = []
        for out_dim, in_dim in self.layer_shapes:
            bound = 1.0 / math.sqrt(in_dim)
            w = torch.empty(pop, out_dim * in_dim, device=device)
            b = torch.empty(pop, out_dim, device=device)
            w.uniform_(-bound, bound, generator=generator)
            b.uniform_(-bound, bound, generator=generator)
            parts += [w, b]
        return torch.cat(parts, dim=1)


DEFAULT_SPEC = PolicySpec()
