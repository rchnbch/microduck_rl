"""TD3 critic + greedy actor, the gradient half of PGA-MAP-Elites.

The critic is what makes PG variation possible: it is trained off-policy on
*every* transition any offspring has ever produced, so it accumulates value
knowledge across the whole archive and can then be differentiated through to
improve an individual elite. The greedy actor is trained alongside it and is
inserted into the archive each generation like any other candidate.

Hyperparameters follow QDax's ``pga_me`` / ``td3`` defaults (twin critics,
target networks, target-policy smoothing with ``policy_noise`` 0.2 clipped at
0.5, ``policy_delay`` 2, Polyak ``soft_tau_update`` 0.005, discount 0.99,
transition batch 256, critic and greedy-actor LR 3e-4).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from qd.pga.evaluate import Transitions
from qd.pga.policy_genome import DEFAULT_SPEC, PolicySpec


@dataclass(frozen=True)
class Td3Cfg:
    """QDax ``pga_me`` defaults unless noted."""

    replay_buffer_size: int = 1_000_000
    batch_size: int = 256
    discount: float = 0.99
    reward_scaling: float = 1.0
    soft_tau_update: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    critic_hidden: tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4

    # --- policy-side step sizes: NOT QDax's, and the reason is measured ----- #
    # QDax's 100 offspring steps at 1e-3 and 300 greedy steps at 3e-4 are tuned
    # for a search that starts from random policies, where a large move costs
    # nothing because there is nothing yet to break. Walking-v2 starts from a
    # distilled PPO walker and only inserts full-episode survivors, and Adam's
    # per-parameter step is bounded by the learning rate, so `steps * lr` is a
    # budget on how far the genome travels — against an initial weight scale of
    # ~1/sqrt(61) = 0.13. `qd.pga.tune_pg` measures where that budget stops
    # being survivable (40 offspring per cell, one trained critic, parents
    # feasible 40% of the time):
    #
    #   steps  lr      |dgenome|   offspring feasible
    #       5  3e-5    1.1e-4      48%
    #      30  3e-5    6.3e-4      55%
    #      10  3e-4    2.1e-3      40%
    #      30  1e-4    2.1e-3      52%
    #      30  3e-4    5.8e-3       0%   <- QDax-scale move, total collapse
    #     100  1e-3    2.9e-2       0%   <- QDax's actual default
    #
    # Feasibility holds to ~2e-3 of mean per-weight drift and collapses by
    # ~6e-3. The defaults below sit at 6.3e-4, comfortably inside the safe
    # region and still moving the genome ~5x further than the GA's iso term.
    greedy_learning_rate: float = 1e-6
    """The greedy actor takes ~150 updates *per iteration* (``num_critic_
    training_steps / policy_delay``), so its budget is 150x an offspring's and
    it needs a correspondingly smaller rate. Measured over 8 validation
    iterations, as ``greedy_survival_fraction`` per iteration:

        3e-4 (QDax)   1.00 0.06 ...           destroyed inside one iteration
        1e-5          1.00 1.00 0.20 0.16 ... gone by iteration 3
        1e-6          1.00 1.00 1.00 1.00 ... upright throughout

    This matters well beyond the greedy actor's own insertions, because TD3
    bootstraps its target off ``greedy_target``: a fallen greedy actor teaches
    a critic whose Q surface is about falling, and PG variation then ascends
    every elite toward it. Offspring feasibility rose from ~45% to ~58% on the
    move from 1e-5 to 1e-6."""

    policy_learning_rate: float = 3e-5
    """Learning rate for PG variation of *offspring*."""

    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 30


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions, resident on the GPU."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.action = torch.zeros(capacity, action_dim, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, device=device)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def add(self, t: Transitions) -> None:
        n = len(t)
        if n == 0:
            return
        if n > self.capacity:  # keep the most recent slice
            t = Transitions(*(x[-self.capacity :] for x in
                              (t.obs, t.action, t.reward, t.next_obs, t.done)))
            n = self.capacity
        idx = (torch.arange(n, device=self.device) + self._cursor) % self.capacity
        self.obs[idx] = t.obs
        self.action[idx] = t.action
        self.reward[idx] = t.reward
        self.next_obs[idx] = t.next_obs
        self.done[idx] = t.done
        self._cursor = int((self._cursor + n) % self.capacity)
        self._size = int(min(self._size + n, self.capacity))

    def sample(self, *shape: int, generator: torch.Generator) -> Transitions:
        """Sample transitions with an arbitrary leading shape.

        ``sample(256)`` gives a flat batch for critic training; ``sample(50, 256)``
        gives each of 50 offspring its own independent batch, which is what
        QDax's vmapped PG variation does.
        """
        if self._size == 0:
            raise RuntimeError("replay buffer is empty")
        idx = torch.randint(
            self._size, shape, device=self.device, generator=generator
        )
        return Transitions(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
        )


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TwinCritic(nn.Module):
    """Two independent Q networks; TD3 takes the min to fight overestimation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, hidden, 1)
        self.q2 = _mlp(obs_dim + action_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1)).squeeze(-1)


class Td3Trainer:
    """Owns the buffer, the twin critics and the continuously trained greedy actor.

    The greedy actor is stored as a flat genome (``(1, genome_dim)``) rather
    than an ``nn.Module`` so that it is the *same object type* as every archive
    elite: it can be inserted, mutated and evaluated by the identical code path.
    """

    def __init__(
        self,
        cfg: Td3Cfg,
        device: str,
        seed: int = 0,
        spec: PolicySpec = DEFAULT_SPEC,
    ):
        self.cfg = cfg
        self.device = device
        self.spec = spec
        self.generator = torch.Generator(device=device).manual_seed(seed)

        self.buffer = ReplayBuffer(
            cfg.replay_buffer_size, spec.obs_dim, spec.action_dim, device
        )
        self.critic = TwinCritic(spec.obs_dim, spec.action_dim, cfg.critic_hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_learning_rate
        )

        self.greedy = spec.initial_population(1, self.generator, device).requires_grad_(True)
        self.greedy_target = self.greedy.detach().clone()
        self.greedy_opt = torch.optim.Adam([self.greedy], lr=cfg.greedy_learning_rate)
        self._updates = 0

    def set_greedy(self, genome: torch.Tensor) -> None:
        """Start the greedy actor from ``genome`` instead of a random init.

        Walking-v2 hands it the distilled PPO walker, and that is not a
        cosmetic head start. The TD3 target is
        ``r + gamma * Q_target(s', greedy_target(s'))``: the critic learns the
        value of *the greedy actor's* behaviour. Bootstrapping off a policy
        that face-plants at 1.3 s teaches a critic whose Q surface is about
        falling, and PG variation then gradient-ascends every elite toward
        that — measured in the v2 validation run as a PG feasibility rate of
        0.0% against GA's 40%. Starting the greedy actor on a walker points
        the whole gradient half of the algorithm at states a survivor can
        actually reach.
        """
        with torch.no_grad():
            self.greedy.copy_(genome.reshape(1, -1).to(self.greedy.device))
        self.greedy_target = self.greedy.detach().clone()
        # Adam's moments were accumulated for the old parameters; keeping them
        # would apply the previous actor's momentum to this one.
        self.greedy_opt = torch.optim.Adam(
            [self.greedy], lr=self.cfg.greedy_learning_rate
        )

    # -- training ------------------------------------------------------------ #

    def train(self, steps: int | None = None) -> dict[str, float]:
        """Run TD3 updates on the buffer; returns the last-step losses."""
        steps = self.cfg.num_critic_training_steps if steps is None else steps
        cfg = self.cfg
        critic_loss = actor_loss = float("nan")
        if len(self.buffer) < cfg.batch_size:
            return {"critic_loss": critic_loss, "greedy_q": actor_loss}

        for _ in range(steps):
            batch = self.buffer.sample(cfg.batch_size, generator=self.generator)

            with torch.no_grad():
                next_action = self.spec.forward(
                    self.greedy_target, batch.next_obs.unsqueeze(0)
                ).squeeze(0)
                noise = (
                    torch.randn(
                        next_action.shape, device=self.device, generator=self.generator
                    )
                    * cfg.policy_noise
                ).clamp(-cfg.noise_clip, cfg.noise_clip)
                # tanh output => the action space is [-1, 1]; smoothing noise
                # must be clipped back into it or the target is off-manifold.
                next_action = (next_action + noise).clamp(-1.0, 1.0)
                q1_t, q2_t = self.critic_target(batch.next_obs, next_action)
                target = batch.reward * cfg.reward_scaling + cfg.discount * (
                    1.0 - batch.done
                ) * torch.min(q1_t, q2_t)

            q1, q2 = self.critic(batch.obs, batch.action)
            loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_opt.step()
            critic_loss = float(loss.detach())

            self._updates += 1
            if self._updates % cfg.policy_delay == 0:
                action = self.spec.forward(self.greedy, batch.obs.unsqueeze(0)).squeeze(0)
                q = self.critic.q1_value(batch.obs, action).mean()
                self.greedy_opt.zero_grad(set_to_none=True)
                (-q).backward()
                self.greedy_opt.step()
                actor_loss = float(q.detach())
                self._polyak()

        return {"critic_loss": critic_loss, "greedy_q": actor_loss}

    def _polyak(self) -> None:
        tau = self.cfg.soft_tau_update
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
            self.greedy_target.mul_(1 - tau).add_(self.greedy.detach(), alpha=tau)

    # -- accessors ----------------------------------------------------------- #

    def greedy_genome(self) -> torch.Tensor:
        """``(1, genome_dim)`` copy of the greedy actor, safe to evaluate."""
        return self.greedy.detach().clone()
