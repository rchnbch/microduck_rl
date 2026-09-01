"""Batched policy rollout on the Velocity task env, with transition collection.

Phase 2 talks to mjlab's low-level ``Scene``/``Simulation`` because an open-loop
CPG needs no observations. Phase 3 needs the repo's **61-D observation
contract**, and hand-rolling that pipeline is exactly the mistake AGENTS.md
warns about, so this builds the real ``Mjlab-Velocity-Flat-MicroDuck`` env cfg
and strips it down instead:

* every domain-randomization event removed (the ``ENABLE_*`` toggles are module
  constants, so they are neutralised by dropping the event terms they install —
  ``expand_bam_friction_fields`` is deliberately **kept**, it is not DR but the
  per-world model-field expansion BAM cannot run without);
* observation corruption off, so a genome has one fitness;
* the spawn pinned (no ±0.5 m offset, no random yaw — "+x displacement" is only
  meaningful from a fixed heading);
* all command ranges zeroed, so the 13-D command block feeds the policy zeros —
  the deployment idle state, and the same conditions Phase 2 ran under;
* terminations and reward terms removed: the rollout runs a fixed number of
  steps and is scored by :class:`qd.common.RolloutMetrics`, identically to
  Phase 2, so the two archives are directly comparable.

The robot itself is swapped for the same deterministic-actuator config Phase 2
uses, which is what makes a QD-score comparison between the two archives mean
something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qd.common import FitnessCfg, RolloutMetrics
from qd.evaluate import FEET_CONTACT_SENSOR, _deterministic_robot_cfg
from qd.evaluate import HarnessCfg as _ActuatorCfg
from qd.pga.policy_genome import DEFAULT_SPEC, PolicySpec

# Event terms the velocity cfg installs purely for domain randomization. Named
# explicitly rather than pattern-matched: a silent no-op here would leave DR on
# and make every fitness noisy, which is far worse than a KeyError on rename.
DR_EVENT_TERMS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_com",
    "randomize_head_com",
    "randomize_mass_inertia",
    "randomize_joint_friction",
    "randomize_armature",
)
# Kept: BAM needs its model fields expanded per world (see AGENTS.md).
KEEP_EVENT_TERMS = ("reset_base", "reset_robot_joints", "reset_action_history",
                    "expand_bam_friction_fields")


@dataclass(frozen=True)
class PolicyHarnessCfg:
    num_envs: int = 128
    device: str = "cuda:0"
    spawn_height: float = 0.125
    """Matches Phase 2; the velocity task itself spawns in (0.12, 0.13)."""


@dataclass
class Transitions:
    """One generation's worth of ``(obs, action, reward, next_obs, done)``.

    Flat over (policy, step), with post-fall steps already dropped — a robot
    that is down contributes no transitions, so the critic never learns from
    the frozen tail.
    """

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def _stripped_velocity_env_cfg(cfg_num_envs: int, spawn_height: float):
    """The velocity env cfg with DR, commands, rewards and terminations removed."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(play=False, rough=False)
    cfg.scene.num_envs = cfg_num_envs

    # Same deterministic BAM actuator as Phase 2, so the two archives are
    # measured under identical physics.
    cfg.scene.entities["robot"] = _deterministic_robot_cfg(_ActuatorCfg())

    for name in DR_EVENT_TERMS:
        cfg.events.pop(name, None)
    unexpected = set(cfg.events) - set(KEEP_EVENT_TERMS)
    if unexpected:
        raise RuntimeError(
            f"unrecognised event terms left after stripping DR: {sorted(unexpected)}. "
            "The velocity cfg grew a term this module has not classified — "
            "classify it in DR_EVENT_TERMS or KEEP_EVENT_TERMS before running."
        )

    # Deterministic spawn: no positional jitter, no random yaw.
    cfg.events["reset_base"].params["pose_range"] = {
        "z": (spawn_height, spawn_height)
    }
    cfg.events["reset_base"].params["velocity_range"] = {}

    # Curricula mutate command/event ranges over time; nothing here wants that.
    cfg.curriculum = {}

    # Commands pinned to zero. The obs slots stay (the 61-D layout is an
    # invariant); they just always read zero, which is the deployment idle
    # state and the condition Phase 2's open-loop gaits ran under.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.ranges.heading = (0.0, 0.0)
    twist.rel_standing_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.rel_heading_envs = 0.0
    for name in ("head_pose", "body_pose"):
        pose = cfg.commands[name]
        pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)

    # Observation noise off: a genome must have one fitness.
    for group in cfg.observations.values():
        group.enable_corruption = False

    # We score the rollout ourselves and never want a mid-rollout reset.
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.episode_length_s = 1e6

    return cfg


class PolicyRolloutHarness:
    """Runs a population of MLP genomes, one per world, on the stripped env."""

    def __init__(
        self,
        cfg: PolicyHarnessCfg,
        fitness: FitnessCfg | None = None,
        spec: PolicySpec = DEFAULT_SPEC,
    ):
        from mjlab.envs import ManagerBasedRlEnv

        self.cfg = cfg
        self.fitness = fitness or FitnessCfg()
        self.spec = spec
        env_cfg = _stripped_velocity_env_cfg(cfg.num_envs, cfg.spawn_height)
        self.env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
        self.robot = self.env.scene["robot"]

        obs, _ = self.env.reset()
        actor_obs = obs["actor"]
        if actor_obs.shape[-1] != spec.obs_dim:
            raise RuntimeError(
                f"actor observation is {actor_obs.shape[-1]}-D, expected "
                f"{spec.obs_dim}. The 61-D contract is a hard invariant "
                "(AGENTS.md) — a policy trained here would not be hot-swappable."
            )

    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs

    @property
    def device(self) -> str:
        return self.cfg.device

    @property
    def control_dt(self) -> float:
        return float(self.env.step_dt)

    # -- readout, mirroring qd.evaluate.MicroduckRolloutHarness -------------- #

    def base_pos(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w

    def projected_gravity(self) -> torch.Tensor:
        return self.robot.data.projected_gravity_b

    def forward_velocity(self) -> torch.Tensor:
        """World-frame +x velocity of the trunk [m/s]."""
        return self.robot.data.root_link_lin_vel_w[:, 0]

    def foot_contact(self) -> torch.Tensor:
        found = self.env.scene.sensors[FEET_CONTACT_SENSOR].data.found
        assert found is not None
        return found.reshape(self.num_envs, -1)[:, :2] > 0

    # -- rollout ------------------------------------------------------------- #

    def rollout(
        self, genomes: torch.Tensor, collect: bool = True, recorder=None
    ) -> tuple[np.ndarray, np.ndarray, dict, Transitions | None]:
        """Evaluate one genome per world and optionally collect transitions.

        The per-step reward is the exact decomposition of the episodic fitness
        that Phase 2 uses::

            r_t        = forward_velocity * dt          while upright
            r_terminal = -fall_penalty * (steps_left / total_steps)   on the fall

        so the undiscounted return equals
        ``displacement - fall_penalty * fraction_of_episode_fallen`` — the
        objective the archive actually ranks on. A critic trained on anything
        else would push PG variation somewhere the archive does not reward.
        """
        if genomes.shape[0] != self.num_envs:
            raise ValueError(
                f"got {genomes.shape[0]} genomes for {self.num_envs} worlds; "
                "the population size must equal the world count"
            )
        fit = self.fitness
        settle_steps = round(fit.settle_seconds / self.control_dt)
        episode_steps = round(fit.episode_seconds / self.control_dt)

        obs_dict, _ = self.env.reset()
        obs = obs_dict["actor"]
        zero_action = torch.zeros(
            self.num_envs, self.spec.action_dim, device=self.device
        )
        for k in range(settle_steps):
            obs_dict = self.env.step(zero_action)[0]
            obs = obs_dict["actor"]
            if recorder is not None:
                recorder("settle", k)

        metrics = RolloutMetrics(self.num_envs, fit, self.device)
        metrics.begin(self.base_pos())

        # Collected as dense (T, N, ...) stacks and masked ONCE at the end.
        # Masking per step (`obs[was_alive]`) would make the output size depend
        # on GPU data, forcing a host sync on every tensor every step — ~1750
        # syncs per rollout, which dominated the wall clock.
        buf: list[tuple[torch.Tensor, ...]] = []
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for step in range(episode_steps):
            with torch.no_grad():
                action = self.spec.forward(genomes, obs)
            obs_dict = self.env.step(action)[0]
            next_obs = obs_dict["actor"]

            was_alive = alive.clone()
            metrics.update(
                self.base_pos(), self.projected_gravity(), self.foot_contact()
            )
            alive = ~metrics.fallen
            just_fell = was_alive & ~alive

            if collect:
                reward = torch.nan_to_num(self.forward_velocity()) * self.control_dt
                steps_left = episode_steps - step - 1
                reward = reward - just_fell.float() * (
                    fit.fall_penalty * steps_left / episode_steps
                )
                buf.append((obs, action, reward, next_obs, just_fell.float(), was_alive))
            obs = next_obs
            if recorder is not None:
                recorder("episode", step)

        fitness, measures, info = metrics.finalize()
        transitions = None
        if collect and buf:
            stacks = [torch.stack(t) for t in zip(*buf)]  # each (T, N, ...)
            keep = stacks[-1].reshape(-1)  # was_alive, flattened over (T, N)
            obs_s, act_s, rew_s, next_s, done_s = stacks[:5]
            transitions = Transitions(
                obs=obs_s.reshape(-1, self.spec.obs_dim)[keep],
                action=act_s.reshape(-1, self.spec.action_dim)[keep],
                reward=rew_s.reshape(-1)[keep],
                next_obs=next_s.reshape(-1, self.spec.obs_dim)[keep],
                done=done_s.reshape(-1)[keep],
            )
        return fitness, measures, info, transitions
