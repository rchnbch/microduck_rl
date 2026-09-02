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
from qd.descriptors import DescriptorCfg
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

    full_collision: bool = True
    """Walking-v2: every shell collides with the ground, not just the soles.

    See :class:`qd.evaluate.HarnessCfg.full_collision`. Passed straight through
    to the shared deterministic robot cfg, so both pipelines change together."""

    fall_check_every: int = 25
    """Control steps between "has every world fallen yet?" checks (0.5 s).

    The rollout stops as soon as no world is upright: post-fall physics must
    not reach fitness, descriptor, replay buffer or recorded ``qpos``, and once
    everything is down there is nothing left worth simulating. The check needs
    a host sync, so it is not run every step — the comment on the transition
    buffer below explains what per-step syncing costs here."""

    full_gait_stats: bool = False
    """Read velocity / joint / actuator channels every step, whether or not the
    archive's descriptor needs them.

    Off during a run — the harness supplies exactly the channels the chosen
    descriptor asks for, and nothing else. On for
    :mod:`qd.select_descriptor`, which measures *every* candidate axis from one
    rollout per genome and therefore needs the lot. These are device-side reads
    with no host sync, so the cost is a few extra kernels per step."""


@dataclass(frozen=True)
class ShapedRewardCfg:
    """Per-step reward for the TD3 critic — *not* the archive's objective.

    v1 made these two the same thing: ``r_t = v_x*dt`` with a pro-rata terminal
    penalty summed exactly to ``displacement - fall_penalty * fallen_fraction``,
    the value the archive ranked on. That identity was the right call while
    fitness was the only thing standing between the search and a face-plant.

    Walking-v2 makes survival a **feasibility constraint** instead (only
    full-episode survivors are ever inserted), so the archive no longer needs
    the objective to encode "don't fall" — and the critic's job changes with
    it. What PG variation now needs from the critic is a dense signal for
    *balance*, which bare forward velocity does not contain: v1's critic could
    not distinguish a controlled step from the first frame of a dive. So the
    per-step reward gains the two terms the PPO velocity recipe uses to teach
    exactly that, and the exact-decomposition identity is deliberately dropped.

    Weights are in metres, so the velocity term is still literally displacement
    and the critic keeps v1's scale. The ratio is borrowed from the velocity
    task's reward stack (``track_linear_velocity`` weight 2.0 against
    ``upright`` weight 2.0, i.e. 1:1): at the 0.3 m/s the twist command tops out
    around, ``upright_weight`` pays the same per second as travel does.
    """

    vel_weight: float = 1.0
    """Multiplier on ``v_x * dt``. 1.0 keeps the term equal to displacement."""

    alive_bonus: float = 0.10
    """Paid per second upright, regardless of posture [m/s]."""

    upright_weight: float = 0.30
    """Paid per second, scaled by ``clip(-projected_gravity_z, 0, 1)`` [m/s].

    Uses the same quantity the velocity task's ``upright`` term does, and the
    same quantity the fall check reads, so the critic is taught balance in the
    coordinate the feasibility gate actually measures (AGENTS.md: a tracking
    reward must measure the same view the policy is judged on)."""

    fall_penalty: float = 1.0
    """Charged once, pro-rata over the unfinished part of the episode [m].

    4x v1's 0.25. It no longer has to stay small to avoid distorting an
    archive objective — inside the gated archive the terminal penalty is
    unreachable, since a fallen solution is simply not inserted — so it is now
    sized to dominate the ~0.3-2 m of travel a diving policy could buy."""


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


def _stripped_velocity_env_cfg(
    cfg_num_envs: int, spawn_height: float, full_collision: bool = True
):
    """The velocity env cfg with DR, commands, rewards and terminations removed."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg(play=False, rough=False)
    cfg.scene.num_envs = cfg_num_envs

    # Same deterministic BAM actuator as Phase 2, so the two archives are
    # measured under identical physics; `full_collision` also swaps the MJCF
    # for the all-collisions model (walking-v2's honest-physics default).
    cfg.scene.entities["robot"] = _deterministic_robot_cfg(
        _ActuatorCfg(full_collision=full_collision)
    )

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
        reward: ShapedRewardCfg | None = None,
        descriptor: DescriptorCfg | None = None,
    ):
        from mjlab.envs import ManagerBasedRlEnv

        self.cfg = cfg
        self.fitness = fitness or FitnessCfg()
        self.reward_cfg = reward or ShapedRewardCfg()
        self.descriptor = descriptor or DescriptorCfg()
        self._collect_extras = bool(
            cfg.full_gait_stats or self.descriptor.needs
        )
        self.spec = spec
        env_cfg = _stripped_velocity_env_cfg(
            cfg.num_envs, cfg.spawn_height, cfg.full_collision
        )
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

    def gait_extras(self) -> dict[str, torch.Tensor] | None:
        """The extra per-step channels the candidate descriptor axes read.

        ``None`` when the archive's descriptor needs none of them, which keeps
        a duty-factor run byte-for-byte the work walking-v2 did. Every field is
        a device-side view; nothing here syncs to the host."""
        if not self._collect_extras:
            return None
        d = self.robot.data
        return {
            "lin_vel_w": d.root_link_lin_vel_w,
            "ang_vel_b": d.root_link_ang_vel_b,
            "joint_vel": d.joint_vel,
            "qfrc_actuator": d.qfrc_actuator,
        }

    # -- rollout ------------------------------------------------------------- #

    def rollout(
        self,
        genomes: torch.Tensor | None = None,
        collect: bool = True,
        recorder=None,
        actor=None,
        on_step=None,
    ) -> tuple[np.ndarray, np.ndarray, dict, Transitions | None]:
        """Evaluate one genome per world and optionally collect transitions.

        The per-step reward is :class:`ShapedRewardCfg`::

            r_t        = vel_weight * v_x * dt
                       + (alive_bonus + upright_weight * upright) * dt   while upright
            r_terminal = -fall_penalty * (steps_left / total_steps)      on the fall

        where ``upright = clip(-projected_gravity_z, 0, 1)``. Read that class
        for why walking-v2 abandons v1's exact-decomposition identity between
        this reward and the archive objective.

        The rollout **stops at the fall**: an env that is down stops
        contributing to fitness, descriptor, transitions and the recorder, and
        the loop breaks entirely once no env is left upright.

        Args:
            genomes: ``(num_envs, genome_dim)``, one policy per world. Optional
                only when ``actor`` is given.
            actor: ``obs -> action`` override, used by :mod:`qd.seed` to drive
                the same harness with a *teacher* policy that is not a genome.
            on_step: ``(step, obs, action, alive) -> None``, called before each
                env step with the mask of worlds still upright — the collection
                hook for behaviour cloning.
        """
        if actor is None:
            if genomes is None:
                raise ValueError("rollout needs either `genomes` or an `actor`")
            if genomes.shape[0] != self.num_envs:
                raise ValueError(
                    f"got {genomes.shape[0]} genomes for {self.num_envs} worlds; "
                    "the population size must equal the world count"
                )
            actor = lambda o: self.spec.forward(genomes, o)
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
                recorder("settle", k, None)

        metrics = RolloutMetrics(
            self.num_envs,
            fit,
            self.device,
            episode_steps,
            descriptor=self.descriptor,
            control_dt=self.control_dt,
        )
        metrics.begin(self.base_pos())

        # Collected as dense (T, N, ...) stacks and masked ONCE at the end.
        # Masking per step (`obs[was_alive]`) would make the output size depend
        # on GPU data, forcing a host sync on every tensor every step — ~1750
        # syncs per rollout, which dominated the wall clock.
        buf: list[tuple[torch.Tensor, ...]] = []
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        rc = self.reward_cfg
        for step in range(episode_steps):
            with torch.no_grad():
                action = actor(obs)
            if on_step is not None:
                on_step(step, obs, action, alive)
            obs_dict = self.env.step(action)[0]
            next_obs = obs_dict["actor"]

            was_alive = alive.clone()
            gravity = self.projected_gravity()
            metrics.update(
                self.base_pos(), gravity, self.foot_contact(), self.gait_extras()
            )
            alive = ~metrics.fallen
            just_fell = was_alive & ~alive

            if collect:
                upright = torch.nan_to_num(-gravity[:, 2]).clamp(0.0, 1.0)
                reward = (
                    rc.vel_weight * torch.nan_to_num(self.forward_velocity())
                    + rc.alive_bonus
                    + rc.upright_weight * upright
                ) * self.control_dt
                steps_left = episode_steps - step - 1
                reward = reward - just_fell.float() * (
                    rc.fall_penalty * steps_left / episode_steps
                )
                buf.append((obs, action, reward, next_obs, just_fell.float(), was_alive))
            obs = next_obs
            if recorder is not None:
                # `was_alive`, not `alive`: the step on which the fall is
                # detected is the last honest frame, and it is the one the
                # metrics count too.
                recorder("episode", step, was_alive)

            # Nothing upright is left to simulate, and post-fall physics must
            # not reach any metric. Checked periodically: the `.any()` forces a
            # host sync, which is exactly what the comment above says not to do
            # every step.
            if (
                self.cfg.fall_check_every
                and (step + 1) % self.cfg.fall_check_every == 0
                and not bool(alive.any())
            ):
                break

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
