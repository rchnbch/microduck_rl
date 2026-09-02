"""Batched mjlab rollout harness: genome batch -> (fitness, behaviour descriptor).

One genome per parallel world; every world advances on the same clock, so a
whole MAP-Elites generation is a single batched rollout.  The harness talks to
mjlab's low-level ``Scene`` / ``Simulation`` directly rather than to the
``Mjlab-Velocity-*`` task: the CPG needs neither velocity commands, nor the
61-D observation stack, nor rewards — only physics, base state and foot
contacts.  (Phase 3 *does* want the observation stack and builds on the task
env instead; both share :mod:`qd.common` for fitness/descriptor/archive.)

Domain randomization is off by design.  The BAM actuator keeps its physics
(voltage control, load-dependent friction) but its per-env DR knobs — battery
voltage range, voltage-sag gain, randomized command lag — are pinned to fixed
values, so a genome's fitness is reproducible and the archive is noise-free.
The one thing kept from the trained-policy setup is a *fixed* command lag
(:data:`FIXED_COMMAND_LAG` steps, the midpoint of the range PPO trains under),
because the lag is real hardware latency rather than randomization.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from qd import cpg_genome
from qd.common import FitnessCfg, RolloutMetrics

# Midpoint of microduck_constants' delay_min_lag=3 / delay_max_lag=6.
FIXED_COMMAND_LAG: int = 4
# Nominal pack voltage; microduck_constants randomizes over (6.5, 8.2).
NOMINAL_VIN: float = 7.4
# Spawn height of the trunk. HOME stands with the lowest foot vertex at
# trunk_z - 0.1172, and the velocity task spawns in (0.12, 0.13); a hair of
# clearance lets the settle phase drop the robot onto the plane.
SPAWN_HEIGHT: float = 0.125

FEET_CONTACT_SENSOR = "feet_ground_contact"


@dataclass(frozen=True)
class HarnessCfg:
    """Simulation-side configuration of the rollout harness."""

    num_envs: int = 128
    """Worlds stepped in parallel. Genome batches are chunked to this size."""

    device: str = "cuda:0"
    physics_dt: float = 0.005
    decimation: int = 4
    """physics_dt * decimation = 20 ms -> the repo's 50 Hz control loop."""

    spawn_height: float = SPAWN_HEIGHT
    command_lag: int = FIXED_COMMAND_LAG
    vin: float = NOMINAL_VIN

    full_collision: bool = True
    """Walking-v2 default: ``robot_allcollisions.xml`` instead of ``robot_walk.xml``.

    ``robot_walk.xml`` gives ground-collision geoms to the two foot soles only —
    trunk, head, hips and thighs are ``contype=0`` or self-collision-only. That
    is sound for PPO, which *terminates* the episode at the fall, and dishonest
    for a QD rollout, which keeps simulating: a toppled robot sinks through the
    floor and the rendered clip shows a duck buried in the plane. The
    all-collisions model gives every shell a ground contact, so a fall lands on
    the floor like a fall. Phase 2/v1 ran with this ``False``; see the v1-vs-v2
    section of the README for the measured cost."""

    fall_check_every: int = 25
    """Control steps between "has every world fallen yet?" checks (0.5 s).

    The rollout stops when nothing is upright any more. Not checked every step:
    ``alive.any()`` forces a host sync."""

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


def _deterministic_robot_cfg(harness_cfg: HarnessCfg):
    """The repo's robot cfg with every actuator DR knob pinned.

    Built with :func:`dataclasses.replace` on the repo's own actuator config so
    the BAM motor model, firmware gain and friction stay in sync with training
    automatically; only the randomization ranges are overridden.

    ``full_collision`` picks the model: ``MICRODUCK_STANDUP_ROBOT_CFG`` is the
    same robot on ``robot_allcollisions.xml`` (identical HOME frame, identical
    actuators, identical ``FULL_COLLISION`` cfg) — only the MJCF differs.
    """
    from mjlab_microduck.robot.microduck_constants import (
        MICRODUCK_STANDUP_ROBOT_CFG,
        MICRODUCK_WALK_ROBOT_CFG,
    )

    cfg = (
        MICRODUCK_STANDUP_ROBOT_CFG
        if harness_cfg.full_collision
        else MICRODUCK_WALK_ROBOT_CFG
    )
    assert cfg.articulation is not None
    actuators = tuple(
        dataclasses.replace(
            act,
            vin=harness_cfg.vin,
            vin_range=None,
            vin_drop_gain_range=None,
            delay_min_lag=harness_cfg.command_lag,
            delay_max_lag=harness_cfg.command_lag,
        )
        for act in cfg.articulation.actuators
    )
    return dataclasses.replace(
        cfg,
        articulation=dataclasses.replace(cfg.articulation, actuators=actuators),
    )


def _feet_contact_sensor_cfg():
    """Left-then-right foot/ground contact, matching the velocity task's sensor."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    return ContactSensorCfg(
        name=FEET_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="geom",
            # LEFT foot first, RIGHT foot second — the descriptor's axis order.
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )


class MicroduckRolloutHarness:
    """Persistent batched simulation of ``robot_walk.xml`` on a flat plane.

    Built once and reused for every generation: compiling the scene and
    capturing the CUDA graphs costs seconds, and the graph pins ``num_envs``,
    so the batch size cannot change afterwards.
    """

    def __init__(self, cfg: HarnessCfg, fitness: FitnessCfg | None = None):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.terrains.terrain_entity import TerrainEntityCfg

        self.cfg = cfg
        self.fitness = fitness or FitnessCfg()

        scene_cfg = SceneCfg(
            num_envs=cfg.num_envs,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": _deterministic_robot_cfg(cfg)},
            sensors=(_feet_contact_sensor_cfg(),),
        )
        self.scene = Scene(scene_cfg, device=cfg.device)
        self.sim = Simulation(
            num_envs=cfg.num_envs,
            cfg=SimulationCfg(mujoco=MujocoCfg(timestep=cfg.physics_dt)),
            model=self.scene.compile(),
            device=cfg.device,
        )
        self.scene.initialize(
            mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data
        )
        # The BAM actuator writes a per-env friction budget into these model
        # fields every step; without expanding them per world the write lands
        # in shared memory (see AGENTS.md, "Actuators are BAM").
        self.sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        self.robot = self.scene.entities["robot"]
        self.leg_joint_ids, leg_names = self.robot.find_joints(
            list(cpg_genome.LEG_JOINT_NAMES), preserve_order=True
        )
        assert tuple(leg_names) == cpg_genome.LEG_JOINT_NAMES, leg_names

        device = cfg.device
        self._leg_ids_t = torch.as_tensor(self.leg_joint_ids, device=device)
        self._home_joint_pos = self.robot.data.default_joint_pos.clone()
        self._home_leg_targets = self._home_joint_pos[:, self._leg_ids_t].clone()
        soft_lo, soft_hi = cpg_genome.soft_leg_joint_limits()
        self._soft_lo = torch.as_tensor(soft_lo, dtype=torch.float32, device=device)
        self._soft_hi = torch.as_tensor(soft_hi, dtype=torch.float32, device=device)

    # -- properties ------------------------------------------------------- #

    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs

    @property
    def device(self) -> str:
        return self.cfg.device

    @property
    def control_dt(self) -> float:
        return self.cfg.control_dt

    @property
    def home_leg_targets(self) -> torch.Tensor:
        """``(N, 10)`` HOME angles of the leg joints, in genome order."""
        return self._home_leg_targets

    # -- simulation ------------------------------------------------------- #

    def reset(self) -> None:
        """Return every world to the HOME pose at the spawn height, at rest."""
        self.sim.reset(None)
        self.scene.reset(None)

        root_state = self.robot.data.default_root_state.clone()
        root_state[:, 2] = self.cfg.spawn_height
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(
            self._home_joint_pos, torch.zeros_like(self._home_joint_pos)
        )
        self.robot.set_joint_position_target(self._home_joint_pos)
        self.scene.write_data_to_sim()
        self.sim.forward()

    def set_leg_targets(self, targets: torch.Tensor) -> None:
        """Command the 10 leg joints; neck/head stay at their HOME target."""
        self.robot.set_joint_position_target(
            targets.clamp(self._soft_lo, self._soft_hi), joint_ids=self._leg_ids_t
        )

    def step(self) -> None:
        """One 50 Hz control step = ``decimation`` physics substeps.

        No ``sim.forward()`` afterwards: ``mj_step`` runs forward kinematics
        *before* integration, so ``xpos``/``xquat`` lag ``qpos`` by one substep
        (5 ms). mjlab's own ``ManagerBasedRlEnv.step`` accepts exactly this
        staleness for rewards and terminations; paying for another forward pass
        every control step to remove 5 ms of lag from a displacement metric
        would cost ~25% throughput for nothing.
        """
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.cfg.physics_dt)

    # -- readout ---------------------------------------------------------- #

    def base_pos(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w

    def projected_gravity(self) -> torch.Tensor:
        return self.robot.data.projected_gravity_b

    def foot_contact(self) -> torch.Tensor:
        """``(N, 2)`` boolean left/right foot-ground contact."""
        found = self.scene.sensors[FEET_CONTACT_SENSOR].data.found
        assert found is not None
        return found.reshape(self.num_envs, -1)[:, :2] > 0

    # -- generic rollout -------------------------------------------------- #

    def rollout(self, controller, recorder=None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Settle at HOME, then run ``controller`` and score the result.

        ``controller(step, t)`` returns ``(num_envs, 10)`` leg targets for
        control step ``step`` at time ``t`` seconds since the CPG started.
        ``recorder(phase, step, alive)``, if given, is called after every
        control step with ``phase`` in ``{"settle", "episode"}`` and ``alive``
        the ``(N,)`` mask of worlds still upright *on that step* (``None``
        during the settle) — used by ``qd/play_elite.py`` and
        ``qd/render_gaits.py`` to log qpos, and to cut each clip at its fall.

        The loop stops once no world is upright: post-fall physics reaches no
        metric, so there is nothing left to simulate.
        """
        fit = self.fitness
        settle_steps = round(fit.settle_seconds / self.control_dt)
        episode_steps = round(fit.episode_seconds / self.control_dt)

        self.reset()
        for k in range(settle_steps):
            self.set_leg_targets(self.home_leg_targets)
            self.step()
            if recorder is not None:
                recorder("settle", k, None)

        metrics = RolloutMetrics(self.num_envs, fit, self.device, episode_steps)
        metrics.begin(self.base_pos())
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(episode_steps):
            self.set_leg_targets(controller(k, k * self.control_dt))
            self.step()
            was_alive = alive.clone()
            metrics.update(
                self.base_pos(), self.projected_gravity(), self.foot_contact()
            )
            alive = ~metrics.fallen
            if recorder is not None:
                recorder("episode", k, was_alive)
            if (
                self.cfg.fall_check_every
                and (k + 1) % self.cfg.fall_check_every == 0
                and not bool(alive.any())
            ):
                break
        return metrics.finalize()

    def close(self) -> None:
        self.scene = None  # type: ignore[assignment]
        self.sim = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# CPG evaluation
# --------------------------------------------------------------------------- #


def cpg_target_trajectory(
    genomes: torch.Tensor, times: torch.Tensor
) -> torch.Tensor:
    """``(T, B, 10)`` leg targets — the GPU twin of :func:`qd.cpg_genome.cpg_targets`.

    Precomputing the whole trajectory keeps the rollout loop free of host->device
    copies; ``tests/test_qd_cpg_genome.py`` pins it to the numpy reference.
    """
    freq = genomes[:, cpg_genome.FREQ_SLICE]  # (B, 1)
    amp = genomes[:, cpg_genome.AMP_SLICE]  # (B, 10)
    phase = genomes[:, cpg_genome.PHASE_SLICE]
    offset = genomes[:, cpg_genome.OFFSET_SLICE]
    t = times.reshape(-1, 1, 1)  # (T, 1, 1)
    angle = 2.0 * math.pi * freq.unsqueeze(0) * t + phase.unsqueeze(0)
    return offset.unsqueeze(0) + amp.unsqueeze(0) * torch.sin(angle)


class CpgEvaluator:
    """Maps a batch of CPG genomes to ``(fitness, measures, info)``.

    Batches larger than ``harness.num_envs`` are chunked; smaller ones are
    padded (the CUDA graph fixes the world count) and the padding sliced off.
    """

    def __init__(self, harness: MicroduckRolloutHarness):
        self.harness = harness
        self.space = cpg_genome.genome_space()

    def evaluate(self, genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        genomes = np.atleast_2d(np.asarray(genomes, dtype=np.float64))
        if genomes.shape[1] != cpg_genome.GENOME_DIM:
            raise ValueError(
                f"expected (B, {cpg_genome.GENOME_DIM}) genomes, got {genomes.shape}"
            )
        # Defensive bound enforcement #2 (the emitter's `bounds=` is #1).
        genomes = self.space.clip(genomes)

        n = genomes.shape[0]
        chunk = self.harness.num_envs
        fits, meas, infos = [], [], []
        for start in range(0, n, chunk):
            block = genomes[start : start + chunk]
            pad = chunk - block.shape[0]
            if pad:
                block = np.concatenate([block, np.repeat(block[:1], pad, axis=0)])
            f, m, info = self._evaluate_chunk(block)
            keep = chunk - pad
            fits.append(f[:keep])
            meas.append(m[:keep])
            infos.append({k: v[:keep] for k, v in info.items()})

        info = {k: np.concatenate([d[k] for d in infos]) for k in infos[0]}
        return np.concatenate(fits), np.concatenate(meas), info

    def _evaluate_chunk(self, block: np.ndarray, recorder=None):
        h = self.harness
        device = h.device
        genomes_t = torch.as_tensor(block, dtype=torch.float32, device=device)
        steps = round(h.fitness.episode_seconds / h.control_dt)
        times = torch.arange(steps, dtype=torch.float32, device=device) * h.control_dt
        traj = cpg_target_trajectory(genomes_t, times)  # (T, B, 10)
        return h.rollout(lambda k, _t: traj[k], recorder=recorder)

    def replay(self, genome: np.ndarray, recorder=None):
        """Evaluate a single genome, broadcast across every world.

        Used by ``qd/play_elite.py``: all worlds run the same deterministic
        genome, so world 0's trajectory is the elite's gait.
        """
        block = np.repeat(
            self.space.clip(np.atleast_2d(genome)), self.harness.num_envs, axis=0
        )
        f, m, info = self._evaluate_chunk(block, recorder=recorder)
        return f[0], m[0], {k: v[0] for k, v in info.items()}
