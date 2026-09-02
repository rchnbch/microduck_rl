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
from qd.descriptors import DescriptorCfg

# Midpoint of microduck_constants' delay_min_lag=3 / delay_max_lag=6.
FIXED_COMMAND_LAG: int = 4
# Nominal pack voltage; microduck_constants randomizes over (6.5, 8.2).
NOMINAL_VIN: float = 7.4
# Spawn height of the trunk. HOME stands with the lowest foot vertex at
# trunk_z - 0.1172, and the velocity task spawns in (0.12, 0.13); a hair of
# clearance lets the settle phase drop the robot onto the plane.
SPAWN_HEIGHT: float = 0.125

FEET_CONTACT_SENSOR = "feet_ground_contact"
GROUND_CONTACT_SENSOR = "qd_ground_contact"
"""Per-geom robot/terrain contact — what the v4 mode classifier reads.

The feet sensor answers "are the soles down", which is the whole story for a
walker and none of it for a crawl. This one covers *every* ground-capable geom
on the robot, so ``f_feet`` / ``f_body`` / ``f_air`` are a partition rather
than three guesses."""

HEAD_CONTACT_BODY = "jaw_soft"
"""The body carrying the three head-shell collision meshes.

Head involvement is what separates an over-the-head roll from a shoulder roll
— the roulade env needed a head-top latch for exactly that distinction."""


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

    full_gait_stats: bool = False
    """Read the extra velocity / joint / actuator channels every step.

    See :class:`qd.pga.evaluate.PolicyHarnessCfg.full_gait_stats`; off unless
    the descriptor needs them or a measurement wants every candidate axis."""

    mode_channels: bool = False
    """Build the per-geom ground-contact sensor and accumulate P2' features.

    v4's gate reads contact *classes*, not foot contact, so it needs a sensor
    the v1-v3 harness never built. Off by default: a v3 command line must keep
    producing byte-for-byte v3 physics, and an extra sensor changes the
    compiled model."""

    shell_friction: float | None = None
    """Override the shell ``mu`` this rollout runs at; ``None`` keeps the repo's.

    The nominal is a literature value for PLA, not a hardware measurement
    (:data:`mjlab_microduck.robot.microduck_constants.SHELL_FRICTION`), so the
    sensitivity of a crawl to it is a number worth having —
    ``qd.check_shell_contacts --sweep-friction`` uses this."""

    njmax: int = 128
    """Constraints allocated per world.

    mjlab's heuristic is sized for the foot-only model and overflows once every
    shell can touch the ground: a robot lying on its side under
    ``full_collision`` produced ``nefc overflow - please increase njmax to 91``
    from ``qd.check_harness``, and an overflow silently *drops* constraints —
    i.e. quietly makes the floor soft again, which is the exact bug
    ``full_collision`` exists to fix. 128 clears the worst observed frame with
    headroom. (The Phase-3 harness runs through ``ManagerBasedRlEnv``, which
    sizes this itself and never overflowed; this is the low-level path.)"""

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
    if harness_cfg.shell_friction is not None:
        cfg = dataclasses.replace(
            cfg, collisions=(_shell_friction_collision(harness_cfg.shell_friction),)
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


def _shell_friction_collision(mu: float):
    """``FULL_COLLISION`` with the shell ``mu`` replaced, feet untouched."""
    import dataclasses as _dc

    from mjlab_microduck.robot.microduck_constants import FULL_COLLISION

    return _dc.replace(
        FULL_COLLISION,
        friction={
            r"^(left|right)_foot_collision$": (1.0,),
            r".*_collision": (float(mu),),
        },
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


@dataclass(frozen=True)
class ContactGeoms:
    """Which ground-capable geoms exist, and which of them are feet or head."""

    names: tuple[str, ...]
    feet: tuple[str, ...]
    head: tuple[str, ...]


def ground_contact_geoms(full_collision: bool = True) -> ContactGeoms:
    """Every geom that can touch the terrain, split into feet / head / rest.

    Read off the spec the robot cfg actually builds rather than hardcoded: the
    contact shell is completed and named at spec-build time
    (:mod:`mjlab_microduck.robot.shell_contacts`), so a re-export from Onshape
    changes this list and nothing else has to be edited."""
    from mjlab_microduck.robot.microduck_constants import (
        get_standup_spec,
        get_walk_spec,
    )
    from mjlab_microduck.robot.shell_contacts import (
        FOOT_GEOM_NAMES,
        collision_geoms_on_bodies,
        ground_collision_geom_names,
    )

    spec = get_standup_spec() if full_collision else get_walk_spec()
    return ContactGeoms(
        names=ground_collision_geom_names(spec),
        feet=FOOT_GEOM_NAMES,
        head=collision_geoms_on_bodies(spec, (HEAD_CONTACT_BODY,)),
    )


def _ground_contact_sensor_cfg(geom_names: tuple[str, ...]):
    """One slot per ground-capable geom, with the net force behind it."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    return ContactSensorCfg(
        name=GROUND_CONTACT_SENSOR,
        primary=ContactMatch(mode="geom", pattern=tuple(geom_names), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )


def resolve_contact_columns(robot, geoms: ContactGeoms):
    """``(names, foot_columns, head_columns)`` in the sensor's own column order.

    The sensor resolves its primaries with ``entity.find_geoms(patterns)``,
    which returns them in *model* order, not in the order they were listed. So
    the column order is asked of the same call rather than assumed — getting
    this wrong would silently swap "the soles are down" for "the head is down"
    in every classifier feature, and every mode label with it.
    """
    _, resolved = robot.find_geoms(list(geoms.names))
    resolved = tuple(resolved)
    feet = tuple(i for i, n in enumerate(resolved) if n in geoms.feet)
    head = tuple(i for i, n in enumerate(resolved) if n in geoms.head)
    if len(feet) != len(geoms.feet):
        raise RuntimeError(
            f"expected {geoms.feet} among the ground-contact geoms, got {resolved}"
        )
    if not head:
        raise RuntimeError(f"no head-shell geom among {resolved}")
    return resolved, feet, head


class MicroduckRolloutHarness:
    """Persistent batched simulation of ``robot_walk.xml`` on a flat plane.

    Built once and reused for every generation: compiling the scene and
    capturing the CUDA graphs costs seconds, and the graph pins ``num_envs``,
    so the batch size cannot change afterwards.
    """

    def __init__(
        self,
        cfg: HarnessCfg,
        fitness: FitnessCfg | None = None,
        descriptor: DescriptorCfg | None = None,
    ):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.terrains.terrain_entity import TerrainEntityCfg

        self.cfg = cfg
        self.fitness = fitness or FitnessCfg()

        sensors = [_feet_contact_sensor_cfg()]
        self.contact_geoms = None
        if cfg.mode_channels:
            self.contact_geoms = ground_contact_geoms(cfg.full_collision)
            sensors.append(_ground_contact_sensor_cfg(self.contact_geoms.names))

        scene_cfg = SceneCfg(
            num_envs=cfg.num_envs,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": _deterministic_robot_cfg(cfg)},
            sensors=tuple(sensors),
        )
        self.scene = Scene(scene_cfg, device=cfg.device)
        self.sim = Simulation(
            num_envs=cfg.num_envs,
            cfg=SimulationCfg(
                mujoco=MujocoCfg(timestep=cfg.physics_dt), njmax=cfg.njmax
            ),
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

        self.descriptor = descriptor or DescriptorCfg()
        self._collect_extras = bool(cfg.full_gait_stats or self.descriptor.needs)

        self.robot = self.scene.entities["robot"]
        self.contact_columns = None
        if self.contact_geoms is not None:
            self.contact_columns = resolve_contact_columns(
                self.robot, self.contact_geoms
            )
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

        # All 14 servos, in the repo's canonical order. The CPG path pins the
        # neck at HOME; the v4 probes do not — a log-roll is driven by hip roll
        # and head yaw in quadrature, and the roulade env needed a chin tuck to
        # get over the head at all (design draft, 5.3).
        self.servo_joint_ids, self.servo_joint_names = self.robot.find_joints(
            [r"^(?!passive_).*"]
        )
        self._servo_ids_t = torch.as_tensor(self.servo_joint_ids, device=device)
        self._home_servo_targets = self._home_joint_pos[:, self._servo_ids_t].clone()
        lo, hi = cpg_genome.soft_joint_limits(self.servo_joint_names)
        self._servo_lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self._servo_hi = torch.as_tensor(hi, dtype=torch.float32, device=device)

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

    @property
    def home_servo_targets(self) -> torch.Tensor:
        """``(N, 14)`` HOME angles of every servo, in ``servo_joint_names`` order."""
        return self._home_servo_targets

    # -- simulation ------------------------------------------------------- #

    def reset(self, pose=None) -> None:
        """Return every world to a spawn pose at rest; HOME standing by default.

        ``pose`` is a :class:`qd.spawn.SpawnPose`. The archive always spawns
        standing (§4.1: displacement is measured from the same start for every
        mode, transition included), so this argument exists for the Stage A'
        *probes*: no open-loop posture survives a standing start on this robot —
        a 0.4 s ramp to SIT topples at 0.36 s and 0 of 42 slow squats held to
        3 s — so a scripted crawl has to be measured from prone, and a scripted
        log-roll from its side.
        """
        self.sim.reset(None)
        self.scene.reset(None)

        root_state = self.robot.data.default_root_state.clone()
        joint_pos = self._home_joint_pos.clone()
        if pose is None:
            root_state[:, 2] = self.cfg.spawn_height
        else:
            root_state[:, 2] = pose.height
            quat = torch.as_tensor(
                pose.quat_wxyz, dtype=root_state.dtype, device=root_state.device
            )
            root_state[:, 3:7] = quat
            for pattern, angle in (pose.joint_pos or {}).items():
                ids, _ = self.robot.find_joints([pattern])
                joint_pos[:, torch.as_tensor(ids, device=joint_pos.device)] = angle
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        self.robot.set_joint_position_target(joint_pos)
        self._spawn_joint_pos = joint_pos
        self.scene.write_data_to_sim()
        self.sim.forward()

    @property
    def spawn_servo_targets(self) -> torch.Tensor:
        """``(N, 14)`` servo angles of the pose the last :meth:`reset` wrote."""
        pos = getattr(self, "_spawn_joint_pos", self._home_joint_pos)
        return pos[:, self._servo_ids_t]

    def set_leg_targets(self, targets: torch.Tensor) -> None:
        """Command the 10 leg joints; neck/head stay at their HOME target."""
        self.robot.set_joint_position_target(
            targets.clamp(self._soft_lo, self._soft_hi), joint_ids=self._leg_ids_t
        )

    def set_servo_targets(self, targets: torch.Tensor) -> None:
        """Command all 14 servos, in ``servo_joint_names`` order."""
        self.robot.set_joint_position_target(
            targets.clamp(self._servo_lo, self._servo_hi), joint_ids=self._servo_ids_t
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

    def gait_extras(self) -> dict[str, torch.Tensor] | None:
        """Extra per-step channels for the candidate descriptor axes.

        Mirrors :meth:`qd.pga.evaluate.PolicyRolloutHarness.gait_extras`, so a
        CPG archive and an MLP archive can be binned on the same axes."""
        if not self._collect_extras:
            return None
        d = self.robot.data
        return {
            "lin_vel_w": d.root_link_lin_vel_w,
            "ang_vel_b": d.root_link_ang_vel_b,
            "joint_vel": d.joint_vel,
            "qfrc_actuator": d.qfrc_actuator,
        }

    def mode_channels(self) -> dict[str, torch.Tensor] | None:
        """Per-geom ground contact + the rates :class:`qd.modes.ModeStats` folds.

        ``None`` when the harness was not built with ``mode_channels``: a v3
        run must not pay for a sensor it does not read."""
        if self.contact_columns is None:
            return None
        sensor = self.scene.sensors[GROUND_CONTACT_SENSOR].data
        found = sensor.found
        assert found is not None
        n_geoms = len(self.contact_columns[0])
        d = self.robot.data
        out = {
            "contact_found": found.reshape(self.num_envs, -1)[:, :n_geoms],
            "ang_vel_w": d.root_link_ang_vel_w,
            "lin_vel_w": d.root_link_lin_vel_w,
        }
        if sensor.force is not None:
            out["contact_force"] = sensor.force.reshape(self.num_envs, -1, 3)[
                :, :n_geoms
            ]
        return out

    def make_mode_stats(self, windows):
        """A :class:`qd.modes.ModeStats` wired to this harness's column order."""
        from qd.modes import ModeStats

        if self.contact_columns is None:
            raise RuntimeError(
                "harness was built without mode_channels; P2' has nothing to read"
            )
        names, feet, head = self.contact_columns
        return ModeStats(
            self.num_envs,
            self.device,
            windows,
            num_contact_geoms=len(names),
            foot_columns=feet,
            head_columns=head,
        )

    # -- generic rollout -------------------------------------------------- #

    def rollout(
        self, controller, recorder=None, mode_stats=None
    ) -> tuple[np.ndarray, np.ndarray, dict]:
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

        metrics = RolloutMetrics(
            self.num_envs,
            fit,
            self.device,
            episode_steps,
            descriptor=self.descriptor,
            control_dt=self.control_dt,
        )
        metrics.begin(self.base_pos())
        accel = None
        if mode_stats is not None:
            from qd.modes import VerticalAccel

            mode_stats.begin(self.base_pos())
            accel = VerticalAccel(self.num_envs, self.device, self.control_dt)
            accel.begin(self.robot.data.root_link_lin_vel_w)
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for k in range(episode_steps):
            self.set_leg_targets(controller(k, k * self.control_dt))
            self.step()
            was_alive = alive.clone()
            metrics.update(
                self.base_pos(),
                self.projected_gravity(),
                self.foot_contact(),
                self.gait_extras(),
            )
            if mode_stats is not None:
                ch = self.mode_channels()
                assert ch is not None
                mode_stats.update(
                    self.base_pos(),
                    self.projected_gravity(),
                    ch["contact_found"],
                    ch["ang_vel_w"],
                    trunk_az=accel.step(ch["lin_vel_w"]),
                    contact_force=ch.get("contact_force"),
                )
            alive = ~metrics.fallen
            if recorder is not None:
                recorder("episode", k, was_alive)
            # Under P2' nothing "falls", so this early-out never fires; it is
            # left in place so a v3 command line keeps its behaviour exactly.
            if (
                self.cfg.fall_check_every
                and (k + 1) % self.cfg.fall_check_every == 0
                and not bool(alive.any())
            ):
                break
        fitness, measures, info = metrics.finalize()
        if mode_stats is not None:
            info.update(mode_stats.finalize().to_info())
        return fitness, measures, info

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

    def evaluate_with_modes(self, block: np.ndarray, mode_stats):
        """One chunk, accumulating P2' features alongside fitness.

        Used by Stage A' to run v1's CPG elites through the v4 gate. Chunk-
        sized only (no splitting): a ``ModeStats`` is bound to a world count,
        and silently reusing one across chunks would mix two rollouts."""
        h = self.harness
        block = self.space.clip(np.atleast_2d(np.asarray(block, dtype=np.float64)))
        if block.shape[0] != h.num_envs:
            raise ValueError(
                f"{block.shape[0]} genomes for {h.num_envs} worlds; "
                "evaluate_with_modes takes exactly one chunk"
            )
        genomes_t = torch.as_tensor(block, dtype=torch.float32, device=h.device)
        steps = round(h.fitness.episode_seconds / h.control_dt)
        times = torch.arange(steps, dtype=torch.float32, device=h.device) * h.control_dt
        traj = cpg_target_trajectory(genomes_t, times)
        _f, _m, info = h.rollout(lambda k, _t: traj[k], mode_stats=mode_stats)
        from qd.modes import ModeFeatures

        return ModeFeatures.from_info(info)

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
