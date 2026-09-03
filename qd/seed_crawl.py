"""Distil v1's rescued CPG crawls into the MLP genome the v4 archive searches.

The design's Q3 routes crawl seeding through a new prone-locomotion PPO task,
on the reasoning that a scripted probe cannot be an archive candidate: open-loop
control cannot survive a standing start on this robot, so a probe has to spawn
prone and the archive spawns standing.

Stage A' found that reasoning has an exception nobody looked for. Five elites
in **v1's MAP-Elites CPG archive** — filed as junk since j002 because the
upright gate called them fallen — spawn standing at HOME, get themselves down,
and crawl. Verified over 128 world-permuted replicas:

| genome | viable | label | median dx | replica sd |
| --- | --- | --- | --- | --- |
| ``v1_cpg_175`` | 1.000 | crawl | +0.543 m | 0.0018 m |
| ``v1_cpg_277`` | 1.000 | crawl | +0.359 m | 0.0057 m |
| ``v1_cpg_169`` | 0.859 | crawl | +0.306 m | 0.0159 m |
| ``v1_cpg_4``   | 0.594 | crawl | +0.447 m | 0.0193 m |
| ``v1_cpg_6``   | 0.461 | crawl | +0.320 m | 0.0131 m |

So the crawl seed exists already and costs a distillation rather than a
training run.

**What has to be bridged.** The teacher is a 31-parameter open-loop CPG driving
ten leg joints in absolute radians on the low-level harness; the student is the
9038-parameter closed-loop MLP that the archive mutates, acting through the
velocity env's ``JointPositionAction`` (``target = HOME + action``, scale 1.0)
over all fourteen servos. Three consequences, all handled here:

* the teacher is **a function of time, not of observation**, so it carries its
  own step counter and must be reset per rollout — everything else in
  :mod:`qd.seed` assumes ``actor(obs)`` is stateless;
* actions are **deltas from HOME**, and the neck/head four are pinned at HOME
  by the CPG, so their labels are exactly zero;
* the genome's ``tanh`` bounds it to +-1 rad, and a CPG offset can sit further
  out than that. The label is clipped and the clipped fraction is *reported*,
  because a teacher the student architecturally cannot reach is a distillation
  that will quietly underperform its teacher.

DAgger applies unchanged and matters more here than for the walker: the crawl's
observation history is nothing like the walker's, and a student trained only on
teacher states would meet its own for the first time at deployment.

    uv run python -m qd.seed_crawl \\
        --archive qd-run-archives/j003/qd/v1/map_elites_final.npz \\
        --indices 175 277 169 4 6 --out logs/qd/seeds/crawl_seeds.npz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from torch import nn

from qd.common import FitnessCfg, load_archive, write_json
from qd.modes import MODES, ClassifierCfg, ViabilityCfg, evaluate_viability
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC, PolicySpec
from qd.seed import SeedCfg


class CpgTeacher:
    """An open-loop CPG genome, presented as the state-conditioned expert DAgger needs.

    **The naive version of this does not work, and the reason is worth keeping.**
    A CPG is a *clock*: its action is a function of time, not of state. Handing
    DAgger a clock breaks its central assumption — the expert must be able to
    answer "given where you are, what would you do", and a clock can only answer
    "at this time, do this". The moment the student drifts out of phase, every
    label is wrong in a way more rounds cannot fix, because the labels
    themselves are mislabelled. Measured with the clock version, over 4 rounds:

    | | teacher states | student states |
    | --- | --- | --- |
    | BC loss | 0.0003 | **0.030** (100x) |
    | ``v1_cpg_175`` displacement | +0.500 m (teacher) | **+0.227 m** |
    | P2' viability | — | **0.000** |

    So the clock is turned into a state-conditioned expert by **phase
    tracking**: at each step the student's actual leg joint positions are
    matched against the CPG's own target trajectory, and the label is the
    action at the *matched* phase rather than at the wall-clock step. A student
    that lags gets labelled with what the teacher does at the point the student
    has actually reached, which is what DAgger needs and what makes the expert
    a function of state again.

    The search is a **local, forward window** around the previous match rather
    than a global argmin. A CPG target is periodic, so a global match on
    position alone is ambiguous — it can jump half a cycle and label the
    student with the opposite half of the stroke. A window that may slip
    backward one step and advance up to ``max_advance`` keeps the phase
    monotone and cheap.

    Two details that cost a run each to find:

    * **The tracker labels; it never drives.** The robot's joints *lag* the
      commanded target (4-step command lag plus a low-kp servo), so matching a
      live pose against the target trajectory always lands behind the command.
      Used to drive, that lag compounds into a slower gait — measured, the
      teacher itself went from +0.500 m to **-0.133 m**, i.e. the tracker
      turned a crawl into a backwards shuffle. Round 0 therefore drives by the
      clock, which is exactly the open-loop gait the archive verified.
    * **The lag is measured, not guessed.** While the teacher drives round 0,
      the tracker runs passively and records ``step - matched_phase``; its
      median is the pipeline's true lag. Labels on later rounds are then
      ``delta[matched_phase + lag + 1]`` — "what the teacher would command
      next, for a robot whose pose is here" — rather than ``delta[phase + 1]``,
      which would ask the student to chase a target it is already behind.

    Returns **actions**, i.e. deltas from HOME in the env's own joint order,
    because that is what the student emits and what ``JointPositionAction``
    (``target = HOME + action``, scale 1.0) consumes.
    """

    def __init__(
        self,
        genome: np.ndarray,
        harness: PolicyRolloutHarness,
        control_dt: float,
        action_dim: int = 14,
        phase_tracking: bool = True,
        max_advance: int = 4,
        back_slip: int = 1,
    ):
        from qd import cpg_genome
        from qd.evaluate import cpg_target_trajectory

        self.control_dt = control_dt
        self.action_dim = action_dim
        self.phase_tracking = phase_tracking
        self.max_advance = max_advance
        self.back_slip = back_slip
        self.harness = harness
        device = harness.device

        joint_names = list(harness.robot.joint_names)
        leg_ids = [joint_names.index(n) for n in cpg_genome.LEG_JOINT_NAMES]
        self._leg_ids = torch.as_tensor(leg_ids, device=device)

        home = harness.robot.data.default_joint_pos[:1].clone()  # (1, nj)
        self._home_legs = home[:, self._leg_ids]

        steps = round(harness.fitness.episode_seconds / control_dt) + max_advance + 2
        times = torch.arange(steps, dtype=torch.float32, device=device) * control_dt
        g = torch.as_tensor(
            np.atleast_2d(genome), dtype=torch.float32, device=device
        )
        # (T, 1, 10) absolute leg targets -> (T, 10), and its delta from HOME
        self._target = cpg_target_trajectory(g, times)[:, 0, :]
        self._delta = self._target - self._home_legs
        self._step = 0
        self._phase: torch.Tensor | None = None
        self.lag: int = 0
        self._lag_samples: list[float] = []

    def reset(self) -> None:
        self._step = 0
        self._phase = None

    def calibrate_lag(self) -> int:
        """Median ``step - matched_phase`` seen while the teacher drove."""
        if self._lag_samples:
            self.lag = round(float(np.median(self._lag_samples)))
        return self.lag

    def _match_phase(self, n: int) -> torch.Tensor:
        """Per-env phase index, from where the robot's legs actually are."""
        device = self._target.device
        if self._phase is None:
            self._phase = torch.zeros(n, dtype=torch.long, device=device)
            return self._phase
        legs = self.harness.robot.data.joint_pos[:, self._leg_ids]  # (N, 10)
        offsets = torch.arange(
            -self.back_slip, self.max_advance + 1, device=device
        )
        cand = (self._phase.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            0, self._target.shape[0] - 2
        )  # (N, W)
        # (N, W, 10) targets at each candidate phase, against the live pose
        ref = self._target[cand]
        dist = torch.linalg.vector_norm(ref - legs.unsqueeze(1), dim=-1)
        self._phase = cand.gather(1, dist.argmin(dim=1, keepdim=True)).squeeze(1)
        return self._phase

    def _emit(self, obs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(
            obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype
        )
        action[:, self._leg_ids] = delta.to(obs.dtype)
        return action

    def drive(self, obs: torch.Tensor) -> torch.Tensor:
        """The open-loop gait, by the clock — the behaviour the archive verified.

        Also advances the phase tracker passively, so round 0 measures the lag
        the labels on later rounds need."""
        k = min(self._step, self._delta.shape[0] - 1)
        if self.phase_tracking:
            phase = self._match_phase(obs.shape[0])
            self._lag_samples.append(float((k - phase).float().median()))
        self._step += 1
        return self._emit(obs, self._delta[k].expand(obs.shape[0], -1))

    def label(self, obs: torch.Tensor) -> torch.Tensor:
        """What the teacher would command next, for a robot whose pose is here."""
        if not self.phase_tracking:
            k = min(self._step, self._delta.shape[0] - 1)
            self._step += 1
            return self._emit(obs, self._delta[k].expand(obs.shape[0], -1))
        phase = self._match_phase(obs.shape[0])
        idx = (phase + self.lag + 1).clamp(max=self._delta.shape[0] - 1)
        return self._emit(obs, self._delta[idx])

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.drive(obs)


@dataclass
class Args:
    archive: Path
    indices: tuple[int, ...] = (175, 277, 169, 4, 6)
    """v1 CPG elites to distil, best first. Defaults are the Stage A' five."""

    out: Path = Path("logs/qd/seeds/crawl_seeds.npz")
    device: str = "cuda:0"
    num_envs: int = 128
    seeding: SeedCfg = field(default_factory=SeedCfg)
    replicas: int = 32
    """World-permuted replicas used to score each distilled seed."""

    phase_tracking: bool = True
    """Label by the phase the student has REACHED, not by the wall clock.

    Off reproduces the measured failure — see :class:`CpgTeacher`."""

    viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    fitness: FitnessCfg = field(
        default_factory=lambda: FitnessCfg(latch_fall=False)
    )


def distil_cpg(
    teacher: CpgTeacher,
    harness: PolicyRolloutHarness,
    cfg: SeedCfg,
    spec: PolicySpec = DEFAULT_SPEC,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[dict]]:
    """Behaviour-clone one CPG crawl into a genome. Returns ``(genome, log)``."""
    device = harness.device
    generator = generator or torch.Generator(device=device).manual_seed(0)
    student = spec.initial_population(1, generator, device).requires_grad_(True)
    opt = torch.optim.Adam([student], lr=cfg.learning_rate)

    obs_bank: list[torch.Tensor] = []
    label_bank: list[torch.Tensor] = []
    log: list[dict] = []

    for rnd in range(cfg.rounds):
        collected_obs: list[torch.Tensor] = []
        collected_lab: list[torch.Tensor] = []

        # The teacher is a clock, so its label for a state is the action it
        # would take at that STEP — which is why it is queried inside the hook
        # rather than replayed afterwards.
        teacher.reset()
        clock = _Clock(teacher)

        def on_step(_step, obs, _action, _alive, out_o=collected_obs,
                    out_l=collected_lab, c=clock):
            out_o.append(obs)
            out_l.append(c.label(obs))

        if rnd == 0:
            actor = clock.drive
        else:
            frozen = student.detach()
            actor = lambda o, g=frozen: spec.forward(g.expand(o.shape[0], -1), o)

        _f, _m, info, _t = harness.rollout(collect=False, actor=actor, on_step=on_step)
        if rnd == 0:
            lag = teacher.calibrate_lag()
            if verbose:
                print(f"    measured pipeline lag: {lag} control steps", flush=True)

        raw = torch.cat(collected_lab)
        obs_bank.append(torch.cat(collected_obs))
        label_bank.append(raw.clamp(-1.0, 1.0))
        obs_all = torch.cat(obs_bank)
        lab_all = torch.cat(label_bank)

        loss_val = float("nan")
        for _ in range(cfg.epochs_per_round):
            idx = torch.randint(
                obs_all.shape[0], (cfg.batch_size,), device=device, generator=generator
            )
            pred = spec.forward(student, obs_all[idx].unsqueeze(0)).squeeze(0)
            loss = nn.functional.mse_loss(pred, lab_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_val = float(loss.detach())

        row = {
            "round": rnd,
            "driver": "teacher" if rnd == 0 else "student",
            "samples": int(obs_all.shape[0]),
            "bc_loss": loss_val,
            "teacher_action_absmax": float(raw.abs().max()),
            "teacher_action_clipped_fraction": float((raw.abs() > 1.0).float().mean()),
            "rollout_mean_displacement_m": float(info["displacement"].mean()),
        }
        log.append(row)
        if verbose:
            print(
                f"  round {rnd} ({row['driver']:>7} driving) | "
                f"{row['samples']:7d} states | bc loss {loss_val:.5f} | "
                f"clipped {row['teacher_action_clipped_fraction'] * 100:4.1f}% | "
                f"driver mean displ {row['rollout_mean_displacement_m']:+.3f} m",
                flush=True,
            )
    return student.detach().reshape(1, -1), log


class _Clock:
    """Keeps the teacher's step counter honest across the two ways it is used.

    The teacher is queried once per step to *label* the state, and on round 0
    it also *drives*. Calling it twice in one step would advance the clock
    twice and label every state with the next step's action, which is a subtle
    off-by-one that shows up as a distillation lagging its teacher by 20 ms.

    On round 0 the drive and the label are the same action by construction —
    the states being labelled are the ones the teacher itself produced — so the
    driven action is cached and handed back rather than recomputed.
    """

    def __init__(self, teacher: CpgTeacher):
        self.teacher = teacher
        self._cached: torch.Tensor | None = None
        self._driving = False

    def drive(self, obs: torch.Tensor) -> torch.Tensor:
        self._cached = self.teacher.drive(obs)
        self._driving = True
        return self._cached

    def label(self, obs: torch.Tensor) -> torch.Tensor:
        if self._driving and self._cached is not None:
            out, self._cached = self._cached, None
            return out
        return self.teacher.label(obs)


def score(genome: torch.Tensor, harness, args: Args) -> dict:
    """Replay a distilled seed and report what the v4 gate would make of it."""
    block = genome.expand(harness.num_envs, -1).contiguous()
    stats = harness.make_mode_stats(args.viability.windows)
    _f, _m, info, _t = harness.rollout(block, collect=False, mode_stats=stats)
    from qd.modes import ModeFeatures

    feats = ModeFeatures.from_info(info)
    v = evaluate_viability(feats, args.viability)
    labels = feats.episode_labels(args.classifier)
    modal = int(np.bincount(labels, minlength=len(MODES)).argmax())
    d = feats.displacement
    return {
        "per_replica_viable": float(v.viable.mean()),
        "label": MODES[modal],
        "label_agreement": float(np.mean(labels == modal)),
        "median_displacement_m": float(np.median(d)),
        "p5_displacement_m": float(np.percentile(d, 5)),
        "replica_sd_m": float(np.std(d)),
        "f_body": float(np.median(feats.f_body)),
        "p95_az": float(np.median(feats.p95_az)),
        "progress_rate": float(v.progress.mean()),
        "label_constancy_rate": float(v.constant_label.mean()),
    }


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_archive(args.archive)
    genomes = np.asarray(data["solution"])
    spec = DEFAULT_SPEC

    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=args.num_envs,
            device=args.device,
            mode_channels=True,
            fall_check_every=0,
        ),
        args.fitness,
        spec,
    )
    generator = torch.Generator(device=args.device).manual_seed(0)

    seeds, report = [], []
    for idx in args.indices:
        print(f"\ndistilling v1_cpg_{idx} ({args.seeding.rounds} DAgger rounds)",
              flush=True)
        teacher = CpgTeacher(
            genomes[idx], harness, harness.control_dt,
            phase_tracking=args.phase_tracking,
        )
        genome, log = distil_cpg(teacher, harness, args.seeding, spec, generator)
        stats = score(genome, harness, args)
        seeds.append(genome)
        report.append({"source": f"v1_cpg_{idx}", "index": int(idx),
                       "distillation": log, **stats})
        print(
            f"  -> viable {stats['per_replica_viable']:.3f} per replica, "
            f"label {stats['label']} ({stats['label_agreement']:.3f}), "
            f"median {stats['median_displacement_m']:+.3f} m",
            flush=True,
        )
    harness.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, genome=torch.cat(seeds).cpu().numpy(),
        sources=np.array([r["source"] for r in report]),
    )
    write_json(out.with_suffix(".json"), report)
    print(f"\nwrote {out} ({len(seeds)} crawl seeds) and its .json")


if __name__ == "__main__":
    main()
