"""Seed the archive with the PPO walker, distilled into the genome architecture.

v1 started PGA-MAP-Elites from ~2000 randomly initialised MLPs. On a robot
whose *passive* HOME hold topples at 1.34 s that is a population of 2000
falling ducks, and the search spent its whole budget learning not to fall
instead of learning to walk diversely. Meanwhile the repo already contains a
policy that walks — the PPO velocity recipe — and Phase 1 re-trained it in
~400 iterations for pocket change.

So walking-v2 starts from that policy.

Why distillation rather than a weight copy
------------------------------------------
The rsl_rl actor is ``61 -> 512 -> 256 -> 128 -> 14``, ELU, **linear** output,
behind an :class:`EmpiricalNormalization` layer. The genome is
``61 -> 64 -> 64 -> 14``, tanh everywhere. Retraining PPO at 64x64 would fix
the widths and the normalizer folds exactly into the first layer, but the
output nonlinearity would still not match, and tanh on the output is not
cosmetic — it is what bounds the action space TD3's target-policy smoothing
clips its noise against. So the seed is *behaviour-cloned* instead: same
inputs, same resulting motion, a genome the rest of the pipeline can mutate.

DAgger, not plain behaviour cloning
-----------------------------------
Observation slot ``[34:48]`` is the previous action, so a student trained only
on the teacher's own trajectories sees the teacher's action history at training
time and its own at deployment — the classic compounding-error setup, and here
it compounds into a fall. Each round after the first therefore rolls out the
**student** and labels those states with the **teacher**, which is exactly
DAgger and costs one extra rollout per round.

The forward command is baked into the weights, not the input
------------------------------------------------------------
The QD env pins all 13 command slots to zero (v1's convention, and the
deployment idle state), and under a zero twist command the PPO walker stands
still — correctly, that is what zero command means. The teacher is therefore
queried at ``obs`` with the twist-vx slot overwritten by
:data:`SeedCfg.teacher_vx` while the student is trained on the **unmodified**
zero-command observation. The student that comes out walks forward when its
input says "stand", which is what a QD genome scored on +x displacement has to
do. Nothing about the 61-D obs contract changes; the command lives in the
weights.

    uv run python -m qd.seed --checkpoint logs/rsl_rl/.../model_399.pt \
        --out logs/qd/seeds/ppo_seed.npz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from torch import nn

from qd.common import FitnessCfg, write_json
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC, PolicySpec

# Actor observation layout (asserted against the live env in `_check_layout`).
TWIST_SLICE = slice(48, 51)
"""``[vx, vy, wz]`` of the 13-D command block — see AGENTS.md's obs invariant."""


@dataclass(frozen=True)
class SeedCfg:
    teacher_vx: float = 0.3
    """Forward twist command fed to the teacher [m/s].

    Inside the velocity task's ``lin_vel_x`` range of (-0.4, 0.4), and well
    clear of its ``walking_threshold`` of 0.01, so the teacher is unambiguously
    in its walking regime rather than its standing one."""

    rounds: int = 4
    """DAgger rounds. Round 0 rolls the teacher out; later rounds roll the
    student out and label with the teacher."""

    epochs_per_round: int = 400
    batch_size: int = 4096
    learning_rate: float = 1e-3

    jitter_count: int = 100
    """Mutated copies of the seed evaluated alongside it at iteration 0.

    A single point cannot start iso+lineDD variation (it needs two parents to
    draw a line between) and a single archive cell cannot start a QD search.
    The jitter is the seed's own neighbourhood, sampled so the archive opens
    with a spread of duty factors instead of one."""

    jitter_sigma: float = 0.02
    """Per-weight Gaussian sigma for the jitter, in genome units.

    4x the ``iso_sigma`` GA variation uses: the point is to *spread* the
    initial archive, and at 0.005 every variant lands in the seed's own cell."""


# --------------------------------------------------------------------------- #
# The teacher
# --------------------------------------------------------------------------- #


class PpoTeacher(nn.Module):
    """The rsl_rl actor, rebuilt from a checkpoint's ``actor_state_dict``.

    Deliberately not constructed through rsl_rl: the state dict is a normalizer
    plus a plain ``Linear/ELU`` stack, and rebuilding it here means the seed
    path has no opinion about which rsl_rl version wrote the file. Layer widths
    are read off the tensors, so a checkpoint trained at other dims still
    loads.
    """

    def __init__(self, state_dict: dict[str, torch.Tensor], eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", state_dict["obs_normalizer._mean"].clone())
        self.register_buffer("std", state_dict["obs_normalizer._std"].clone())

        idx = sorted(
            int(k.split(".")[1]) for k in state_dict if k.endswith(".weight") and
            k.startswith("mlp.")
        )
        layers: list[nn.Module] = []
        for n, i in enumerate(idx):
            w = state_dict[f"mlp.{i}.weight"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(state_dict[f"mlp.{i}.bias"])
            layers.append(layer)
            if n < len(idx) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.obs_dim = int(self.mean.shape[-1])
        self.requires_grad_(False).eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + self.eps))


def load_teacher(checkpoint: Path, device: str) -> PpoTeacher:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "actor_state_dict" not in payload:
        raise RuntimeError(
            f"{checkpoint} has no 'actor_state_dict' — is it an rsl_rl checkpoint?"
        )
    return PpoTeacher(payload["actor_state_dict"]).to(device)


# --------------------------------------------------------------------------- #
# Distillation
# --------------------------------------------------------------------------- #


def _check_layout(harness: PolicyRolloutHarness) -> None:
    """Fail loudly if the twist command is not where this module writes it."""
    om = harness.env.observation_manager
    offset = 0
    for name, dim in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
        width = int(dim[0]) if hasattr(dim, "__len__") else int(dim)
        if name == "command":
            if (offset, offset + width) != (TWIST_SLICE.start, TWIST_SLICE.stop):
                raise RuntimeError(
                    f"twist command sits at [{offset}:{offset + width}], not "
                    f"[{TWIST_SLICE.start}:{TWIST_SLICE.stop}]. The 61-D obs "
                    "layout moved; fix TWIST_SLICE before seeding."
                )
            return
        offset += width
    raise RuntimeError("no 'command' term in the actor observation group")


def distil_seed(
    teacher: PpoTeacher,
    harness: PolicyRolloutHarness,
    cfg: SeedCfg,
    spec: PolicySpec = DEFAULT_SPEC,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, list[dict]]:
    """Behaviour-clone ``teacher`` into one genome. Returns ``(genome, log)``."""
    _check_layout(harness)
    device = harness.device
    generator = generator or torch.Generator(device=device).manual_seed(0)
    student = spec.initial_population(1, generator, device).requires_grad_(True)
    opt = torch.optim.Adam([student], lr=cfg.learning_rate)

    def teacher_action(obs: torch.Tensor) -> torch.Tensor:
        commanded = obs.clone()
        commanded[:, TWIST_SLICE] = torch.tensor(
            [cfg.teacher_vx, 0.0, 0.0], device=obs.device
        )
        return teacher(commanded)

    obs_bank: list[torch.Tensor] = []
    label_bank: list[torch.Tensor] = []
    log: list[dict] = []

    for rnd in range(cfg.rounds):
        collected_obs: list[torch.Tensor] = []
        collected_lab: list[torch.Tensor] = []
        collected_alive: list[torch.Tensor] = []

        # Bound as defaults rather than captured: the lists are rebuilt every
        # DAgger round, and a late-bound closure would append into the wrong one.
        def on_step(
            _step,
            obs,
            _action,
            alive,
            obs_out=collected_obs,
            lab_out=collected_lab,
            alive_out=collected_alive,
        ):
            obs_out.append(obs)
            lab_out.append(teacher_action(obs))
            alive_out.append(alive)

        if rnd == 0:
            # Round 0: the teacher drives, so the states are the teacher's own.
            actor = teacher_action
        else:
            frozen = student.detach()
            actor = lambda o, g=frozen: spec.forward(
                g.expand(o.shape[0], -1), o
            )

        _, _, info, _ = harness.rollout(collect=False, actor=actor, on_step=on_step)

        keep = torch.cat(collected_alive)  # only upright states are labelled
        obs_bank.append(torch.cat(collected_obs)[keep])
        # tanh bounds the student to +-1 rad; a teacher action outside that is
        # unreachable, so clip the *target* rather than let the regression chase
        # a value the architecture cannot express. `raw_max` reports how often
        # that bites.
        raw = torch.cat(collected_lab)[keep]
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
            "rollout_survivors": int((~info["fell"]).sum()),
            "rollout_max_displacement_m": float(info["displacement"].max()),
            "rollout_mean_displacement_m": float(info["displacement"].mean()),
        }
        log.append(row)
        if verbose:
            print(
                f"  round {rnd} ({row['driver']:>7} driving) | "
                f"{row['samples']:7d} labelled states | bc loss {loss_val:.5f} | "
                f"driver survivors {row['rollout_survivors']:3d}/{harness.num_envs} "
                f"max displ {row['rollout_max_displacement_m']:+.3f} m",
                flush=True,
            )

    return student.detach().reshape(1, -1), log


def jitter(
    seed: torch.Tensor, count: int, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    """``(count, D)`` Gaussian-perturbed copies of a ``(1, D)`` seed."""
    if count <= 0:
        return seed.new_zeros((0, seed.shape[1]))
    noise = torch.randn(
        (count, seed.shape[1]), device=seed.device, generator=generator
    )
    return seed + sigma * noise


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@dataclass
class Args:
    checkpoint: Path
    """rsl_rl ``model_*.pt`` of a trained velocity policy."""

    out: Path = Path("logs/qd/seeds/ppo_seed.npz")
    num_envs: int = 256
    """Worlds per DAgger rollout; also the number of labelled trajectories."""

    device: str = "cuda:0"
    seed: int = 0
    full_collision: bool = True
    seeding: SeedCfg = field(default_factory=SeedCfg)
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    teacher = load_teacher(args.checkpoint, args.device)
    print(f"teacher: {args.checkpoint} ({teacher.obs_dim}-D obs)", flush=True)

    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=args.num_envs,
            device=args.device,
            full_collision=args.full_collision,
        ),
        args.fitness,
    )
    if teacher.obs_dim != DEFAULT_SPEC.obs_dim:
        raise RuntimeError(
            f"teacher expects {teacher.obs_dim}-D observations, the genome "
            f"contract is {DEFAULT_SPEC.obs_dim}-D"
        )

    genome, log = distil_seed(teacher, harness, args.seeding, generator=generator)

    # The number that matters: does the *student* walk, on its own, under the
    # physics the archive will judge it by?
    fitness, measures, info, _ = harness.rollout(
        genome.expand(args.num_envs, -1).contiguous(), collect=False
    )
    verdict = {
        "checkpoint": str(args.checkpoint),
        "teacher_vx": args.seeding.teacher_vx,
        "full_collision": args.full_collision,
        "episode_seconds": args.fitness.episode_seconds,
        "seed_survived": bool(~info["fell"][0]),
        "seed_displacement_m": float(info["displacement"][0]),
        "seed_fitness_m": float(fitness[0]),
        "seed_duty": [float(measures[0, 0]), float(measures[0, 1])],
        # The seed is one genome in num_envs identical worlds: the spread over
        # those worlds IS this simulator's closed-loop non-determinism, measured
        # on the very policy that has to survive it.
        "replica_survival_rate": float((~info["fell"]).mean()),
        "replica_displacement_mean_m": float(info["displacement"].mean()),
        "replica_displacement_min_m": float(info["displacement"].min()),
        "replica_displacement_max_m": float(info["displacement"].max()),
        "dagger": log,
    }
    print(
        f"\nseed: survived={verdict['seed_survived']} "
        f"displacement={verdict['seed_displacement_m']:+.3f} m  "
        f"duty=({measures[0, 0]:.2f}, {measures[0, 1]:.2f})\n"
        f"across {args.num_envs} identical replicas: "
        f"{verdict['replica_survival_rate'] * 100:.1f}% survive, displacement "
        f"{verdict['replica_displacement_min_m']:+.3f} .. "
        f"{verdict['replica_displacement_max_m']:+.3f} m",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, genome=genome.cpu().numpy())
    write_json(args.out.with_suffix(".json"), verdict)
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
