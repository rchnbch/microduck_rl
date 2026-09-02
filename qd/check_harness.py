"""Physics sanity checks for the QD rollout harness — run before any long run.

AGENTS.md: *verify physics assumptions in sim BEFORE training*. Three questions,
all of which calibrate the objective in :class:`qd.common.FitnessCfg`:

1. **How long does a passive HOME hold survive?** Checked on TILT as well as
   height — a settle test that only records z reports a fallen robot as
   "resting fine". On the walk model the answer is ~1.34 s: HOME is *not* a
   stable equilibrium, which is why the fall penalty is charged pro-rata for
   time spent down rather than as a flat subtraction.
2. **What is the standing trunk height?** Measured, never carried across model
   revisions. It must sit clear of ``FitnessCfg.fall_height``, or an upright
   rollout would be scored as fallen.
3. **Do contacts and the descriptor read back sanely?** A hand-written trotting
   CPG should move forward, and a random batch should spread across the archive
   rather than collapse into one cell.

Run::

    uv run python -m qd.check_harness
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import tyro

from qd import cpg_genome
from qd.common import FitnessCfg
from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness


@dataclass
class Args:
    num_envs: int = 64
    device: str = "cuda:0"
    settle_seconds: float = 3.0
    """Longer than the rollout's settle, per the AGENTS.md 3 s rest-pose test."""
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    harness = MicroduckRolloutHarness(
        HarnessCfg(num_envs=args.num_envs, device=args.device), args.fitness
    )

    # --- 1 & 2: hold HOME and watch height, tilt and time-to-topple --------- #
    harness.reset()
    steps = round(args.settle_seconds / harness.control_dt)
    settled_z = None
    fall_step = None
    tilt_at = {}
    for k in range(steps):
        harness.set_leg_targets(harness.home_leg_targets)
        harness.step()
        t = (k + 1) * harness.control_dt
        z = harness.base_pos()[:, 2]
        upright = -harness.projected_gravity()[:, 2]
        tilt = torch.rad2deg(torch.arccos(upright.clamp(-1, 1)))
        if settled_z is None and t >= args.fitness.settle_seconds:
            settled_z = float(z.mean())
            tilt_at["settle"] = float(tilt.mean())
        down = (z < args.fitness.fall_height) | (upright < args.fitness.fall_tilt_cos)
        if fall_step is None and bool(down.any()):
            fall_step = k
            tilt_at["fall"] = float(tilt.max())

    z = harness.base_pos()[:, 2]
    tilt_deg = torch.rad2deg(torch.arccos((-harness.projected_gravity()[:, 2]).clamp(-1, 1)))
    contact = harness.foot_contact()
    found = harness.scene.sensors["feet_ground_contact"].data.found

    print(f"--- passive HOME hold, {args.settle_seconds:.1f} s, {args.num_envs} worlds ---")
    print(f"contact tensor shape from sensor: {tuple(found.shape)}")
    print(
        f"settled trunk z at t={args.fitness.settle_seconds:.2f}s: {settled_z:.4f} m "
        f"(tilt {tilt_at.get('settle', float('nan')):.1f} deg)"
    )
    print(f"fall thresholds : height {args.fitness.fall_height} m, tilt {args.fitness.fall_tilt_deg} deg")
    print(f"height margin at settle: {(settled_z - args.fitness.fall_height) * 1000:.0f} mm")
    if fall_step is None:
        print(f"HOME held upright for the full {args.settle_seconds:.1f} s")
    else:
        print(
            f"HOME is NOT a passive equilibrium: first world trips the fall check at "
            f"t={(fall_step + 1) * harness.control_dt:.2f} s (tilt {tilt_at['fall']:.0f} deg)"
        )
    print(f"end state: z {z.mean():.4f} m, tilt {tilt_deg.mean():.1f} deg, contact L/R "
          f"{contact[:, 0].float().mean():.2f}/{contact[:, 1].float().mean():.2f}")

    assert settled_z is not None
    assert settled_z - args.fitness.fall_height > 0.02, (
        "the settled standing height is not clear of fall_height — every upright "
        "rollout would be scored as fallen"
    )

    # --- 3: a hand-written trot moves forward -------------------------------- #
    space = cpg_genome.genome_space()
    home_legs = harness.home_leg_targets[0].detach().cpu().numpy()
    trot = _hand_written_trot(space, home_legs)
    rng = np.random.default_rng(0)
    batch = np.concatenate([trot[None], space.sample(args.num_envs - 1, rng)])
    evaluator = CpgEvaluator(harness)
    fitness, measures, info = evaluator.evaluate(batch)

    print("\n--- one hand-written trot + random genomes ---")
    print(
        f"trot   : fitness {fitness[0]:+.4f} m  displacement {info['displacement'][0]:+.3f} m  "
        f"duty (L,R) = ({measures[0, 0]:.2f}, {measures[0, 1]:.2f})  "
        f"survived {info['survival_fraction'][0] * 100:.0f}%"
    )
    print(
        f"random : fitness mean {fitness[1:].mean():+.4f}  max {fitness[1:].max():+.4f}  "
        f"fell {info['fell'][1:].mean() * 100:.0f}%  "
        f"survival mean {info['survival_fraction'][1:].mean() * 100:.0f}%  "
        f"max {info['survival_fraction'][1:].max() * 100:.0f}%"
    )
    print(
        f"descriptor spread: L {measures[:, 0].min():.2f}-{measures[:, 0].max():.2f}  "
        f"R {measures[:, 1].min():.2f}-{measures[:, 1].max():.2f}"
    )
    filled = len({(int(a * 20), int(b * 20)) for a, b in np.clip(measures, 0, 0.999)})
    print(f"distinct 20x20 cells hit by {len(batch)} genomes: {filled}")
    assert np.all(np.isfinite(fitness)), "non-finite fitness"
    assert np.all((measures >= 0) & (measures <= 1)), "descriptor outside [0,1]"
    print("\nchecks passed")


def _hand_written_trot(space, home_leg_angles: np.ndarray) -> np.ndarray:
    """A plausible anti-phase leg swing — a reference point, not a tuned gait.

    Oscillates hip_pitch / knee / ankle about the HOME stance with the two legs
    half a cycle apart. If even this cannot move the robot forward, the harness
    (not the search) is broken.
    """
    g = 0.5 * (space.lower + space.upper)
    g[cpg_genome.FREQ_SLICE] = 1.6
    amp = np.zeros(cpg_genome.NUM_LEG_JOINTS)
    phase = np.zeros(cpg_genome.NUM_LEG_JOINTS)
    for base, leg_phase in ((0, 0.0), (5, np.pi)):
        amp[base + 2] = 0.30  # hip_pitch
        amp[base + 3] = 0.30  # knee
        amp[base + 4] = 0.20  # ankle
        phase[base + 2] = leg_phase
        phase[base + 3] = leg_phase + np.pi / 2
        phase[base + 4] = leg_phase
    g[cpg_genome.AMP_SLICE] = amp
    g[cpg_genome.PHASE_SLICE] = phase
    g[cpg_genome.OFFSET_SLICE] = home_leg_angles
    return space.clip(g)


if __name__ == "__main__":
    main()
