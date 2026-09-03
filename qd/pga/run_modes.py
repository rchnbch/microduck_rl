"""PGA-MAP-Elites over modes — walking-v4.

Same genome (61->64->64->14 MLP), same objective (median +x displacement over
world-permuted replicas), same variation operators as v3. Three things change,
and they are the design accepted in ``docs/qd_mode_descriptors_draft.md``:

1. **The gate is P2', not "upright".** v3 declared a crawl dead at
   ``base z < 0.075`` and a roll dead at ``tilt > 60 deg`` before the first
   step — every non-walking rest pose this robot has is "fallen" to v3. P2'
   asks instead whether the candidate is *still going and still the same
   thing*: sustained +x progress in every 2 s window from the second onward,
   one mode label across those windows, finite state, and a cap on p95 |a_z|.
   See :mod:`qd.modes`.

2. **The archive is one 20x20 grid per mode**, walk on v3's measured axes and
   the rest on their own. Modes never compete for a cell, so "best crawl" is a
   meaningful thing to have. Parents are budgeted equally across non-empty
   modes, or 300 walkers supply 95 % of the offspring and a five-elite crawl
   archive never gets varied. See :mod:`qd.hierarchy`.

3. **Incumbents are re-tested.** A random tenth of the archive each iteration,
   folded into a per-elite running pass rate, with eviction below the gate.
   This is v3's named-but-unbuilt fix for the winner's curse *on the survival
   predicate* — and P2' is a less deterministic predicate than "upright", so
   the curse bites harder here than it did there.

The critic's per-step reward changes with the gate: ``ShapedRewardCfg`` paid
``0.10 + 0.30 * upright`` per second, which under modes is a critic that thinks
crawling is bad and would gradient-ascend every crawl toward standing up. It
becomes progress minus impact (:class:`ModeRewardCfg`), which is mode-agnostic.

Launch::

    uv run python -m qd.pga.run_modes --iterations 50 --batch-size 1024 \\
        --initial-solutions 1024 --insertion-replicas 8 \\
        --seed-genomes walk=logs/qd/seeds/ppo_seeds.npz \\
        --out-dir logs/qd/modes_v4
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, write_json
from qd.hierarchy import ModeArchives, default_mode_cfgs
from qd.modes import (
    MODES,
    ModeFeatures,
    ViabilityCfg,
    evaluate_viability,
    label_agreement,
)
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.td3 import Td3Cfg, Td3Trainer
from qd.pga.variation import ISO_SIGMA, LINE_SIGMA, isoline_variation, pg_variation
from qd.seed import SeedCfg, seed_family


@dataclass(frozen=True)
class ModeRewardCfg:
    """Per-step critic reward: progress minus impact.

    ``ShapedRewardCfg`` paid an alive bonus plus ``upright``, which was right
    while "upright" *was* the feasibility constraint and is actively wrong now:
    a critic that pays for uprightness teaches PG variation to stand every
    crawl up. Both surviving terms are mode-agnostic — a crawl and a walk are
    scored by how far they get and how gently they do it — and the impact term
    is the same quantity the P2' cap reads, so the critic and the gate agree
    about what "violent" means (AGENTS.md: a reward and its gate must measure
    the same view).

    PG variation will still mostly refine walkers until a descriptor- or
    mode-conditioned critic exists. That is the same open item v3 named; one
    critic and one buffer here, with transitions tagged by mode so a per-mode
    critic is a later option rather than a rewrite.
    """

    vel_weight: float = 1.0
    """Multiplier on ``v_x * dt``; 1.0 keeps the term literally displacement."""

    impact_weight: float = 0.02
    """Charged per unit of ``|a_z| * dt`` [m per m/s^2].

    Sized so a 10 m/s^2 sustained impact costs 0.2 m per second — comparable to
    the travel a lunging policy buys with it — rather than by taste."""


@dataclass
class Args:
    out_dir: Path = Path("logs/qd/modes_v4")

    iterations: int = 50
    batch_size: int = 1024
    initial_solutions: int = 1024

    seed_genomes: dict[str, Path] = field(default_factory=dict)
    """``mode=path.npz`` per seeded mode.

    v2 measured that isotropic mutation cannot leave the feasible manifold it
    starts on, and P2' widens the feasible set without building a bridge to the
    new parts of it. Every mode needs a seed that already does the thing, from
    a standing start."""

    seeding: SeedCfg = field(default_factory=SeedCfg)

    proportion_mutation_ga: float = 0.5
    iso_sigma: float = ISO_SIGMA
    line_sigma: float = LINE_SIGMA

    insertion_replicas: int = 8
    insertion_permute_worlds: bool = True
    """v3's measured fix: a world index carries a persistent bias (same slot
    0.071 m of displacement spread, different slots 0.469 m), so re-running the
    identical block samples a sixth of the noise the verification measures."""

    label_agreement_min: int = 7
    """Replicas that must agree on the episode mode label, out of the replica
    count. A candidate whose replicas disagree is not a robust anything.

    Stays at 7-of-8 while viability drops to 5-of-8: they are different
    questions. "Does it keep going" is a chaotic quantity on this simulator;
    "is it a walk or a crawl" is not — every probe measured 1.000 replica
    label agreement except the deliberately borderline ones."""

    viable_min: int = 5
    """Replicas in which P2' must hold, out of ``insertion_replicas``.

    The design asked for unanimous. **Stage A' measured that unanimous-8 admits
    13 % of known-good walkers** — a Microduck walker passes P2' about 0.78 of
    the time per replica because it falls, and 0.78^8 = 0.13. 5-of-8 admits
    83 % of them. The threshold barely moves the negatives (5.3 -> 6.1 expected
    admissions out of 1661, and five of those are the real crawls found in v1's
    archive), because their per-replica rates are bimodal: ~0 or 1.0.

    This is §1.4's escape hatch, invoked with the number next to it: *"if
    Stage A' measures the pass rate of the known robust probes below 0.95 per
    replica, the rule becomes k-of-8, and the reason is written down."* The
    measured rate is **0.78**.

    The cost is v2's: a weaker gate readmits marginal policies through the
    winner's curse over ~50,000 offspring. The designed answer is built and on
    by default — incumbent re-testing with eviction, below."""

    retest_fraction: float = 0.1
    retest_min_pass_rate: float = 0.60
    """Eviction threshold for an elite's running pass rate.

    Set from the measurement, not from the gate: a real walker passes P2' at
    ~0.78 and the marginal junk at ~0.125, so 0.60 sits in the empty middle. A
    bar at the gate's own 5/8 = 0.625, or at v3's 0.875, would evict genuine
    walkers for being what they measurably are."""

    viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    fitness: FitnessCfg = field(
        default_factory=lambda: FitnessCfg(latch_fall=False)
    )
    reward: ModeRewardCfg = field(default_factory=ModeRewardCfg)
    td3: Td3Cfg = field(default_factory=Td3Cfg)

    seed: int = 0
    device: str = "cuda:0"
    checkpoint_every: int = 5
    budget_checkpoint_evals: int | None = None


# --------------------------------------------------------------------------- #
# Replica folding
# --------------------------------------------------------------------------- #


@dataclass
class Verdict:
    """One candidate block's insertion verdict, folded over replicas."""

    viable: np.ndarray
    fitness: np.ndarray
    label: np.ndarray
    agreement: np.ndarray
    axes: dict[str, np.ndarray]
    clause_rates: dict[str, float]


def fold_replicas(
    per_replica: list[tuple[ModeFeatures, dict[str, np.ndarray]]],
    cfg: ViabilityCfg,
    viable_min: int,
    label_agreement_min: int,
    label_over_viable_only: bool = True,
) -> Verdict:
    """Apply §4.2's insertion rule across replicas.

    Three deliberate asymmetries, each because the failure mode being fixed is
    *luck*:

    * **viability is k-of-N**, k measured rather than assumed. The design
      asked for unanimous; Stage A' measured that a known-good walker passes
      P2' 0.78 of the time, so unanimous-8 would admit 13 % of the walkers the
      archive is supposed to hold. See ``Args.viable_min``;
    * **fitness is the median** — not the max, which is the luck-ranking being
      removed, and not the mean, which one catastrophic replica drags around;
    * **the label must agree across replicas** — a robot that sometimes walks
      and sometimes crawls is not a robust anything, and admitting it is how
      the chaos would leak into the archive's *geography* rather than only into
      its fitness.

    ``label_over_viable_only`` decides *which* replicas the label has to agree
    across, and the default changed on measurement. The design says all of
    them; but a replica in which the walker fell early is already rejected by
    the progress clause, and it is also the replica whose label flipped —
    because the fall is what flipped it. Counting it twice charges one event to
    two clauses. Measured on v3's own elites: replica label agreement over
    *all* replicas is 0.836-0.922, which fails a 7-of-8 bar 25-40 % of the
    time; over the *viable* replicas it is ~1.0, because a viable replica is by
    definition one that did not fall.

    The clause keeps its teeth where it was aimed: a candidate that walks
    viably in five replicas and crawls viably in three still fails, because the
    disagreement is among rollouts that all succeeded. Set ``False`` for the
    design's literal reading; both are reported in ``qd.check_knowns``.
    """
    verdicts = [evaluate_viability(f, cfg) for f, _a in per_replica]
    viable_stack = np.stack([v.viable for v in verdicts])
    labels = np.stack([v.label for v in verdicts])
    modal, agreeing = label_agreement(labels)

    n = viable_stack.shape[0]
    passes = viable_stack.sum(axis=0)
    viable = passes >= min(viable_min, n)

    if label_over_viable_only:
        # Agreement among the replicas that actually succeeded, scaled back to
        # the gate's k-of-n scale so the threshold means the same thing however
        # many replicas happened to be viable.
        agree_viable = (labels == modal[None, :]) & viable_stack
        share = agree_viable.sum(axis=0) / np.maximum(passes, 1)
        agreeing = np.where(passes > 0, share * n, 0.0)
    viable &= agreeing >= min(label_agreement_min, n)

    fitness = np.median(
        np.stack([f.displacement for f, _a in per_replica]), axis=0
    )
    axis_names = per_replica[0][1].keys()
    axes = {
        name: np.median(np.stack([a[name] for _f, a in per_replica]), axis=0)
        for name in axis_names
    }
    clause_rates = {
        k: float(np.mean([v.rates()[k] for v in verdicts]))
        for k in ("finite", "progress", "constant_label", "impact")
    }
    clause_rates["label_agreement"] = float(
        np.mean(agreeing >= min(label_agreement_min, n))
    )
    clause_rates["viable_replicas_mean"] = float(np.mean(passes))
    return Verdict(viable, fitness, modal, agreeing, axes, clause_rates)


def evaluate_block(
    harness,
    block: torch.Tensor,
    reps: int,
    generator: torch.Generator,
    viability: ViabilityCfg,
    permute: bool,
    viable_min: int,
    label_agreement_min: int,
    bank=None,
    reward=None,
    label_over_viable_only: bool = True,
) -> Verdict:
    """Roll a block out ``reps`` times, permuting world assignment each time."""
    per_replica = []
    n = int(block.shape[0])
    for _ in range(reps):
        if permute:
            order = torch.randperm(n, generator=generator, device=block.device)
            inv = torch.argsort(order).cpu().numpy()
        else:
            order = torch.arange(n, device=block.device)
            inv = np.arange(n)
        stats = harness.make_mode_stats(viability.windows)
        _f, _m, info, transitions = harness.rollout(
            block[order],
            collect=bank is not None,
            mode_stats=stats,
            mode_reward=reward,
        )
        if bank is not None:
            bank(transitions)
        info = {k: v[inv] for k, v in info.items()}
        features = ModeFeatures.from_info(info)
        axes = {k[len("axis/") :]: v for k, v in info.items() if k.startswith("axis/")}
        per_replica.append((features, axes))
    return fold_replicas(
        per_replica, viability, viable_min, label_agreement_min,
        label_over_viable_only,
    )


# --------------------------------------------------------------------------- #


def insert(archives: ModeArchives, genomes: torch.Tensor, verdict: Verdict) -> dict:
    """Offer the viable candidates to the sub-archive their label names."""
    solutions = genomes.detach().cpu().numpy()
    per_mode: dict[str, dict[str, float]] = {}
    inserted = 0
    for m, mode in enumerate(MODES):
        pick = verdict.viable & (verdict.label == m)
        if not pick.any():
            continue
        status = archives.add(
            mode,
            solutions[pick],
            verdict.fitness[pick],
            {k: v[pick] for k, v in verdict.axes.items()},
        )
        n_in = int(np.sum(np.asarray(status) > 0))
        inserted += n_in
        per_mode[mode] = {
            "offered": int(pick.sum()),
            "inserted": n_in,
        }
    total = max(len(solutions), 1)
    return {
        "insertion_rate": inserted / total,
        "feasible_rate": float(verdict.viable.mean()),
        "per_mode": per_mode,
    }


def _log_row(it: int, archives: ModeArchives, evals: int, rates: dict,
             retest, elapsed: float) -> str:
    occ = archives.occupancy()
    counts = " ".join(f"{m[:2]}{occ[m]:4d}" for m in MODES if m in occ)
    return (
        f"it {it:4d} | evals {evals:7d} | {counts} "
        f"| feas {rates['feasible_rate'] * 100:5.1f}% "
        f"| ins {rates['insertion_rate'] * 100:5.1f}% "
        f"| retest {retest.tested:3d} pass {retest.pass_rate * 100:5.1f}% "
        f"evict {retest.evicted:3d} | {elapsed:6.1f}s"
    )


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = DEFAULT_SPEC
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    archives = ModeArchives(
        solution_dim=spec.genome_dim,
        cfgs=default_mode_cfgs(),
        qd_score_offset=args.fitness.min_fitness,
        seed=args.seed,
    )
    num_envs = args.batch_size + 1
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=num_envs,
            device=args.device,
            mode_channels=True,
            full_gait_stats=True,
            fall_check_every=0,
        ),
        args.fitness,
        spec,
    )
    trainer = Td3Trainer(args.td3, args.device, seed=args.seed, spec=spec)
    reps = max(1, args.insertion_replicas)

    def evaluate(block: torch.Tensor) -> Verdict:
        return evaluate_block(
            harness,
            block,
            reps,
            generator,
            args.viability,
            args.insertion_permute_worlds,
            args.viable_min,
            args.label_agreement_min,
            bank=trainer.buffer.add,
            reward=args.reward,
        )

    history: list[dict] = []
    t_start = time.perf_counter()
    evals = 0

    # --- iteration 0: seeds per mode, then random MLPs ---------------------- #
    seed_info: dict[str, dict] = {}
    for mode, path in args.seed_genomes.items():
        if mode not in MODES:
            raise SystemExit(f"--seed-genomes names unknown mode {mode!r}")
        with np.load(path) as f:
            seeds = torch.as_tensor(
                f["genome"], dtype=torch.float32, device=args.device
            ).reshape(-1, spec.genome_dim)
        if mode == "walk":
            trainer.set_greedy(seeds[:1])
        family = seed_family(
            seeds,
            args.seeding.jitter_count,
            args.seeding.jitter_sigmas or args.seeding.jitter_sigma,
            generator,
        )
        block = torch.cat(
            [family, spec.initial_population(max(0, num_envs - len(family)),
                                             generator, args.device)]
        )[:num_envs]
        verdict = evaluate(block)
        rates = insert(archives, block, verdict)
        evals += num_envs * reps
        n_seeds = len(seeds)
        seed_info[mode] = {
            "seeds": n_seeds,
            "seeds_viable": int(verdict.viable[:n_seeds].sum()),
            "seed_labels": [MODES[int(i)] for i in verdict.label[:n_seeds]],
            "seed_displacements_m": verdict.fitness[:n_seeds].tolist(),
            "family_viable": int(verdict.viable[: len(family)].sum()),
            "family_size": len(family),
            **rates,
        }
        print(
            f"seed {mode}: {seed_info[mode]['seeds_viable']}/{n_seeds} viable, "
            f"labels {seed_info[mode]['seed_labels']}, "
            f"family {seed_info[mode]['family_viable']}/{len(family)}",
            flush=True,
        )

    remaining = args.initial_solutions
    rates = {"insertion_rate": float("nan"), "feasible_rate": float("nan")}
    while remaining > 0:
        block = spec.initial_population(num_envs, generator, args.device)
        verdict = evaluate(block)
        rates = insert(archives, block, verdict)
        evals += min(remaining, num_envs) * reps
        remaining -= num_envs

    from qd.hierarchy import RetestOutcome

    history.append(
        {
            "iteration": 0,
            "evaluations": evals,
            "elapsed_s": time.perf_counter() - t_start,
            "occupancy": archives.occupancy(),
            "seeds": seed_info,
            **rates,
        }
    )
    print(_log_row(0, archives, evals, rates, RetestOutcome(), time.perf_counter() - t_start),
          flush=True)

    # --- the loop ----------------------------------------------------------- #
    num_ga = round(args.proportion_mutation_ga * args.batch_size)
    num_pg = args.batch_size - num_ga

    for it in range(1, args.iterations + 1):
        losses = trainer.train()

        parents_a = archives.sample_parents(num_ga, rng)
        parents_b = archives.sample_parents(num_ga, rng)
        if len(parents_a) < num_ga or len(parents_b) < num_ga:
            # Nothing feasible to vary yet: random MLPs keep the buffer filling
            # and say plainly in the log that the search has no foothold.
            ga = spec.initial_population(num_ga, generator, args.device)
        else:
            ga = isoline_variation(
                torch.as_tensor(parents_a, dtype=torch.float32, device=args.device),
                torch.as_tensor(parents_b, dtype=torch.float32, device=args.device),
                generator,
                iso_sigma=args.iso_sigma,
                line_sigma=args.line_sigma,
            )
        pg_parents = archives.sample_parents(num_pg, rng)
        if len(pg_parents) < num_pg:
            pg = spec.initial_population(num_pg, generator, args.device)
        else:
            pg = pg_variation(
                torch.as_tensor(pg_parents, dtype=torch.float32, device=args.device),
                trainer,
                spec=spec,
            )
        population = torch.cat([ga, pg, trainer.greedy_genome()])

        verdict = evaluate(population)
        evals += population.shape[0] * reps
        rates = insert(archives, population, verdict)

        # --- incumbent re-test ---------------------------------------------- #
        retest = RetestOutcome()
        sample = archives.sample_incumbents(args.retest_fraction, rng)
        if sample:
            block = torch.as_tensor(
                np.stack([g for _m, _c, g in sample]),
                dtype=torch.float32,
                device=args.device,
            )
            pad = num_envs - len(block)
            if pad > 0:
                block = torch.cat([block, block[:1].repeat(pad, 1)])
            rv = evaluate_block(
                harness, block, reps, generator, args.viability,
                args.insertion_permute_worlds, args.viable_min,
                args.label_agreement_min, bank=None,
            )
            evals += len(sample) * reps
            retest = archives.record_retest(
                [
                    (m, c, g, bool(rv.viable[i]) and MODES[int(rv.label[i])] == m)
                    for i, (m, c, g) in enumerate(sample)
                ],
                args.retest_min_pass_rate,
            )

        elapsed = time.perf_counter() - t_start
        history.append(
            {
                "iteration": it,
                "evaluations": evals,
                "elapsed_s": elapsed,
                "occupancy": archives.occupancy(),
                "parent_budget": archives.parent_budget(num_ga),
                "clause_rates": verdict.clause_rates,
                "retest_tested": retest.tested,
                "retest_pass_rate": retest.pass_rate,
                "retest_evicted": retest.evicted,
                "retest_evicted_by_mode": retest.evicted_by_mode,
                "running_pass_rates": archives.pass_rates(),
                **rates,
                **losses,
            }
        )
        print(_log_row(it, archives, evals, rates, retest, elapsed), flush=True)

        if args.checkpoint_every and it % args.checkpoint_every == 0:
            archives.save(out / f"it{it:04d}", _meta(args, it, evals))
            write_json(out / "history.json", history)

    archives.save(out / "final", _meta(args, args.iterations, evals))
    write_json(out / "history.json", history)
    write_json(
        out / "summary.json",
        {
            "algorithm": "PGA-MAP-Elites over modes (v4)",
            "evaluations": evals,
            "wall_clock_s": time.perf_counter() - t_start,
            "insertion_replicas": reps,
            "viable_min": args.viable_min,
            "label_agreement_min": args.label_agreement_min,
            "occupancy": archives.occupancy(),
            "stats": archives.stats(),
            "running_pass_rates": archives.pass_rates(),
            "args": args,
        },
    )
    print(f"\nwrote {out}/final/archive_<mode>.npz", flush=True)


def _meta(args: Args, it: int, evals: int) -> dict:
    return {
        "algorithm": "pga_me_modes_v4",
        "iteration": it,
        "evaluations": evals,
        "genome": f"mlp{DEFAULT_SPEC.obs_dim}-"
        + "-".join(str(h) for h in DEFAULT_SPEC.hidden)
        + f"-{DEFAULT_SPEC.action_dim}",
        "episode_seconds": args.fitness.episode_seconds,
        "insertion_replicas": args.insertion_replicas,
        "insertion_permute_worlds": args.insertion_permute_worlds,
        "predicate": "P2'",
        "window_seconds": args.viability.windows.window_seconds,
        "stride_seconds": args.viability.windows.stride_seconds,
        "d_min": args.viability.d_min,
        "impact_cap": args.viability.impact_cap,
    }


if __name__ == "__main__":
    main()
