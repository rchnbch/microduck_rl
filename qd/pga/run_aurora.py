"""PGA-MAP-Elites in a LEARNED behaviour space — walking-v5 (AURORA-style).

Same genome (61->64->64->14 MLP), same objective (median +x displacement over
world-permuted replicas), same variation operators and same incumbent
re-testing as v4. Four things change, each aimed at a measured v4 failure
(j017's assessment):

1. **No mode names anywhere in the insertion path.** The gate is P2''
   (:func:`qd.behaviour.evaluate_viability_v5`): finite, windowed progress,
   impact cap, and a *stationarity* clause on the support signature in place
   of P2''s constant-label clause. The classifier is not imported here.

2. **The archive is a CVT over a learned latent space**
   (:class:`qd.latent_archive.LatentArchive`). Descriptors are the encoding
   of ~80 order-invariant trajectory statistics (:mod:`qd.behaviour`) by an
   autoencoder (:mod:`qd.behaviour_space`). There are no per-mode grids and
   no per-mode parent budget, so the v4 mechanism that starved empty modes of
   parents does not exist to fail.

3. **Explicit novelty pressure.** Parents are sampled from the whole archive
   with weight ``1 / (1 + neighbours within radius)`` — sparse regions of the
   latent space are varied more, and every elite keeps a positive weight.

4. **The encoder is retrained as the archive grows** (every
   ``retrain_every`` iterations, on the archive's feature vectors plus a
   reservoir of recent candidates), the noise unit re-measured on recent
   viable candidates' replicas, the centroids regenerated, and the archive
   re-encoded — AURORA's loop. Every checkpoint also records **coverage on the
   frozen evaluation centroids** (``--eval-space``), which is the number that
   is comparable across iterations, runs and v4.

Launch::

    uv run python -m qd.pga.run_aurora --iterations 49 --batch-size 1024 \\
        --eval-space logs/qd/v5/space_eval/space_ae.npz \\
        --eval-centroids logs/qd/v5/space_eval/centroids_ae.npz \\
        --seed-genomes walk qd-run-archives/j004/seeds/ppo_seeds.npz \\
                       crawl qd-run-archives/j007/seeds/crawl_seeds_viable.npz \\
        --out-dir logs/qd/aurora_v5
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.behaviour import (
    BehaviourCfg,
    BehaviourFeatures,
    ReplicaVerdict,
    ViabilityV5Cfg,
    fold_replicas_v5,
)
from qd.behaviour_space import (
    BehaviourSpace,
    TrainCfg,
    cluster_modes,
    coverage,
    fit_space,
    latent_bounds,
    make_centroids,
)
from qd.common import FitnessCfg, write_json
from qd.hierarchy import RetestOutcome
from qd.latent_archive import LatentArchive
from qd.modes import ModeFeatures
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.run_modes import ModeRewardCfg
from qd.pga.td3 import Td3Cfg, Td3Trainer
from qd.pga.variation import ISO_SIGMA, LINE_SIGMA, isoline_variation, pg_variation
from qd.seed import SeedCfg, seed_family


@dataclass
class Args:
    out_dir: Path = Path("logs/qd/aurora_v5")
    eval_space: Path = Path("logs/qd/v5/space_eval/space_ae.npz")
    """The FROZEN evaluation encoder. Read-only; used for the comparable
    coverage curve and as the initial search encoder."""
    eval_centroids: Path = Path("logs/qd/v5/space_eval/centroids_ae.npz")

    iterations: int = 50
    batch_size: int = 1024
    initial_solutions: int = 1024

    seed_genomes: dict[str, Path] = field(default_factory=dict)
    """``name=path.npz``; names are for the log only — nothing routes on them."""
    seeding: SeedCfg = field(default_factory=SeedCfg)

    proportion_mutation_ga: float = 0.5
    iso_sigma: float = ISO_SIGMA
    line_sigma: float = LINE_SIGMA

    insertion_replicas: int = 8
    insertion_permute_worlds: bool = True
    """Standing rule (v3): a world index carries persistent bias; replicas
    permute world assignment."""
    viable_min: int = 5
    """P2'' must hold in this many of ``insertion_replicas``. The verification
    bar is the same 5-of-8 (j017: declared up front, not moved)."""

    novelty_radius_cells: float = 2.0
    """Neighbourhood radius for sparsity weighting, in units of the median
    centroid spacing. 0 = uniform parent sampling (ablation)."""

    retrain_every: int = 10
    """Iterations between encoder retrains; 0 disables (frozen search space)."""
    retrain_last: int = 10
    """No retrain in the final ``retrain_last`` iterations, so the archive
    handed to verification was filled under one geometry."""
    reservoir: int = 20_000
    """Recent candidate feature rows kept for retraining (viable or not)."""
    search_centroids: int = 1024
    refit_centroids_at: int = 2
    """Iteration at which the search centroids are first refitted to the
    viable candidates seen so far (same encoder, no retrain). 0 disables and
    keeps the evaluation set's box-uniform centroids — attempt 1's setting."""
    train: TrainCfg = field(default_factory=TrainCfg)

    retest_fraction: float = 0.1
    retest_min_pass_rate: float = 0.60

    viability: ViabilityV5Cfg = field(default_factory=ViabilityV5Cfg)
    behaviour: BehaviourCfg = field(default_factory=BehaviourCfg)
    fitness: FitnessCfg = field(default_factory=lambda: FitnessCfg(latch_fall=False))
    reward: ModeRewardCfg = field(default_factory=ModeRewardCfg)
    td3: Td3Cfg = field(default_factory=Td3Cfg)

    seed: int = 0
    device: str = "cuda:0"
    checkpoint_every: int = 5
    modes_every: int = 5
    """Iterations between (CPU-cheap) mode counts on the frozen space, for the log."""


# --------------------------------------------------------------------------- #


def evaluate_block(
    harness: PolicyRolloutHarness,
    block: torch.Tensor,
    reps: int,
    generator: torch.Generator,
    args: Args,
    bank=None,
    reward=None,
) -> tuple[ReplicaVerdict, np.ndarray]:
    """Roll a block ``reps`` times with permuted worlds; returns the folded verdict
    and the per-replica feature stack ``(R, N, D)`` (for noise calibration)."""
    per_replica = []
    stacks = []
    n = int(block.shape[0])
    for _ in range(reps):
        if args.insertion_permute_worlds:
            order = torch.randperm(n, generator=generator, device=block.device)
            inv = torch.argsort(order).cpu().numpy()
        else:
            order = torch.arange(n, device=block.device)
            inv = np.arange(n)
        ms = harness.make_mode_stats(args.viability.windows)
        bs = harness.make_behaviour_stats(args.viability.windows, args.behaviour)
        _f, _m, info, transitions = harness.rollout(
            block[order], collect=bank is not None, mode_stats=ms,
            behaviour_stats=bs, mode_reward=reward,
        )
        if bank is not None:
            bank(transitions)
        info = {k: v[inv] for k, v in info.items()}
        feats = BehaviourFeatures.from_info(info, bs.names).vector
        per_replica.append((ModeFeatures.from_info(info), feats))
        stacks.append(feats)
    return fold_replicas_v5(per_replica, args.viability, args.viable_min), np.stack(stacks)


class Reservoir:
    """Recent candidate features for retraining, and recent viable replica
    groups for the noise unit."""

    def __init__(self, size: int, rng: np.random.Generator):
        self.rows: deque[np.ndarray] = deque()
        self.count = 0
        self.size = size
        self.rng = rng
        self.groups: deque[np.ndarray] = deque(maxlen=2000)
        self.viable_rows: deque[np.ndarray] = deque()

    def add(self, verdict: ReplicaVerdict, stack: np.ndarray) -> None:
        self.rows.append(verdict.features.astype(np.float32))
        self.count += len(verdict.features)
        while sum(len(r) for r in self.rows) > self.size and len(self.rows) > 1:
            self.rows.popleft()
        # per-genome replica groups for noise calibration, and the viable
        # candidates' median features for data-driven centroids
        for j in np.flatnonzero(verdict.viable):
            self.groups.append(stack[:, j])
        if verdict.viable.any():
            self.viable_rows.append(verdict.features[verdict.viable].astype(np.float32))
            while sum(len(r) for r in self.viable_rows) > self.size and len(self.viable_rows) > 1:
                self.viable_rows.popleft()

    def features(self) -> np.ndarray:
        return np.concatenate(list(self.rows)) if self.rows else np.zeros((0, 0), np.float32)

    def viable_features(self) -> np.ndarray:
        return (
            np.concatenate(list(self.viable_rows))
            if self.viable_rows else np.zeros((0, 0), np.float32)
        )


def search_centroids(space: BehaviourSpace, archive: LatentArchive, res: Reservoir, args: Args) -> np.ndarray:
    """Centroids for the SEARCH archive: k-means on the latents of viable
    candidates seen so far (plus the archive), falling back to the box CVT
    when too few exist.

    Attempt 1 of the long run used the evaluation set's box-uniform centroids
    for the search too, and measured why that is wrong for a search archive:
    the box spans the junk end of the manifold (random MLPs), so the known
    modes are ~5 cells of walk and ~16 of crawl; at that resolution walkers
    were evicted faster than five cells could hold them, and the archive was
    15 cells and one walker by iteration 10. A data-driven CVT puts cells
    where viable behaviour is (the standard CVT-MAP-Elites construction when
    the reachable region is unknown), while the frozen EVALUATION centroids
    stay box-uniform and untouched, so every reported number is still on the
    pre-registered grid.
    """
    from sklearn.cluster import KMeans

    arc = archive.data()["features"]
    v = res.viable_features()
    parts = [a for a in (arc, v) if len(a)]
    x = np.concatenate(parts) if parts else np.zeros((0, 0))
    k = args.search_centroids
    if len(x) < k:
        all_rows = res.features()
        if len(all_rows):
            bounds = latent_bounds(space.encode(all_rows))
        else:
            bounds = np.tile([[-3.0, 3.0]], (space.latent_dim, 1)).astype(np.float32)
        return make_centroids(k, bounds, seed=args.seed)
    z = space.encode(x)
    km = KMeans(n_clusters=k, n_init=1, random_state=args.seed).fit(z)
    return km.cluster_centers_.astype(np.float32)


def retrain(space: BehaviourSpace, archive: LatentArchive, res: Reservoir, args: Args) -> tuple[BehaviourSpace, dict]:
    """Fit a new search encoder on archive + reservoir features, re-measure the
    noise unit, regenerate centroids, re-encode the archive."""
    arc = archive.data()["features"]
    x = np.concatenate([arc, res.features()]) if len(arc) else res.features()
    cfg = TrainCfg(**{**args.train.__dict__, "kind": "ae"})
    new = fit_space(x, space.names, cfg)
    groups = list(res.groups)
    if len(groups) >= 10:
        new.calibrate_to_noise(groups)
    centroids = search_centroids(new, archive, res, args)
    moved = archive.re_encode(new.encode, centroids)
    info = {"rows": len(x), "val_mse": new.train_info.get("val_mse"), **moved}
    return new, info


def eval_coverage(archive: LatentArchive, eval_space: BehaviourSpace, eval_centroids: np.ndarray) -> int:
    d = archive.data()
    if len(d["features"]) == 0:
        return 0
    return coverage(eval_space.encode(d["features"]), eval_centroids)[0]


def eval_modes(archive: LatentArchive, eval_space: BehaviourSpace, noise: float, bounds: np.ndarray) -> int:
    d = archive.data()
    if len(d["features"]) == 0:
        return 0
    return cluster_modes(eval_space.encode(d["features"]), noise, bounds).n_modes


def _log_row(it, archive, evals, rates, retest, cov, modes, elapsed, extra="") -> str:
    return (
        f"it {it:4d} | evals {evals:7d} | cells {len(archive):4d} eval-cov {cov:4d} modes {modes:2d} "
        f"| feas {rates['feasible_rate'] * 100:5.1f}% ins {rates['insertion_rate'] * 100:5.1f}% "
        f"| retest {retest.tested:3d} pass {retest.pass_rate * 100:5.1f}% evict {retest.evicted:3d} "
        f"| {elapsed:7.1f}s {extra}"
    )


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = DEFAULT_SPEC
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    eval_space = BehaviourSpace.load(args.eval_space)
    with np.load(args.eval_centroids) as f:
        eval_centroids = f["centroids"]
        eval_bounds = f["bounds"]
        eval_noise = float(f["noise"])
    # the search space starts as a copy of the frozen evaluation space
    search_space = BehaviourSpace.load(args.eval_space)
    archive = LatentArchive(eval_centroids.copy(), spec.genome_dim)
    radius = args.novelty_radius_cells * archive.centroid_spacing()

    num_envs = args.batch_size + 1
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=num_envs, device=args.device, mode_channels=True,
            full_gait_stats=True, fall_check_every=0,
        ),
        args.fitness,
        spec,
    )
    trainer = Td3Trainer(args.td3, args.device, seed=args.seed, spec=spec)
    reps = max(1, args.insertion_replicas)
    reservoir = Reservoir(args.reservoir, rng)

    def evaluate(block: torch.Tensor):
        return evaluate_block(
            harness, block, reps, generator, args, bank=trainer.buffer.add, reward=args.reward
        )

    def insert(block: torch.Tensor, verdict: ReplicaVerdict) -> dict:
        sol = block.detach().cpu().numpy()
        pick = verdict.viable
        n_in = 0
        if pick.any():
            z = search_space.encode(verdict.features[pick])
            status = archive.add(sol[pick], verdict.fitness[pick], verdict.features[pick], z)
            n_in = int(np.sum(status > 0))
        return {
            "insertion_rate": n_in / max(len(sol), 1),
            "feasible_rate": float(pick.mean()),
            "offered": int(pick.sum()),
            "inserted": n_in,
        }

    history: list[dict] = []
    t_start = time.perf_counter()
    evals = 0

    # --- iteration 0: seeds, then random MLPs ------------------------------- #
    seed_info: dict[str, dict] = {}
    for name, path in args.seed_genomes.items():
        with np.load(path) as f:
            seeds = torch.as_tensor(
                f["genome"], dtype=torch.float32, device=args.device
            ).reshape(-1, spec.genome_dim)
        if not trainer.buffer or name == "walk":
            trainer.set_greedy(seeds[:1])
        family = seed_family(
            seeds, args.seeding.jitter_count,
            args.seeding.jitter_sigmas or args.seeding.jitter_sigma, generator,
        )
        block = torch.cat(
            [family, spec.initial_population(max(0, num_envs - len(family)), generator, args.device)]
        )[:num_envs]
        verdict, stack = evaluate(block)
        reservoir.add(verdict, stack)
        rates = insert(block, verdict)
        evals += num_envs * reps
        seed_info[name] = {
            "seeds": len(seeds),
            "seeds_viable": int(verdict.viable[: len(seeds)].sum()),
            "seed_displacements_m": verdict.fitness[: len(seeds)].tolist(),
            "family_viable": int(verdict.viable[: len(family)].sum()),
            "family_size": len(family),
            **rates,
        }
        print(f"seed {name}: {seed_info[name]}", flush=True)

    remaining = args.initial_solutions
    rates = {"insertion_rate": float("nan"), "feasible_rate": float("nan")}
    while remaining > 0:
        block = spec.initial_population(num_envs, generator, args.device)
        verdict, stack = evaluate(block)
        reservoir.add(verdict, stack)
        rates = insert(block, verdict)
        evals += min(remaining, num_envs) * reps
        remaining -= num_envs

    cov = eval_coverage(archive, eval_space, eval_centroids)
    modes = eval_modes(archive, eval_space, eval_noise, eval_bounds)
    history.append(
        {
            "iteration": 0, "evaluations": evals, "elapsed_s": time.perf_counter() - t_start,
            "cells": len(archive), "eval_coverage": cov, "eval_modes": modes,
            "seeds": seed_info, **rates,
        }
    )
    print(_log_row(0, archive, evals, rates, RetestOutcome(), cov, modes, time.perf_counter() - t_start), flush=True)

    # --- the loop ----------------------------------------------------------- #
    num_ga = round(args.proportion_mutation_ga * args.batch_size)
    num_pg = args.batch_size - num_ga

    for it in range(1, args.iterations + 1):
        losses = trainer.train()
        r = radius if args.novelty_radius_cells > 0 else None

        parents_a = archive.sample_parents(num_ga, rng, r)
        parents_b = archive.sample_parents(num_ga, rng, r)
        if len(parents_a) < num_ga:
            ga = spec.initial_population(num_ga, generator, args.device)
        else:
            ga = isoline_variation(
                torch.as_tensor(parents_a, dtype=torch.float32, device=args.device),
                torch.as_tensor(parents_b, dtype=torch.float32, device=args.device),
                generator, iso_sigma=args.iso_sigma, line_sigma=args.line_sigma,
            )
        pg_parents = archive.sample_parents(num_pg, rng, r)
        if len(pg_parents) < num_pg:
            pg = spec.initial_population(num_pg, generator, args.device)
        else:
            pg = pg_variation(
                torch.as_tensor(pg_parents, dtype=torch.float32, device=args.device),
                trainer, spec=spec,
            )
        population = torch.cat([ga, pg, trainer.greedy_genome()])

        verdict, stack = evaluate(population)
        reservoir.add(verdict, stack)
        evals += population.shape[0] * reps
        rates = insert(population, verdict)

        # --- incumbent re-test ---------------------------------------------- #
        retest = RetestOutcome()
        sample = archive.sample_incumbents(args.retest_fraction, rng, num_envs)
        if sample:
            block = torch.as_tensor(
                np.stack([g for _c, g in sample]), dtype=torch.float32, device=args.device
            )
            pad = num_envs - len(block)
            if pad > 0:
                block = torch.cat([block, block[:1].repeat(pad, 1)])
            rv, _s = evaluate_block(harness, block, reps, generator, args, bank=None)
            evals += len(sample) * reps
            retest = archive.record_retest(
                [(c, g, bool(rv.viable[i])) for i, (c, g) in enumerate(sample)],
                args.retest_min_pass_rate,
            )

        # --- AURORA retrain ------------------------------------------------- #
        retrain_info = None
        if args.refit_centroids_at and it == args.refit_centroids_at:
            moved = archive.re_encode(
                search_space.encode, search_centroids(search_space, archive, reservoir, args)
            )
            radius = args.novelty_radius_cells * archive.centroid_spacing()
            retrain_info = {"centroids_refit": moved, "radius": radius}
            print(f"  refit search centroids to viable candidates: {retrain_info}", flush=True)
        if (
            args.retrain_every
            and it % args.retrain_every == 0
            and it <= args.iterations - args.retrain_last
            and len(archive) >= 10
        ):
            search_space, retrain_info = retrain(search_space, archive, reservoir, args)
            radius = args.novelty_radius_cells * archive.centroid_spacing()
            search_space.save(out / f"search_space_it{it:04d}.npz")
            print(f"  retrained encoder: {retrain_info}", flush=True)

        elapsed = time.perf_counter() - t_start
        cov = eval_coverage(archive, eval_space, eval_centroids)
        modes = (
            eval_modes(archive, eval_space, eval_noise, eval_bounds)
            if args.modes_every and it % args.modes_every == 0
            else history[-1].get("eval_modes", 0)
        )
        history.append(
            {
                "iteration": it, "evaluations": evals, "elapsed_s": elapsed,
                "cells": len(archive), "eval_coverage": cov, "eval_modes": modes,
                "clause_rates": verdict.clause_rates,
                "retest_tested": retest.tested, "retest_pass_rate": retest.pass_rate,
                "retest_evicted": retest.evicted,
                "running_pass_rate": archive.running_pass_rate(),
                "parent_weight_entropy": archive.parent_weight_entropy(radius) if r else 1.0,
                "retrain": retrain_info,
                **rates, **losses,
            }
        )
        print(_log_row(it, archive, evals, rates, retest, cov, modes, elapsed), flush=True)

        if args.checkpoint_every and it % args.checkpoint_every == 0:
            archive.save(out / f"it{it:04d}.npz", _meta(args, it, evals))
            write_json(out / "history.json", history)

    archive.save(out / "final.npz", _meta(args, args.iterations, evals))
    search_space.save(out / "search_space_final.npz")
    write_json(out / "history.json", history)
    write_json(
        out / "summary.json",
        {
            "algorithm": "PGA-MAP-Elites in a learned behaviour space (v5, AURORA-style)",
            "evaluations": evals,
            "wall_clock_s": time.perf_counter() - t_start,
            "insertion_replicas": reps,
            "viable_min": args.viable_min,
            "cells": len(archive),
            "eval_coverage": eval_coverage(archive, eval_space, eval_centroids),
            "eval_modes": eval_modes(archive, eval_space, eval_noise, eval_bounds),
            "stats": archive.stats(),
            "running_pass_rate": archive.running_pass_rate(),
            "args": args,
        },
    )
    print(f"\nwrote {out}/final.npz", flush=True)


def _meta(args: Args, it: int, evals: int) -> dict:
    return {
        "algorithm": "pga_me_aurora_v5",
        "iteration": it,
        "evaluations": evals,
        "genome": f"mlp{DEFAULT_SPEC.obs_dim}-" + "-".join(str(h) for h in DEFAULT_SPEC.hidden) + f"-{DEFAULT_SPEC.action_dim}",
        "episode_seconds": args.fitness.episode_seconds,
        "insertion_replicas": args.insertion_replicas,
        "insertion_permute_worlds": args.insertion_permute_worlds,
        "viable_min": args.viable_min,
        "predicate": "P2''",
        "d_min": args.viability.d_min,
        "impact_cap": args.viability.impact_cap,
        "delta_max": args.viability.delta_max,
        "eval_space": str(args.eval_space),
        "eval_centroids": str(args.eval_centroids),
        "novelty_radius_cells": args.novelty_radius_cells,
        "retrain_every": args.retrain_every,
    }


if __name__ == "__main__":
    main()
