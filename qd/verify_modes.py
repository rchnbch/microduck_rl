"""Independent P2' verification — the honest numbers for a v4 archive.

`qd.verify_archive` re-rolls every elite and asks whether it stayed **upright**.
That is v3's question and it has no answer for a crawl. This asks v4's: over
fresh world-permuted replicas, does the elite still clear P2', still carry the
same mode label, and still land in the cell it was filed under?

Three things it does that the v3 verifier does not:

* **scores at the gate's own threshold.** j004's lesson was that insertion
  evidence must match what verification demands; the corollary runs the other
  way too. Insertion admits at ``--viable-min`` of 8, so verification scores at
  the same k — at 7-of-8 a genuinely 0.78-robust walker passes only ~45 %, and
  a ">= 90 % of elites survive" criterion would be unachievable arithmetic
  rather than a quality bar. The **full strictness sweep is always printed**,
  so a stricter reading is one column away.
* **checks the label, not only survival.** An elite filed under "crawl" that
  verifies as a walk is not a verified crawl, however far it travelled.
* **re-bins on the mode's own descriptor**, and reports the resolvable
  resolution per sub-archive, because a cell count on a grid finer than the
  descriptor's reproducibility counts quantization rather than behaviour.

    uv run python -m qd.verify_modes --archive logs/qd/modes_v4/final/archive_walk.npz
    uv run python -m qd.verify_modes --archive <v3 archive> --as-mode walk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, load_archive, save_archive, write_json
from qd.descriptors import DescriptorCfg, grid_indices
from qd.modes import (
    MODES,
    ModeFeatures,
    ViabilityCfg,
    evaluate_viability,
    label_agreement,
)


@dataclass
class Args:
    archive: Path
    replicas: int = 8
    viable_min: int = 5
    """Replicas in which P2' must hold. Matches the insertion gate by default."""

    label_agreement_min: int = 7
    label_over_viable_only: bool = True

    as_mode: str | None = None
    """Mode to score against when the archive does not name one.

    Used to re-verify a v3 archive under P2': it predates modes, so it carries
    no mode in its metadata, and the question being asked of it is "how many of
    these are still verified *walkers* under the v4 rule"."""

    out: Path | None = None
    device: str = "cuda:0"
    max_envs: int = 512
    viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    fitness: FitnessCfg = field(
        default_factory=lambda: FitnessCfg(latch_fall=False)
    )


def rollout_replicas(
    solutions: np.ndarray,
    reps: int,
    args: Args,
    descriptor: DescriptorCfg,
) -> tuple[list[ModeFeatures], list[dict[str, np.ndarray]]]:
    """``reps`` world-permuted rollouts of every solution.

    Permuted, because a world index carries a persistent bias on this simulator
    (v3: 0.071 m of displacement spread within a slot against 0.469 m across
    slots), so replicating in place samples a sixth of the noise this pass
    exists to measure.
    """
    from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
    from qd.pga.policy_genome import DEFAULT_SPEC

    spec = DEFAULT_SPEC
    n = len(solutions)
    chunk = min(args.max_envs, max(n, 1))
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=chunk,
            device=args.device,
            mode_channels=True,
            full_gait_stats=True,
            fall_check_every=0,
        ),
        args.fitness,
        spec,
        descriptor=descriptor,
    )
    generator = torch.Generator(device=args.device).manual_seed(0)
    per_rep_features: list[list[ModeFeatures]] = [[] for _ in range(reps)]
    per_rep_axes: list[list[dict]] = [[] for _ in range(reps)]

    for start in range(0, n, chunk):
        block = solutions[start : start + chunk]
        keep = len(block)
        if keep < chunk:
            block = np.concatenate([block, np.repeat(block[:1], chunk - keep, axis=0)])
        block_t = torch.as_tensor(block, dtype=torch.float32, device=args.device)
        for r in range(reps):
            order = torch.randperm(chunk, generator=generator, device=args.device)
            inv = torch.argsort(order).cpu().numpy()
            stats = harness.make_mode_stats(args.viability.windows)
            _f, _m, info, _t = harness.rollout(
                block_t[order], collect=False, mode_stats=stats
            )
            info = {k: v[inv] for k, v in info.items()}
            feats = ModeFeatures.from_info(info)
            per_rep_features[r].append(_slice_features(feats, keep))
            per_rep_axes[r].append(
                {
                    k[len("axis/") :]: v[:keep]
                    for k, v in info.items()
                    if k.startswith("axis/")
                }
            )
    harness.close()

    merged_f = [_concat_features(chunks) for chunks in per_rep_features]
    merged_a = [
        {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        for chunks in per_rep_axes
    ]
    return merged_f, merged_a


def _slice_features(f: ModeFeatures, keep: int) -> ModeFeatures:
    kw = {n: getattr(f, n)[:, :keep] for n in ModeFeatures.PER_WINDOW}
    kw.update({n: getattr(f, n)[:keep] for n in ModeFeatures.PER_EPISODE})
    return ModeFeatures(**kw)


def _concat_features(chunks: list[ModeFeatures]) -> ModeFeatures:
    kw = {
        n: np.concatenate([getattr(c, n) for c in chunks], axis=1)
        for n in ModeFeatures.PER_WINDOW
    }
    kw.update(
        {
            n: np.concatenate([getattr(c, n) for c in chunks])
            for n in ModeFeatures.PER_EPISODE
        }
    )
    return ModeFeatures(**kw)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    data = load_archive(args.archive)
    solutions = np.asarray(data["solution"])
    meta = data.get("meta") or {}
    mode = args.as_mode or meta.get("mode")
    if mode is None:
        raise SystemExit(
            f"{args.archive} names no mode; pass --as-mode to say what it should "
            "verify as (a v3 archive predates modes and verifies as 'walk')"
        )
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}")
    descriptor = DescriptorCfg.from_meta(meta)
    dims = tuple(int(x) for x in data["grid_dims"])
    reps = max(1, args.replicas)
    n = len(solutions)

    print(
        f"verifying {n} elites of {args.archive} as '{mode}' "
        f"x {reps} permuted replicas = {n * reps} rollouts",
        flush=True,
    )
    features, axes = rollout_replicas(solutions, reps, args, descriptor)

    cfg = args.viability
    verdicts = [evaluate_viability(f, cfg) for f in features]
    viable = np.stack([v.viable for v in verdicts])  # (reps, n)
    labels = np.stack([v.label for v in verdicts])
    modal, agree_all = label_agreement(labels)
    passes = viable.sum(axis=0)

    if args.label_over_viable_only:
        agree = np.where(
            passes > 0,
            ((labels == modal[None, :]) & viable).sum(axis=0)
            / np.maximum(passes, 1)
            * reps,
            0.0,
        )
    else:
        agree = agree_all.astype(float)

    displacement = np.stack([f.displacement for f in features])
    median_displacement = np.median(displacement, axis=0)
    median_axes = {
        k: np.median(np.stack([a[k] for a in axes]), axis=0) for k in axes[0]
    }
    median_measures = descriptor.measures(median_axes)

    right_mode = np.array([MODES[int(m)] == mode for m in modal])
    survived = (
        (passes >= args.viable_min)
        & (agree >= args.label_agreement_min)
        & right_mode
    )

    # Resolvable resolution, the same construction v3 uses: an archived measure
    # is a median over `reps` replicas, so its standard error is the within-
    # elite sd over root-reps; two elites one bin apart differ by a quantity
    # whose sd is root-2 times that, so bins 2*root-2 sigma wide are the finest
    # at which adjacent cells are separated at about two sigma.
    measures = np.stack(
        [descriptor.measures({k: a[k] for k in a}) for a in axes]
    )  # (reps, n, 2)
    sigma = np.array(
        [measures[:, :, k].std(axis=0).mean() / np.sqrt(reps) for k in range(2)]
    )
    span = np.array([hi - lo for lo, hi in descriptor.ranges])
    resolvable_dims = tuple(
        int(np.clip(np.floor(span[k] / (2 * np.sqrt(2) * sigma[k])), 1, dims[k]))
        for k in range(2)
    )
    resolvable = grid_indices(median_measures, resolvable_dims, descriptor.ranges)

    def cells(mask, threshold=0.0):
        keep = mask & (median_displacement >= threshold)
        if not keep.any():
            return 0
        return len(np.unique(resolvable[keep], axis=0))

    sweep = []
    for k in range(1, reps + 1):
        s = (passes >= k) & (agree >= args.label_agreement_min) & right_mode
        sweep.append(
            {
                "viable_min": k,
                "elites": int(s.sum()),
                "resolvable_cells_0.25m": cells(s, 0.25),
                "best_median_displacement_m": (
                    float(median_displacement[s].max()) if s.any() else None
                ),
            }
        )

    summary = {
        "archive": str(args.archive),
        "mode": mode,
        "replicas": reps,
        "viable_min": args.viable_min,
        "label_agreement_min": args.label_agreement_min,
        "label_over_viable_only": args.label_over_viable_only,
        "predicate": {
            "window_seconds": cfg.windows.window_seconds,
            "stride_seconds": cfg.windows.stride_seconds,
            "d_min": cfg.d_min,
            "impact_cap": cfg.impact_cap,
        },
        "elites_raw": n,
        "elites_verified": int(survived.sum()),
        "archive_robustness": float(survived.mean()),
        "mean_viable_replicas": float(passes.mean() / reps),
        "wrong_mode": int((~right_mode).sum()),
        "wrong_mode_labels": {
            m: int(sum(1 for x in modal[~right_mode] if MODES[int(x)] == m))
            for m in MODES
            if any(MODES[int(x)] == m for x in modal[~right_mode])
        },
        "failed_label_agreement": int(
            ((passes >= args.viable_min) & (agree < args.label_agreement_min)).sum()
        ),
        "best_median_displacement_m": (
            float(median_displacement[survived].max()) if survived.any() else None
        ),
        "resolvable_dims": list(resolvable_dims),
        "grid_dims": list(dims),
        "resolvable_cells_verified_0.25m": cells(survived, 0.25),
        "resolvable_cells_verified_0.50m": cells(survived, 0.50),
        "raw_cells_verified_0.25m": (
            len(np.unique(data["index"][survived & (median_displacement >= 0.25)]))
            if (survived & (median_displacement >= 0.25)).any()
            else 0
        ),
        "archive_optimism_m": float(
            np.mean(np.asarray(data["objective"]) - median_displacement)
        ),
        "strictness_sweep": sweep,
    }
    _print(summary)

    out = args.out or args.archive.with_name(args.archive.stem + "_p2verified.npz")
    save_archive_subset(data, survived, median_displacement, median_measures, out, summary)
    write_json(Path(str(out).replace(".npz", ".json")), summary)
    print(f"\nwrote {out} and its .json")


def save_archive_subset(data, keep, fitness, measures, out: Path, summary: dict):
    """Write the verified elites back out, on their verified numbers."""
    from qd.common import make_archive

    if not keep.any():
        print("nothing survived; not writing an archive")
        return
    dims = tuple(int(x) for x in data["grid_dims"])
    ranges = [tuple(r) for r in np.asarray(data["measure_ranges"])]
    archive = make_archive(
        solution_dim=int(np.asarray(data["solution"]).shape[1]),
        grid_dims=dims,
        measure_ranges=ranges,
        qd_score_offset=-5.0,
    )
    archive.add(
        np.asarray(data["solution"])[keep], fitness[keep], measures[keep]
    )
    meta = dict(data.get("meta") or {})
    meta["p2_verified"] = summary
    save_archive(archive, out, meta)


def _print(s: dict) -> None:
    print(f"\n=== {s['mode']} @ {s['viable_min']}-of-{s['replicas']} ===")
    print(f"   elites raw                 {s['elites_raw']}")
    print(f"   elites verified            {s['elites_verified']}")
    print(f"   archive robustness         {s['archive_robustness'] * 100:.1f}%")
    print(f"   mean viable replicas       {s['mean_viable_replicas']:.3f}")
    print(f"   wrong mode                 {s['wrong_mode']} {s['wrong_mode_labels']}")
    print(f"   failed label agreement     {s['failed_label_agreement']}")
    print(f"   best verified median       {s['best_median_displacement_m']}")
    print(
        f"   resolvable grid            {s['resolvable_dims']} "
        f"of {s['grid_dims']}"
    )
    print(
        f"   cells >= 0.25 m            "
        f"{s['resolvable_cells_verified_0.25m']} resolvable / "
        f"{s['raw_cells_verified_0.25m']} raw"
    )
    print(f"   archive optimism           {s['archive_optimism_m']:+.3f} m")
    print("\n   strictness sweep")
    print(f"   {'k of N':8s} {'elites':>8s} {'cells >=0.25m':>14s} {'best median':>12s}")
    for row in s["strictness_sweep"]:
        best = row["best_median_displacement_m"]
        print(
            f"   {row['viable_min']:<8d} {row['elites']:8d} "
            f"{row['resolvable_cells_0.25m']:14d} "
            f"{'—' if best is None else format(best, '+.3f'):>12s}"
        )


if __name__ == "__main__":
    main()
