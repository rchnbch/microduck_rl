"""Roll genome sets out under world-permuted replicas and keep the behaviour features.

The one dataset three things are read from:

* **encoder training** (:mod:`qd.train_behaviour_space`) — every replica of
  every genome, viable or not. The space has to place junk too, or the search
  cannot tell a new mode from a new way of falling over;
* **replica noise** — the latent scatter of the same genome across replicas,
  which is the unit every "distinct" rule is written in;
* **the v4 re-embedding** — v4's final archives are among the sets, rolled out
  under the same replicas and the same gate as everything v5 will produce, so
  the comparison is on identical footing.

Labels from v4's classifier are recorded per replica **for reporting only**
(purity tables in the checkpoint report). Nothing downstream that inserts into
an archive reads them.

    uv run python -m qd.collect_behaviour --out logs/qd/v5/behaviour_data.npz \\
        --archives v4_walk=qd-run-archives/j007/modes_v4/final/archive_walk.npz \\
                   v4_crawl=qd-run-archives/j007/modes_v4/final/archive_crawl.npz \\
        --seeds walk_seeds=qd-run-archives/j004/seeds/ppo_seeds.npz \\
                crawl_seeds=qd-run-archives/j007/seeds/crawl_seeds.npz \\
        --random 1024 --jitter 512
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.behaviour import (
    BehaviourCfg,
    BehaviourFeatures,
    ViabilityV5Cfg,
    evaluate_viability_v5,
)
from qd.common import FitnessCfg, load_archive, write_json
from qd.modes import ModeFeatures, ViabilityCfg, evaluate_viability
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC


@dataclass
class Args:
    out: Path = Path("logs/qd/v5/behaviour_data.npz")
    archives: dict[str, Path] = field(default_factory=dict)
    """``name=path`` archive npz files (``solution`` key)."""
    seeds: dict[str, Path] = field(default_factory=dict)
    """``name=path`` seed npz files (``genome`` key)."""
    random: int = 1024
    """Random MLP genomes — the junk end of the manifold."""
    jitter: int = 512
    """Jittered copies of archive genomes, spread over ``jitter_sigmas``.

    Between a mode and junk lies the transition region a search actually
    explores; the encoder should have seen it."""
    jitter_sigmas: tuple[float, ...] = (0.005, 0.02, 0.05, 0.1)
    replicas: int = 8
    max_envs: int = 1024
    viability: ViabilityV5Cfg = field(default_factory=ViabilityV5Cfg)
    v4_viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    behaviour: BehaviourCfg = field(default_factory=BehaviourCfg)
    fitness: FitnessCfg = field(default_factory=lambda: FitnessCfg(latch_fall=False))
    seed: int = 0
    device: str = "cuda:0"


def rollout_sets(
    genomes: np.ndarray,
    reps: int,
    args: Args,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """``reps`` world-permuted rollouts of every genome; per-replica arrays ``(R, M, ...)``."""
    spec = DEFAULT_SPEC
    n = len(genomes)
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
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    windows = args.viability.windows
    names: tuple[str, ...] = ()
    per_rep: list[list[dict[str, np.ndarray]]] = [[] for _ in range(reps)]

    for start in range(0, n, chunk):
        block = genomes[start : start + chunk]
        keep = len(block)
        if keep < chunk:
            block = np.concatenate([block, np.repeat(block[:1], chunk - keep, axis=0)])
        block_t = torch.as_tensor(block, dtype=torch.float32, device=args.device)
        for r in range(reps):
            order = torch.randperm(chunk, generator=generator, device=args.device)
            inv = torch.argsort(order).cpu().numpy()
            ms = harness.make_mode_stats(windows)
            bs = harness.make_behaviour_stats(windows, args.behaviour)
            names = bs.names
            _f, _m, info, _t = harness.rollout(
                block_t[order], collect=False, mode_stats=ms, behaviour_stats=bs
            )
            info = {k: v[inv][:keep] for k, v in info.items()}
            mf = ModeFeatures.from_info(info)
            bf = BehaviourFeatures.from_info(info, names)
            v5 = evaluate_viability_v5(mf, args.viability)
            v4 = evaluate_viability(mf, args.v4_viability)
            per_rep[r].append(
                {
                    "features": bf.vector,
                    "viable": v5.viable,
                    "progress": v5.progress,
                    "stationary": v5.stationary,
                    "impact": v5.impact,
                    "finite": v5.finite,
                    "max_delta": v5.max_delta,
                    "p2_viable": v4.viable,
                    "v4_label": v4.label,
                    "displacement": mf.displacement,
                    "f_body": mf.f_body,
                    "f_air": mf.f_air,
                    "rotation_rate": mf.rotation_rate,
                    "p95_az": mf.p95_az,
                }
            )
        print(f"  rolled {min(start + chunk, n)}/{n} x {reps} replicas", flush=True)
    harness.close()
    keys = per_rep[0][0].keys()
    out = {
        k: np.stack([np.concatenate([c[k] for c in chunks]) for chunks in per_rep])
        for k in keys
    }
    return out, names


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    spec = DEFAULT_SPEC
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    sets: list[tuple[str, np.ndarray]] = []
    archive_pool = []
    for name, path in args.archives.items():
        sol = np.asarray(load_archive(path)["solution"], dtype=np.float32)
        sets.append((name, sol))
        archive_pool.append(sol)
    for name, path in args.seeds.items():
        with np.load(path) as f:
            g = np.asarray(f["genome"], dtype=np.float32).reshape(-1, spec.genome_dim)
        sets.append((name, g))
    if args.jitter > 0 and archive_pool:
        pool = np.concatenate(archive_pool)
        per_sigma = max(1, args.jitter // len(args.jitter_sigmas))
        fam = []
        for s in args.jitter_sigmas:
            base = torch.as_tensor(
                pool[rng.integers(0, len(pool), size=per_sigma)], device=args.device
            )
            noise = torch.randn(base.shape, device=args.device, generator=generator) * s
            fam.append((base + noise).cpu().numpy())
        sets.append(("jitter", np.concatenate(fam).astype(np.float32)))
    if args.random > 0:
        sets.append(
            ("random", spec.initial_population(args.random, generator, args.device).cpu().numpy())
        )

    genomes = np.concatenate([g for _n, g in sets])
    set_names = np.concatenate([[n] * len(g) for n, g in sets])
    print("sets:", {n: len(g) for n, g in sets}, "total", len(genomes), flush=True)

    data, names = rollout_sets(genomes, args.replicas, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        genomes=genomes,
        set=set_names.astype("U32"),
        feature_names=np.array(names, dtype="U64"),
        **data,
    )

    k = args.viability
    viable_count = data["viable"].sum(axis=0)
    summary = {
        "sets": {n: len(g) for n, g in sets},
        "replicas": args.replicas,
        "feature_dim": len(names),
        "gate": {
            "d_min": k.d_min, "impact_cap": k.impact_cap, "delta_max": k.delta_max,
            "exempt_seconds": k.exempt_seconds,
        },
        "per_set": {},
    }
    for n, _g in sets:
        m = set_names == n
        vc = viable_count[m]
        summary["per_set"][n] = {
            "genomes": int(m.sum()),
            "viable_5of8": int(np.sum(vc >= 5)),
            "viable_replicas_mean": float(vc.mean() / args.replicas),
            "p2_viable_5of8": int(np.sum(data["p2_viable"][:, m].sum(axis=0) >= 5)),
            "clause_rates": {
                c: float(data[c][:, m].mean())
                for c in ("finite", "progress", "stationary", "impact")
            },
            "max_delta_percentiles_over_viable_replicas": (
                np.percentile(
                    data["max_delta"][:, m][data["progress"][:, m] & data["impact"][:, m]],
                    [50, 90, 95, 99, 100],
                ).tolist()
                if np.any(data["progress"][:, m]) else None
            ),
            "median_displacement_median": float(np.median(np.median(data["displacement"][:, m], axis=0))),
            "v4_label_replica_counts": {
                str(l): int(np.sum(data["v4_label"][:, m] == l)) for l in range(5)
            },
        }
    write_json(args.out.with_suffix(".json"), summary)
    for n, d in summary["per_set"].items():
        print(f"{n:12s} {d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
