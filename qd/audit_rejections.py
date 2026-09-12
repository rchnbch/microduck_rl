"""What did v4's gate reject? — closing j017's instrumentation gap.

v4 logged ``per_mode`` counts for the candidates it ADMITTED. Roll, hop and
"other" read zero at every iteration, and that zero has two readings the log
cannot tell apart: *nothing novel was ever produced*, or *novel things were
produced and killed* by the progress / constant-label / label-agreement
clauses. ~14,000 of the 50k offspring failed those clauses with no record of
what they were.

The rejected candidates themselves are gone (the run kept elites only), so this
regenerates a sample the way v4 did — GA isoline offspring from the final v4
archives under v4's own parent budget — evaluates them under v4's own gate at
the same 8 permuted replicas, and keeps every per-replica verdict:

* the modal episode label of each REJECTED candidate (what
  ``run_modes.rejected_labels`` now logs per iteration);
* per replica, whether a rollout labelled roll / hop / other cleared the
  label-free clauses (finite, windowed progress, impact cap) — that is a novel
  behaviour that *appeared*; whether it then died on constancy or agreement is
  the *suppressed* reading;
* the clause each rejection is charged to.

PG offspring are not regenerated: v4 saved no critic. GA was half of every v4
batch and the half that explores, so the sample is representative of the
exploration channel and not of the whole.

    uv run python -m qd.audit_rejections \\
        --archive-dir qd-run-archives/j007/modes_v4/final --batches 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.modes import (
    MODES,
    ModeFeatures,
    ViabilityCfg,
    evaluate_viability,
    label_agreement,
)
from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
from qd.pga.policy_genome import DEFAULT_SPEC
from qd.pga.variation import ISO_SIGMA, LINE_SIGMA, isoline_variation

NOVEL = tuple(m for m in MODES if m not in ("walk", "crawl"))


@dataclass
class Args:
    archive_dir: Path = Path("qd-run-archives/j007/modes_v4/final")
    out: Path = Path("logs/qd/audit_rejections")
    batches: int = 3
    batch_size: int = 1024
    replicas: int = 8
    viable_min: int = 5
    label_agreement_min: int = 7
    iso_sigma: float = ISO_SIGMA
    line_sigma: float = LINE_SIGMA
    viability: ViabilityCfg = field(default_factory=ViabilityCfg)
    fitness: FitnessCfg = field(default_factory=lambda: FitnessCfg(latch_fall=False))
    seed: int = 0
    device: str = "cuda:0"


def budgeted_parents(pools: dict[str, np.ndarray], n: int, rng) -> np.ndarray:
    """v4's ``ModeArchives.sample_parents``: equal share per NON-EMPTY mode."""
    live = [m for m in MODES if m in pools and len(pools[m]) > 0]
    share, extra = divmod(n, len(live))
    out = []
    for i, mode in enumerate(live):
        k = share + (1 if i < extra else 0)
        out.append(pools[mode][rng.integers(0, len(pools[mode]), size=k)])
    return np.concatenate(out)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    pools = {}
    for mode in MODES:
        path = args.archive_dir / f"archive_{mode}.npz"
        if path.exists():
            sol = load_archive(path)["solution"]
            if len(sol):
                pools[mode] = np.asarray(sol, dtype=np.float32)
    print("parent pools:", {m: len(p) for m, p in pools.items()}, flush=True)

    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=args.batch_size,
            device=args.device,
            mode_channels=True,
            full_gait_stats=True,
            fall_check_every=0,
        ),
        args.fitness,
        DEFAULT_SPEC,
    )

    # Per candidate, per replica.
    rep_label: list[np.ndarray] = []      # (R, N) episode label
    rep_viable: list[np.ndarray] = []     # (R, N) full P2'
    rep_labelfree: list[np.ndarray] = []  # (R, N) finite & progress & impact
    rep_constant: list[np.ndarray] = []   # (R, N) clause 3
    rep_dx: list[np.ndarray] = []

    for b in range(args.batches):
        pa = torch.as_tensor(budgeted_parents(pools, args.batch_size, rng), device=args.device)
        pb = torch.as_tensor(budgeted_parents(pools, args.batch_size, rng), device=args.device)
        block = isoline_variation(pa, pb, generator, args.iso_sigma, args.line_sigma)
        labels, viable, labelfree, constant, dx = [], [], [], [], []
        for _ in range(args.replicas):
            order = torch.randperm(args.batch_size, generator=generator, device=args.device)
            inv = torch.argsort(order).cpu().numpy()
            stats = harness.make_mode_stats(args.viability.windows)
            _f, _m, info, _t = harness.rollout(block[order], collect=False, mode_stats=stats)
            info = {k: v[inv] for k, v in info.items()}
            feats = ModeFeatures.from_info(info)
            v = evaluate_viability(feats, args.viability)
            labels.append(v.label)
            viable.append(v.viable)
            labelfree.append(v.finite & v.progress & v.impact)
            constant.append(v.constant_label)
            dx.append(feats.displacement)
        rep_label.append(np.stack(labels))
        rep_viable.append(np.stack(viable))
        rep_labelfree.append(np.stack(labelfree))
        rep_constant.append(np.stack(constant))
        rep_dx.append(np.stack(dx))
        print(f"batch {b + 1}/{args.batches} done", flush=True)
    harness.close()

    L = np.concatenate(rep_label, axis=1)
    V = np.concatenate(rep_viable, axis=1)
    F = np.concatenate(rep_labelfree, axis=1)
    C = np.concatenate(rep_constant, axis=1)
    DX = np.concatenate(rep_dx, axis=1)
    R, N = L.shape

    # --- v4's insertion rule, reproduced ----------------------------------- #
    modal, _agreeing_all = label_agreement(L)
    passes = V.sum(axis=0)
    k_ok = passes >= args.viable_min
    agree_viable = (L == modal[None, :]) & V
    share = agree_viable.sum(axis=0) / np.maximum(passes, 1)
    agreeing = np.where(passes > 0, share * R, 0.0)
    agree_ok = agreeing >= args.label_agreement_min
    admitted = k_ok & agree_ok
    rejected = ~admitted

    def by_mode(mask_n: np.ndarray) -> dict[str, int]:
        return {m: int(np.sum(mask_n & (modal == i))) for i, m in enumerate(MODES)}

    # --- novel behaviours: appeared? suppressed? --------------------------- #
    novel_idx = [MODES.index(m) for m in NOVEL]
    novel_rep = np.isin(L, novel_idx)                       # (R, N) replica carries a novel label
    novel_rep_labelfree = novel_rep & F                     # ... and cleared the label-free clauses
    novel_rep_viable = novel_rep & V                        # ... and cleared clause 3 too
    per_label_rep = {
        m: {
            "replicas_labelled": int(np.sum(L == i)),
            "replicas_labelled_and_labelfree_ok": int(np.sum((L == i) & F)),
            "replicas_labelled_and_p2_viable": int(np.sum((L == i) & V)),
            "median_dx_when_labelled_m": float(np.median(DX[L == i])) if np.any(L == i) else float("nan"),
        }
        for i, m in enumerate(MODES)
    }
    # Candidates for which a novel-labelled replica passed the label-free
    # clauses: the behaviour existed and moved. How many of them were killed
    # by (a) clause 3 constancy, (b) k-of-8, (c) label agreement?
    cand_novel_appeared = novel_rep_labelfree.any(axis=0)
    cand_novel_lf_count = novel_rep_labelfree.sum(axis=0)
    cand_novel_viable_count = novel_rep_viable.sum(axis=0)
    killed_by_constancy = cand_novel_appeared & (cand_novel_viable_count == 0)
    novel_k_ok = cand_novel_viable_count >= args.viable_min
    # would a novel candidate have been admitted under a novel modal label?
    admitted_novel = admitted & np.isin(modal, novel_idx)

    # --- clause attribution for rejections --------------------------------- #
    fail_k = rejected & ~k_ok
    fail_agree_only = rejected & k_ok & ~agree_ok
    labelfree_passes = F.sum(axis=0)
    would_pass_labelfree_k = labelfree_passes >= args.viable_min
    # rejected purely by the label machinery: label-free clauses pass k-of-8,
    # but constancy (clause 3) and/or agreement killed it.
    label_only_rejections = rejected & would_pass_labelfree_k

    summary = {
        "candidates": int(N),
        "replicas": int(R),
        "operator": "GA isoline from v4 final archives under v4 parent budget (no PG)",
        "admitted": int(admitted.sum()),
        "admitted_by_modal_label": by_mode(admitted),
        "rejected": int(rejected.sum()),
        "rejected_by_modal_label": by_mode(rejected),
        "rejected_failing_k_of_8": int(fail_k.sum()),
        "rejected_failing_agreement_only": int(fail_agree_only.sum()),
        "rejected_but_labelfree_clauses_pass_k_of_8": int(label_only_rejections.sum()),
        "rejected_but_labelfree_pass_by_modal_label": by_mode(label_only_rejections),
        "per_label_replica_counts": per_label_rep,
        "novel": {
            "labels": list(NOVEL),
            "candidates_with_any_novel_replica": int(novel_rep.any(axis=0).sum()),
            "candidates_novel_replica_cleared_labelfree": int(cand_novel_appeared.sum()),
            "of_those_all_novel_replicas_failed_constancy": int(killed_by_constancy.sum()),
            "candidates_novel_viable_in_k_of_8": int(novel_k_ok.sum()),
            "candidates_admitted_under_novel_label": int(admitted_novel.sum()),
            "hist_novel_labelfree_replicas_per_candidate": np.bincount(
                cand_novel_lf_count, minlength=R + 1
            ).tolist(),
            "hist_novel_viable_replicas_per_candidate": np.bincount(
                cand_novel_viable_count, minlength=R + 1
            ).tolist(),
        },
        "clause_rates_per_replica": {
            "labelfree_ok": float(F.mean()),
            "constant_label": float(C.mean()),
            "p2_viable": float(V.mean()),
        },
        "admission_rate": float(admitted.mean()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    np.savez_compressed(
        args.out / "replicas.npz", label=L, viable=V, labelfree=F, constant=C, dx=DX,
        modal=modal, admitted=admitted,
    )
    _print(summary)


def _print(s: dict) -> None:
    print(f"\n{s['candidates']} candidates x {s['replicas']} replicas ({s['operator']})")
    print(f"admitted {s['admitted']} {s['admitted_by_modal_label']}")
    print(f"rejected {s['rejected']} by modal label {s['rejected_by_modal_label']}")
    print(f"  failing k-of-8 viability: {s['rejected_failing_k_of_8']}")
    print(f"  passing k-of-8 but failing label agreement: {s['rejected_failing_agreement_only']}")
    print(
        "  rejected although label-FREE clauses pass k-of-8 (label machinery only): "
        f"{s['rejected_but_labelfree_clauses_pass_k_of_8']} "
        f"{s['rejected_but_labelfree_pass_by_modal_label']}"
    )
    print("per-replica label counts:")
    for m, d in s["per_label_replica_counts"].items():
        print(f"  {m:6s} {d}")
    n = s["novel"]
    print(f"novel labels {n['labels']}:")
    print(f"  candidates with any novel-labelled replica:        {n['candidates_with_any_novel_replica']}")
    print(f"  ... whose novel replica cleared label-free clauses: {n['candidates_novel_replica_cleared_labelfree']}")
    print(f"      of those, every such replica failed constancy: {n['of_those_all_novel_replicas_failed_constancy']}")
    print(f"  candidates novel-viable in >= k of 8:              {n['candidates_novel_viable_in_k_of_8']}")
    print(f"  candidates admitted under a novel label:           {n['candidates_admitted_under_novel_label']}")
    print(f"  hist novel label-free replicas / candidate: {n['hist_novel_labelfree_replicas_per_candidate']}")
    print(f"  hist novel viable replicas / candidate:     {n['hist_novel_viable_replicas_per_candidate']}")
    print(f"clause rates per replica: {s['clause_rates_per_replica']}")


if __name__ == "__main__":
    main()
