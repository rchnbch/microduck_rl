"""Stage A' — calibrating P2' and the mode classifier by measurement.

v3's Stage A asked one question of nineteen candidate axes: *does this axis
separate gaits we already know are different, by more than one genome's own
replica noise?* Stage A' asks the same question of a **predicate** and a
**classifier**, on a probe set built for the purpose (:mod:`qd.probes`), and it
asks it before either is allowed to gate an archive.

Five measurements, each writing its own section of ``stage_a_prime.json``:

1. **Predicate calibration.** Pass rate of P2' per replica over the grid
   (W in {1.5, 2, 3} s) x (d_min in {0.02, 0.05, 0.10} m), for every positive
   and every negative. The chosen setting maximises the margin between the
   worst positive and the best negative, subject to positives >= 0.95 and
   negatives <= 0.05 per replica. **If no setting clears both bars that is the
   finding**; the bars do not move.
2. **Impact cap.** p95 ``|a_z|`` per probe, and whether a cap set at 1.5x the
   worst intended-mode positive excludes the degenerates that get past clauses
   2-3. If nothing degenerate gets past them, the cap is unnecessary and is
   reported as such rather than added for tidiness.
3. **Classifier thresholds.** Per-probe median and replica sd of every
   classifier feature, per episode and per window; between-mode separation in
   units of replica sd; label agreement across replicas and across windows. A
   threshold is only usable with >= 5 replica sds of margin on both sides.
4. **Chaos profile per mode.** Displacement sd across replicas, per mode. If a
   mode is *more* chaotic than walking, its replica count and gate rule need
   their own setting rather than walking's.
5. **Negative rejection at the chosen setting**, including the v1 archives —
   632 elites that were optimised to dive — and random MLPs.

Run::

    uv run python -m qd.stage_a_prime --out logs/qd/stage_a_prime \\
        --seeds qd-run-archives/j004/seeds/ppo_seeds.npz \\
        --v3-archive qd-run-archives/j004/pga_me_v3/archive_final_verified.npz \\
        --v1-archives qd-run-archives/j003/qd/... .npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.modes import (
    MODES,
    ClassifierCfg,
    ModeFeatures,
    ViabilityCfg,
    WindowCfg,
    evaluate_viability,
    label_agreement,
)

FEATURE_NAMES = (
    "f_body",
    "f_air",
    "f_feet",
    "f_head",
    "f_inverted",
    "rotation_rate",
    "p95_az",
)

WINDOW_GRID = ((1.5, 0.5), (2.0, 1.0), (3.0, 1.0))
"""(window seconds, stride seconds). W = 1.5 needs the finer stride to stay a
whole multiple of it — see :class:`qd.modes.WindowCfg`."""

DMIN_GRID = (0.02, 0.05, 0.10)


@dataclass
class Args:
    out: Path = Path("logs/qd/stage_a_prime")
    device: str = "cuda:0"
    num_envs: int = 128
    """Worlds per batch. Also the replica count for a scripted probe: one probe
    per world means the spread across worlds *is* the replica noise."""

    genome_replicas: int = 32
    """World-permuted replicas per MLP genome positive (seeds, v3 elites)."""

    negative_replicas: int = 8
    """Replicas per v1 elite. 8 is the insertion rule's N: a negative only has
    to fail the gate the archive will actually apply."""

    seeds: Path | None = None
    v3_archive: Path | None = None
    v1_archives: tuple[Path, ...] = ()
    v1_cpg_archive: Path | None = None
    """v1's MAP-Elites CPG archive (31-D genomes) — 340 more free negatives."""
    random_mlps: int = 1024

    episode_seconds: float = 7.0
    refresh: bool = False
    """Recompute every stage even if its cache is on disk."""

    skip_genomes: bool = False
    """Scripted probes only — much faster, and enough for sections 1-4."""


# --------------------------------------------------------------------------- #
# Collecting features
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    """One behaviour's features across its replicas."""

    name: str
    intended_mode: str
    positive: bool
    source: str
    features: dict[float, ModeFeatures]
    """Keyed by window length: P2' has to be re-accumulated per (W, stride)."""

    displacement: np.ndarray


def save_results(results: list[ProbeResult], path: Path) -> Path:
    """Cache one stage's features so a crash cannot cost the whole sweep.

    Stage A' is ~300 batched rollouts and the first attempt died *after* the
    last one, tidying up a harness. Each stage now writes its own ``.npz`` and
    a rerun skips what is already on disk — which also makes the sweep cheap to
    re-analyse when a threshold moves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    meta = []
    for i, r in enumerate(results):
        meta.append(
            {
                "name": r.name,
                "intended_mode": r.intended_mode,
                "positive": r.positive,
                "source": r.source,
                "windows": sorted(r.features),
            }
        )
        for w, feats in r.features.items():
            for field_name in ModeFeatures.PER_WINDOW + ModeFeatures.PER_EPISODE:
                payload[f"{i}|{w:g}|{field_name}"] = getattr(feats, field_name)
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)
    return path


def load_results(path: Path) -> list[ProbeResult]:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta_json"]))
        out = []
        for i, m in enumerate(meta):
            features = {}
            for w in m["windows"]:
                kwargs = {
                    name: f[f"{i}|{w:g}|{name}"]
                    for name in ModeFeatures.PER_WINDOW + ModeFeatures.PER_EPISODE
                }
                features[float(w)] = ModeFeatures(**kwargs)
            out.append(
                ProbeResult(
                    m["name"], m["intended_mode"], bool(m["positive"]), m["source"],
                    features, features[2.0].displacement,
                )
            )
    return out


def cached(name: str, args: Args, build) -> list[ProbeResult]:
    """Run ``build`` unless its cache is already on disk."""
    path = Path(args.out) / f"features_{name}.npz"
    if path.exists() and not args.refresh:
        print(f"   [{name}] reusing {path}", flush=True)
        return load_results(path)
    results = build(args)
    if results:
        save_results(results, path)
        print(f"   [{name}] cached {len(results)} results -> {path}", flush=True)
    return results


def _window_cfgs(episode_seconds: float, control_dt: float) -> dict[float, WindowCfg]:
    return {
        w: WindowCfg(
            episode_seconds=episode_seconds,
            window_seconds=w,
            stride_seconds=stride,
            control_dt=control_dt,
        )
        for w, stride in WINDOW_GRID
    }


def scripted_results(args: Args) -> list[ProbeResult]:
    """Every scripted probe, at every window length, ``num_envs`` replicas each."""
    from qd import probes, spawn
    from qd.evaluate import HarnessCfg, MicroduckRolloutHarness

    fit = FitnessCfg(episode_seconds=args.episode_seconds, latch_fall=False)
    harness = MicroduckRolloutHarness(
        HarnessCfg(
            num_envs=args.num_envs,
            device=args.device,
            mode_channels=True,
            njmax=192,
            fall_check_every=0,
        ),
        fit,
    )
    windows = _window_cfgs(args.episode_seconds, harness.control_dt)
    out: list[ProbeResult] = []
    for probe in probes.PROBES:
        per_window: dict[float, ModeFeatures] = {}
        for w, cfg in windows.items():
            per_window[w] = probes.run_probe_batch(
                harness, [probe] * harness.num_envs, spawn.get(probe.spawn), cfg
            )
        out.append(
            ProbeResult(
                name=probe.name,
                intended_mode=probe.mode,
                positive=probe.positive,
                source="scripted",
                features=per_window,
                displacement=per_window[2.0].displacement,
            )
        )
    harness.close()
    return out


def genome_results(args: Args) -> list[ProbeResult]:
    """Walk positives and the v1 negatives, on the MLP harness, spawning standing."""
    from qd.pga.evaluate import PolicyHarnessCfg, PolicyRolloutHarness
    from qd.pga.policy_genome import DEFAULT_SPEC

    spec = DEFAULT_SPEC
    fit = FitnessCfg(episode_seconds=args.episode_seconds, latch_fall=False)
    harness = PolicyRolloutHarness(
        PolicyHarnessCfg(
            num_envs=args.num_envs,
            device=args.device,
            mode_channels=True,
            fall_check_every=0,
        ),
        fit,
        spec,
    )
    windows = _window_cfgs(args.episode_seconds, harness.control_dt)
    generator = torch.Generator(device=args.device).manual_seed(0)
    out: list[ProbeResult] = []

    def run_one(genome: np.ndarray, name: str, mode: str, positive: bool, source: str):
        """One genome in every world: ``num_envs`` replicas across worlds."""
        block = torch.as_tensor(
            np.repeat(np.atleast_2d(genome), harness.num_envs, axis=0),
            dtype=torch.float32,
            device=args.device,
        )
        per_window = {}
        for w, cfg in windows.items():
            stats = harness.make_mode_stats(cfg)
            _f, _m, info, _t = harness.rollout(block, collect=False, mode_stats=stats)
            per_window[w] = ModeFeatures.from_info(info)
        out.append(
            ProbeResult(name, mode, positive, source, per_window,
                        per_window[2.0].displacement)
        )

    if args.seeds is not None:
        with np.load(args.seeds) as f:
            seeds = np.asarray(f["genome"]).reshape(-1, spec.genome_dim)
        for i, g in enumerate(seeds):
            run_one(g, f"walk_seed_{i}", "walk", True, "ppo_seed")

    if args.v3_archive is not None:
        data = load_archive(args.v3_archive)
        order = np.argsort(-np.asarray(data["objective"]))[:5]
        for rank, idx in enumerate(order):
            run_one(
                data["solution"][idx], f"v3_elite_{rank}", "walk", True, "v3_archive"
            )

    # Negatives are evaluated in bulk: a negative only has to fail, so it needs
    # the gate's own replica count rather than a full noise profile.
    negatives: list[tuple[str, np.ndarray]] = []
    for path in args.v1_archives:
        data = load_archive(path)
        for i, sol in enumerate(np.asarray(data["solution"])):
            if sol.shape[-1] == spec.genome_dim:
                negatives.append((f"{Path(path).stem}_{i}", sol))
    if args.random_mlps:
        pop = spec.initial_population(args.random_mlps, generator, args.device)
        for i, g in enumerate(pop.cpu().numpy()):
            negatives.append((f"random_mlp_{i}", g))

    for start in range(0, len(negatives), harness.num_envs):
        chunk = negatives[start : start + harness.num_envs]
        block = np.stack([g for _n, g in chunk])
        pad = harness.num_envs - len(chunk)
        if pad:
            block = np.concatenate([block, np.repeat(block[:1], pad, axis=0)])
        block_t = torch.as_tensor(block, dtype=torch.float32, device=args.device)
        per_rep: dict[float, list[ModeFeatures]] = {w: [] for w in windows}
        for _ in range(args.negative_replicas):
            order = torch.randperm(harness.num_envs, generator=generator, device=args.device)
            inv = torch.argsort(order).cpu().numpy()
            for w, cfg in windows.items():
                stats = harness.make_mode_stats(cfg)
                _f, _m, info, _t = harness.rollout(
                    block_t[order], collect=False, mode_stats=stats
                )
                feats = ModeFeatures.from_info({k: v[inv] for k, v in info.items()})
                per_rep[w].append(feats)
        for i, (name, _g) in enumerate(chunk):
            per_window = {
                w: _stack_features([r for r in per_rep[w]], i) for w in windows
            }
            out.append(
                ProbeResult(name, "none", False, "v1_or_random", per_window,
                            per_window[2.0].displacement)
            )
    harness.close()
    return out


def cpg_negative_results(args: Args) -> list[ProbeResult]:
    """v1's 340 open-loop CPG elites, evaluated as negatives.

    They are free negatives with a property the MLP divers do not have: they
    are *open-loop*, so their fitness is near-deterministic (v1 measured
    re-evaluation noise <= 4.4 mm). A predicate that admits one of these admits
    it reliably, which makes them the sharpest negatives in the set.
    """
    from qd.evaluate import CpgEvaluator, HarnessCfg, MicroduckRolloutHarness

    if not args.v1_cpg_archive:
        return []
    data = load_archive(args.v1_cpg_archive)
    genomes = np.asarray(data["solution"])
    fit = FitnessCfg(episode_seconds=args.episode_seconds, latch_fall=False)
    harness = MicroduckRolloutHarness(
        HarnessCfg(
            num_envs=args.num_envs,
            device=args.device,
            mode_channels=True,
            njmax=192,
            fall_check_every=0,
        ),
        fit,
    )
    evaluator = CpgEvaluator(harness)
    windows = _window_cfgs(args.episode_seconds, harness.control_dt)
    out: list[ProbeResult] = []
    for start in range(0, len(genomes), harness.num_envs):
        block = genomes[start : start + harness.num_envs]
        keep = len(block)
        if keep < harness.num_envs:
            block = np.concatenate(
                [block, np.repeat(block[:1], harness.num_envs - keep, axis=0)]
            )
        per_window = {}
        for w, cfg in windows.items():
            stats = harness.make_mode_stats(cfg)
            per_window[w] = evaluator.evaluate_with_modes(block, stats)
        for i in range(keep):
            feats = {w: _select(per_window[w], i) for w in windows}
            out.append(
                ProbeResult(
                    f"v1_cpg_{start + i}", "none", False, "v1_cpg", feats,
                    feats[2.0].displacement,
                )
            )
    harness.close()
    return out


def _select(features: ModeFeatures, idx: int) -> ModeFeatures:
    """One env's features, kept as a length-1 batch."""
    kwargs = {n: getattr(features, n)[:, idx : idx + 1] for n in ModeFeatures.PER_WINDOW}
    kwargs.update(
        {n: getattr(features, n)[idx : idx + 1] for n in ModeFeatures.PER_EPISODE}
    )
    return ModeFeatures(**kwargs)


def _stack_features(reps: list[ModeFeatures], idx: int) -> ModeFeatures:
    """Pull genome ``idx`` out of each replica and stack them as an R-wide batch."""
    kwargs = {}
    for name in ModeFeatures.PER_WINDOW:
        kwargs[name] = np.stack([getattr(r, name)[:, idx] for r in reps], axis=1)
    for name in ModeFeatures.PER_EPISODE:
        kwargs[name] = np.stack([getattr(r, name)[idx] for r in reps])
    return ModeFeatures(**kwargs)


# --------------------------------------------------------------------------- #
# 1. Predicate calibration
# --------------------------------------------------------------------------- #


def split_positives(results: list[ProbeResult]) -> tuple[list[str], list[str]]:
    """Positives that move forward at all, and those that do not.

    A probe labelled "crawl" that travels -0.13 m is not a positive example of
    forward locomotion; it is a failed guess at how to crawl, and including it
    in "worst positive" would make every calibration setting look equally bad
    for a reason that has nothing to do with the predicate.

    The split is mechanical — median displacement over replicas > 0 — and the
    excluded names are reported. This is the §6.3 case "a positive probe fails
    P2' at every (W, d_min)": the mode does not move forward under *this*
    open-loop parameterisation, which is a physics answer about the probe, not
    a reason to move a bar.
    """
    moving, stuck = [], []
    for r in results:
        if not r.positive:
            continue
        (moving if float(np.median(r.displacement)) > 0.0 else stuck).append(r.name)
    return moving, stuck


def predicate_sweep(results: list[ProbeResult], classifier: ClassifierCfg) -> dict:
    """Per-replica P2' pass rate for every probe at every (W, d_min)."""
    moving, _stuck = split_positives(results)
    rows: dict[str, dict] = {}
    for w, stride in WINDOW_GRID:
        for d_min in DMIN_GRID:
            key = f"W={w:g},d_min={d_min:g}"
            cfg = ViabilityCfg(
                windows=WindowCfg(window_seconds=w, stride_seconds=stride),
                classifier=classifier,
                d_min=d_min,
            )
            per_probe = {}
            for r in results:
                verdict = evaluate_viability(r.features[w], cfg)
                per_probe[r.name] = float(np.mean(verdict.viable))
            positives = [per_probe[n] for n in moving]
            negatives = [per_probe[r.name] for r in results if not r.positive]
            rows[key] = {
                "per_probe": per_probe,
                "worst_positive": min(positives) if positives else None,
                "best_negative": max(negatives) if negatives else None,
                "margin": (min(positives) - max(negatives))
                if positives and negatives
                else None,
                "positives_at_or_above_0.95": sum(p >= 0.95 for p in positives),
                "positives": len(positives),
                "negatives_at_or_below_0.05": sum(n <= 0.05 for n in negatives),
                "negatives": len(negatives),
            }
    return rows


def choose_setting(sweep: dict) -> dict:
    """The (W, d_min) with the largest positive/negative margin that clears both bars."""
    clearing = {
        k: v
        for k, v in sweep.items()
        if v["positives"]
        and v["negatives"]
        and v["positives_at_or_above_0.95"] == v["positives"]
        and v["negatives_at_or_below_0.05"] == v["negatives"]
    }
    pool = clearing or sweep
    best = max(pool, key=lambda k: (pool[k]["margin"] or -1e9))
    return {
        "setting": best,
        "cleared_both_bars": bool(clearing),
        **{k: v for k, v in sweep[best].items() if k != "per_probe"},
    }


# --------------------------------------------------------------------------- #
# 2. Impact cap
# --------------------------------------------------------------------------- #


def impact_cap_report(
    results: list[ProbeResult], classifier: ClassifierCfg, window: float, d_min: float
) -> dict:
    """Does a cap at 1.5x the worst intended-mode positive earn its place?"""
    stride = WINDOW_GRID[[g[0] for g in WINDOW_GRID].index(window)][1]
    cfg = ViabilityCfg(
        windows=WindowCfg(window_seconds=window, stride_seconds=stride),
        classifier=classifier,
        d_min=d_min,
    )
    per_probe = {
        r.name: {
            "median_p95_az": float(np.median(r.features[window].p95_az)),
            "positive": r.positive,
            "passes_clauses_2_3": float(
                np.mean(evaluate_viability(r.features[window], cfg).viable)
            ),
        }
        for r in results
    }
    positives = [v["median_p95_az"] for v in per_probe.values() if v["positive"]]
    cap = 1.5 * max(positives) if positives else None
    survivors = {
        n: v
        for n, v in per_probe.items()
        if not v["positive"] and v["passes_clauses_2_3"] > 0.05
    }
    excluded = (
        {n: v for n, v in survivors.items() if v["median_p95_az"] > (cap or np.inf)}
        if cap
        else {}
    )
    return {
        "cap": cap,
        "worst_positive_p95_az": max(positives) if positives else None,
        "degenerates_passing_clauses_2_3": sorted(survivors),
        "of_those_excluded_by_the_cap": sorted(excluded),
        "cap_earns_its_place": bool(survivors) and len(excluded) >= 0.9 * len(survivors),
        "per_probe": per_probe,
    }


# --------------------------------------------------------------------------- #
# 3. Classifier thresholds and label stability
# --------------------------------------------------------------------------- #


def classifier_report(
    results: list[ProbeResult], classifier: ClassifierCfg, window: float = 2.0
) -> dict:
    per_probe = {}
    for r in results:
        feats = r.features[window]
        labels = feats.episode_labels(classifier)
        modal, agree = label_agreement(labels[None, :])
        modal_label, count = np.bincount(labels, minlength=len(MODES)).argmax(), None
        n = len(labels)
        wlabels = feats.window_labels(classifier)
        constancy = float(np.mean(np.all(wlabels[1:] == wlabels[1], axis=0)))
        per_probe[r.name] = {
            "intended_mode": r.intended_mode,
            "positive": r.positive,
            "modal_label": MODES[int(modal_label)],
            "replica_agreement": float(np.mean(labels == modal_label)),
            "window_constancy": constancy,
            "replicas": int(n),
            **{
                f"{k}_median": float(np.median(getattr(feats, k)))
                for k in FEATURE_NAMES
            },
            **{f"{k}_sd": float(np.std(getattr(feats, k))) for k in FEATURE_NAMES},
        }
        del modal, agree, count

    # Cluster the *intended* modes and report the gap between adjacent ones in
    # units of replica sd, which is the only thing that makes a threshold a
    # rule rather than a coin flip.
    clusters: dict[str, dict] = {}
    for mode in sorted({r.intended_mode for r in results if r.positive}):
        members = [
            v for v in per_probe.values() if v["positive"] and v["intended_mode"] == mode
        ]
        if not members:
            continue
        clusters[mode] = {
            k: {
                "median_of_medians": float(np.median([m[f"{k}_median"] for m in members])),
                "min": float(np.min([m[f"{k}_median"] for m in members])),
                "max": float(np.max([m[f"{k}_median"] for m in members])),
                "mean_replica_sd": float(np.mean([m[f"{k}_sd"] for m in members])),
            }
            for k in FEATURE_NAMES
        }
        clusters[mode]["probes"] = len(members)
    return {"per_probe": per_probe, "clusters": clusters}


def threshold_margin(clusters: dict, feature: str, lo_mode: str, hi_mode: str,
                     threshold: float) -> dict | None:
    """Margin in replica sds on both sides of a threshold between two clusters."""
    if lo_mode not in clusters or hi_mode not in clusters:
        return None
    lo, hi = clusters[lo_mode][feature], clusters[hi_mode][feature]
    sd = max(lo["mean_replica_sd"], hi["mean_replica_sd"], 1e-9)
    return {
        "feature": feature,
        "threshold": threshold,
        "below_cluster": lo_mode,
        "above_cluster": hi_mode,
        "gap_below_sds": (threshold - lo["max"]) / sd,
        "gap_above_sds": (hi["min"] - threshold) / sd,
        "usable": min((threshold - lo["max"]) / sd, (hi["min"] - threshold) / sd) >= 5.0,
    }


# --------------------------------------------------------------------------- #
# 4. Chaos per mode
# --------------------------------------------------------------------------- #


def chaos_report(results: list[ProbeResult], window: float = 2.0) -> dict:
    out: dict[str, dict] = {}
    for mode in sorted({r.intended_mode for r in results}):
        members = [r for r in results if r.intended_mode == mode and r.positive]
        if not members:
            continue
        out[mode] = {
            "probes": len(members),
            "mean_replica_sd_displacement_m": float(
                np.mean([np.std(r.features[window].displacement) for r in members])
            ),
            "max_replica_sd_displacement_m": float(
                np.max([np.std(r.features[window].displacement) for r in members])
            ),
            "median_displacement_m": float(
                np.median([np.median(r.features[window].displacement) for r in members])
            ),
        }
    return out


# --------------------------------------------------------------------------- #


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    classifier = ClassifierCfg()

    results = cached("scripted", args, scripted_results)
    if not args.skip_genomes:
        results += cached("genomes", args, genome_results)
        results += cached("cpg", args, cpg_negative_results)

    sweep = predicate_sweep(results, classifier)
    chosen = choose_setting(sweep)
    w = float(chosen["setting"].split(",")[0].split("=")[1])
    d = float(chosen["setting"].split("d_min=")[1])

    moving, stuck = split_positives(results)
    report = {
        "calibration_positives": moving,
        "positives_that_do_not_move_forward": {
            r.name: {
                "median_displacement_m": float(np.median(r.displacement)),
                "median_rotation_rate": float(np.median(r.features[2.0].rotation_rate)),
                "intended_mode": r.intended_mode,
            }
            for r in results
            if r.name in stuck
        },
        "probes": [
            {
                "name": r.name,
                "intended_mode": r.intended_mode,
                "positive": r.positive,
                "source": r.source,
                "median_displacement_m": float(np.median(r.displacement)),
                "replicas": int(r.displacement.shape[0]),
            }
            for r in results
        ],
        "predicate_sweep": sweep,
        "chosen": chosen,
        "impact": impact_cap_report(results, classifier, w, d),
        "classifier": classifier_report(results, classifier),
        "chaos": chaos_report(results),
    }
    report["threshold_margins"] = [
        m
        for m in (
            threshold_margin(report["classifier"]["clusters"], "f_body", "walk", "crawl", classifier.crawl_body_min),
            threshold_margin(report["classifier"]["clusters"], "rotation_rate", "crawl", "roll", classifier.roll_rate_min),
        )
        if m is not None
    ]

    _print(report, args)
    write_json(Path(args.out) / "stage_a_prime.json", report)
    print(f"\nwrote {args.out}/stage_a_prime.json")


def _print(report: dict, args: Args) -> None:
    print("\n=== probes ===")
    print(f"   {'probe':30s} {'mode':6s} {'+/-':3s} {'src':10s} {'median dx':>10s} reps")
    for p in report["probes"][:40]:
        print(
            f"   {p['name']:30s} {p['intended_mode']:6s} "
            f"{'+' if p['positive'] else '-':3s} {p['source']:10s} "
            f"{p['median_displacement_m']:10.4f} {p['replicas']}"
        )
    if len(report["probes"]) > 40:
        print(f"   ... and {len(report['probes']) - 40} more")

    print("\n=== positives that do not move forward (excluded from calibration) ===")
    for name, v in report["positives_that_do_not_move_forward"].items():
        print(
            f"   {name:30s} intended {v['intended_mode']:6s} "
            f"median dx {v['median_displacement_m']:+.4f} m  "
            f"rotation {v['median_rotation_rate']:.3f} rad/s"
        )
    print(f"   calibration positives: {len(report['calibration_positives'])}")

    print("\n=== 1. predicate sweep (per-replica pass rate) ===")
    print(f"   {'setting':22s} {'worst pos':>10s} {'best neg':>9s} {'margin':>8s} "
          f"{'pos>=.95':>9s} {'neg<=.05':>9s}")
    for k, v in report["predicate_sweep"].items():
        wp = v["worst_positive"]
        bn = v["best_negative"]
        print(
            f"   {k:22s} {wp if wp is None else round(wp, 3):>10} "
            f"{bn if bn is None else round(bn, 3):>9} "
            f"{v['margin'] if v['margin'] is None else round(v['margin'], 3):>8} "
            f"{v['positives_at_or_above_0.95']:>4}/{v['positives']:<4} "
            f"{v['negatives_at_or_below_0.05']:>4}/{v['negatives']:<4}"
        )
    c = report["chosen"]
    print(f"\n   chosen: {c['setting']}  cleared both bars: {c['cleared_both_bars']}")

    print("\n=== 2. impact cap ===")
    imp = report["impact"]
    print(f"   worst intended-mode positive p95|a_z| = {imp['worst_positive_p95_az']}")
    print(f"   cap (1.5x) = {imp['cap']}")
    print(f"   degenerates passing clauses 2-3: {imp['degenerates_passing_clauses_2_3']}")
    print(f"   of those, excluded by the cap:  {imp['of_those_excluded_by_the_cap']}")
    print(f"   cap earns its place: {imp['cap_earns_its_place']}")

    print("\n=== 3. classifier ===")
    print(f"   {'probe':30s} {'intended':8s} {'got':6s} {'agree':>6s} {'const':>6s} "
          f"{'f_body':>7s} {'f_air':>6s} {'rot':>6s}")
    for name, v in list(report["classifier"]["per_probe"].items())[:40]:
        print(
            f"   {name:30s} {v['intended_mode']:8s} {v['modal_label']:6s} "
            f"{v['replica_agreement']:6.2f} {v['window_constancy']:6.2f} "
            f"{v['f_body_median']:7.3f} {v['f_air_median']:6.3f} "
            f"{v['rotation_rate_median']:6.3f}"
        )
    for m in report["threshold_margins"]:
        print(
            f"   threshold {m['feature']} >= {m['threshold']} "
            f"({m['below_cluster']} | {m['above_cluster']}): "
            f"{m['gap_below_sds']:.1f} sd below, {m['gap_above_sds']:.1f} sd above, "
            f"usable={m['usable']}"
        )

    print("\n=== 4. chaos per mode ===")
    for mode, v in report["chaos"].items():
        print(
            f"   {mode:6s} probes {v['probes']:2d}  median dx "
            f"{v['median_displacement_m']:+.3f} m  replica sd "
            f"{v['mean_replica_sd_displacement_m']:.4f} m "
            f"(max {v['max_replica_sd_displacement_m']:.4f})"
        )
    del args


if __name__ == "__main__":
    main()
