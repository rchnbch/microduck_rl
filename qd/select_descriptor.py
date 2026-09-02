"""Pick the archive's behaviour axes by measurement — walking-v3, Stage A.

Walking-v2 shipped a descriptor this robot cannot move in, and only found out
after a 207k-evaluation run. The failure was cheap to have caught: the question
"does this axis separate gaits we already know are different, by more than one
genome's own replica noise?" costs eleven genomes times sixty-four rollouts,
about a minute on an RTX 3060.

So v3 asks it first, of every candidate axis in :data:`qd.descriptors.AXES`, on
a fixed **measurement set** of gaits that are known to be distinct and known to
be robust:

* the six PPO teacher gaits distilled by :mod:`qd.seed` — twist commands from a
  0.1 m/s shuffle to a 0.4 m/s stride plus a strafe and a turn, spanning a 6x
  range in how far the teacher travels, every one of them 98-100% replica-robust;
* the five elites of j003's archive that survived 7-of-8 verification.

Two numbers per axis, and the ratio between them is the whole selection:

* **between-gait spread** — the range of the eleven gaits' *median* values. How
  much of the axis real, feasible, structurally different gaits actually use.
* **within-genome replica noise** — the mean across gaits of one genome's
  standard deviation over its replicas. What the simulator's chaos alone moves
  the axis by, with the genome held byte-identical.

An axis whose spread is smaller than its noise is measuring the contact solver,
not the gait. Duty factor is included as the control and is expected to fail:
v2 measured its spread at 0.03 against a replica sd of 0.014.

    uv run python -m qd.select_descriptor \\
        --seeds qd-run-archives/j003/qd/seeds/ppo_seeds.npz \\
        --elites qd-run-archives/j003/qd/pga_me_v2_reeval/archive_final_verified.npz \\
        --replicas 64 --out logs/qd/descriptor_selection

Writes ``selection.json`` (every number) and ``selection.md`` (the table that
goes in the README), and prints the ranked axis pairs with how many of the
eleven gaits each pair separates into distinct cells of a 20x20 grid.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

from qd.common import FitnessCfg, load_archive, write_json
from qd.descriptors import AXES, AXIS_NAMES, DescriptorCfg
from qd.replay import infer_kind, reevaluate

# The eleven gaits, in the order the table prints them.
SEED_LABELS = (
    "seed vx=0.10",
    "seed vx=0.20",
    "seed vx=0.30",
    "seed vx=0.40",
    "seed fwd+strafe",
    "seed fwd+turn",
)


@dataclass
class Args:
    seeds: Path = Path("qd-run-archives/j003/qd/seeds/ppo_seeds.npz")
    """``.npz`` from :mod:`qd.seed` holding the distilled teacher gaits."""

    elites: Path | None = Path(
        "qd-run-archives/j003/qd/pga_me_v2_reeval/archive_final_verified.npz"
    )
    """Verified archive whose elites join the measurement set (may be None)."""

    replicas: int = 64
    """Byte-identical rollouts per genome.

    64 puts the standard error of a per-gait median at about an eighth of the
    replica sd, which is small next to the spreads being compared."""

    grid_dims: tuple[int, int] = (20, 20)
    pad_fraction: float = 0.10
    """Grid ranges are the measurement set's extremes widened by this much of
    the span on each side, so the seeds do not sit in the boundary cells."""

    min_ratio: float = 3.0
    """Spread-to-noise below which an axis is not offered as a candidate."""

    max_noise_fraction: float = 0.10
    """Acceptance threshold: replica sd must be under this share of the axis's
    measurement-set range. The job's criterion, applied here rather than
    checked afterwards."""

    mutation_probe: int = 512
    """Feasible GA offspring of the measurement set to probe for *reachability*.

    Spread and noise say whether an axis can tell known gaits apart. They do
    not say whether the search can **move** along it — and that is the half
    walking-v2 got wrong: duty factor separated nothing *and* isotropic weight
    mutation could not shift it, so the archive never grew. This probe mutates
    the measurement set with the run's own GA operator (iso+lineDD at the tuned
    sigmas), keeps the offspring that stay upright, and reports how far each
    axis travels among them, in units of that axis's replica noise. 0 disables
    it."""

    device: str = "cuda:0"
    max_envs: int = 512
    out: Path = Path("logs/qd/descriptor_selection")
    fitness: FitnessCfg = field(default_factory=FitnessCfg)


def _load_measurement_set(args: Args) -> tuple[np.ndarray, list[str]]:
    with np.load(args.seeds) as f:
        seeds = np.asarray(f["genome"]).reshape(-1, np.asarray(f["genome"]).shape[-1])
    labels = [
        SEED_LABELS[i] if i < len(SEED_LABELS) else f"seed {i}" for i in range(len(seeds))
    ]
    genomes = [seeds]
    if args.elites is not None and Path(args.elites).exists():
        data = load_archive(args.elites)
        elites = np.asarray(data["solution"])
        if elites.shape[1] != seeds.shape[1]:
            raise SystemExit(
                f"elite genomes are {elites.shape[1]}-D and seeds are "
                f"{seeds.shape[1]}-D; they must come from the same genome class."
            )
        genomes.append(elites)
        labels += [f"j003 elite {i} (cell {int(c)})" for i, c in enumerate(data["index"])]
    return np.concatenate(genomes).astype(np.float32), labels


def measure(args: Args) -> dict:
    """Roll the measurement set out and return every candidate axis's stats."""
    genomes, labels = _load_measurement_set(args)
    n, reps = len(genomes), max(1, args.replicas)
    print(
        f"measuring {n} gaits x {reps} replicas = {n * reps} rollouts "
        f"over {len(AXIS_NAMES)} candidate axes",
        flush=True,
    )

    # Gait-major, so a reshape recovers the per-gait axis after chunking.
    _, _, info, _dt = reevaluate(
        np.repeat(genomes, reps, axis=0),
        infer_kind(genomes),
        args.fitness,
        args.device,
        args.max_envs,
        full_gait_stats=True,
    )
    upright = ~info["fell"].reshape(n, reps)
    displacement = info["displacement"].reshape(n, reps)

    median_displacement = np.median(displacement, axis=1)
    rows = []
    per_axis_values: dict[str, np.ndarray] = {}
    for name in AXIS_NAMES:
        values = info[f"axis/{name}"].reshape(n, reps).astype(np.float64)
        # A fallen replica's descriptor is averaged over a truncated episode and
        # is not a measurement of the gait; the measurement set is 98-100%
        # robust, so this drops a handful of rows, never a gait.
        values = np.where(upright, values, np.nan)
        per_axis_values[name] = values
        if np.all(np.isnan(values)):
            rows.append(
                {
                    "axis": name,
                    "label": AXES[name].label,
                    "supplied": False,
                }
            )
            continue
        medians = np.nanmedian(values, axis=1)
        sds = np.nanstd(values, axis=1)
        spread = float(np.nanmax(medians) - np.nanmin(medians))
        noise = float(np.nanmean(sds))
        full_range = float(np.nanmax(values) - np.nanmin(values))
        rows.append(
            {
                "axis": name,
                "label": AXES[name].label,
                "supplied": True,
                "gait_medians": medians.tolist(),
                "gait_replica_sds": sds.tolist(),
                "between_gait_spread": spread,
                "replica_sd": noise,
                "measurement_range": full_range,
                "measurement_min": float(np.nanmin(values)),
                "measurement_max": float(np.nanmax(values)),
                "spread_to_noise": float(spread / noise) if noise > 0 else float("inf"),
                # An axis that is a monotone restatement of the objective makes
                # the archive a fitness ladder rather than a behaviour map: its
                # "diverse" cells are just slower walkers. Reported so the
                # choice can weigh that explicitly.
                "corr_with_displacement": (
                    float(np.corrcoef(medians, median_displacement)[0, 1])
                    if np.std(medians) > 0
                    else float("nan")
                ),
                "noise_fraction_of_range": (
                    float(noise / full_range) if full_range > 0 else float("inf")
                ),
            }
        )

    return {
        "labels": labels,
        "replicas": reps,
        "survival_rate": upright.mean(axis=1).tolist(),
        "median_displacement_m": np.median(displacement, axis=1).tolist(),
        "axes": rows,
        "_values": per_axis_values,
    }


def probe_mutation_reach(args: Args, result: dict) -> dict:
    """How far the run's own GA operator moves each axis, among survivors.

    One batched rollout of ``--mutation-probe`` iso+lineDD offspring of the
    measurement set. Only full-episode survivors are counted, because only
    those are ever inserted — an axis that moves only by falling over is the
    v1 coverage mistake in a new costume.
    """
    import torch

    from qd.pga.variation import ISO_SIGMA, LINE_SIGMA, isoline_variation

    genomes, _ = _load_measurement_set(args)
    n_probe = int(args.mutation_probe)
    gen = torch.Generator(device=args.device).manual_seed(0)
    parents = torch.as_tensor(genomes, dtype=torch.float32, device=args.device)
    pick = lambda: parents[
        torch.randint(len(parents), (n_probe,), generator=gen, device=args.device)
    ]
    offspring = isoline_variation(
        pick(), pick(), gen, iso_sigma=ISO_SIGMA, line_sigma=LINE_SIGMA
    )
    print(
        f"mutation-reach probe: {n_probe} iso+lineDD offspring "
        f"(iso {ISO_SIGMA}, line {LINE_SIGMA}) of the measurement set",
        flush=True,
    )
    _, _, info, _dt = reevaluate(
        offspring.cpu().numpy(),
        "mlp",
        args.fitness,
        args.device,
        args.max_envs,
        full_gait_stats=True,
    )
    alive = ~info["fell"]
    out = {"offspring": n_probe, "feasible": int(alive.sum())}
    by_name = {r["axis"]: r for r in result["axes"] if r.get("supplied")}
    result["_probe_values"] = {
        name: info[f"axis/{name}"][alive].astype(np.float64) for name in by_name
    }
    for name, row in by_name.items():
        values = info[f"axis/{name}"][alive].astype(np.float64)
        values = values[np.isfinite(values)]
        if values.size < 8:
            continue
        lo, hi = np.percentile(values, [5, 95])
        row["mutation_reach"] = float(hi - lo)
        row["mutation_reach_sds"] = (
            float((hi - lo) / row["replica_sd"])
            if row["replica_sd"] > 0
            else float("inf")
        )
        row["mutation_p5_p95"] = [float(lo), float(hi)]
    return out


def padded_range(row: dict, pad: float) -> tuple[float, float]:
    lo, hi = row["measurement_min"], row["measurement_max"]
    span = max(hi - lo, 1e-6)
    return (lo - pad * span, hi + pad * span)


def _cells(values_x, values_y, rx, ry, dims) -> np.ndarray:
    """Cell index of each gait's median under a 20x20 grid over ``rx``/``ry``."""
    def binned(v, r, d):
        lo, hi = r
        idx = np.floor((np.asarray(v) - lo) / (hi - lo) * d)
        return np.clip(idx, 0, d - 1).astype(int)

    return binned(values_x, rx, dims[0]) * dims[1] + binned(values_y, ry, dims[1])


def rank_pairs(result: dict, args: Args) -> list[dict]:
    """Every admissible axis pair, ranked by how many gaits it separates."""
    by_name = {r["axis"]: r for r in result["axes"] if r.get("supplied")}
    ok = [
        name
        for name, r in by_name.items()
        if r["noise_fraction_of_range"] < args.max_noise_fraction
        and r["spread_to_noise"] >= args.min_ratio
    ]
    pairs = []
    for x, y in itertools.combinations(ok, 2):
        if {x, y} <= {"energy_per_meter", "cost_of_transport"} or {x, y} <= {
            "duty_left",
            "duty_right",
            "duty_mean",
        }:
            # Same quantity twice: cost of transport is energy per metre over
            # `mass * g`, and the duty factors of a biped that has to alternate
            # its feet are each other's complement. A pair like that wastes one
            # of the archive's two dimensions.
            continue
        rx = padded_range(by_name[x], args.pad_fraction)
        ry = padded_range(by_name[y], args.pad_fraction)
        mx = np.asarray(by_name[x]["gait_medians"])
        my = np.asarray(by_name[y]["gait_medians"])
        cells = _cells(mx, my, rx, ry, args.grid_dims)
        # The number that actually predicts archive coverage: how many distinct
        # cells the *feasible mutation probe* lands in. Cell separation of the
        # eleven known gaits says the axes can tell real gaits apart; this says
        # the search can reach cells to put them in.
        probe = result.get("_probe_values") or {}
        probe_cells = 0
        if x in probe and y in probe:
            px, py = probe[x], probe[y]
            finite = np.isfinite(px) & np.isfinite(py)
            if finite.any():
                probe_cells = int(
                    len(np.unique(_cells(px[finite], py[finite], rx, ry, args.grid_dims)))
                )
        # Two axes that say the same thing waste a grid dimension; the
        # correlation is reported so a highly redundant pair can be rejected
        # even when it happens to score well on cell count.
        corr = float(np.corrcoef(mx, my)[0, 1]) if len(mx) > 2 else float("nan")
        pairs.append(
            {
                "axes": [x, y],
                "ranges": [list(rx), list(ry)],
                "distinct_cells": int(len(np.unique(cells))),
                "probe_cells": probe_cells,
                "min_ratio": float(
                    min(by_name[x]["spread_to_noise"], by_name[y]["spread_to_noise"])
                ),
                "abs_correlation": abs(corr),
                "abs_corr_with_fitness": [
                    abs(by_name[x]["corr_with_displacement"]),
                    abs(by_name[y]["corr_with_displacement"]),
                ],
                "mutation_reach_sds": [
                    by_name[x].get("mutation_reach_sds", float("nan")),
                    by_name[y].get("mutation_reach_sds", float("nan")),
                ],
            }
        )
    pairs.sort(
        key=lambda p: (
            -p["probe_cells"],
            -p["distinct_cells"],
            p["abs_correlation"],
        )
    )
    return pairs


def to_markdown(result: dict, pairs: list[dict], args: Args) -> str:
    lines = [
        "| axis | between-gait spread | replica sd | sd / range | spread / sd "
        "| \\|corr\\| with displacement | mutation reach (in replica sds) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [r for r in result["axes"] if r.get("supplied")]
    rows.sort(key=lambda r: -r["spread_to_noise"])
    for r in rows:
        mark = (
            "**"
            if r["noise_fraction_of_range"] < args.max_noise_fraction
            and r["spread_to_noise"] >= args.min_ratio
            else ""
        )
        reach = r.get("mutation_reach_sds")
        lines.append(
            f"| {mark}{r['label']}{mark} | {r['between_gait_spread']:.4g} "
            f"| {r['replica_sd']:.4g} | {r['noise_fraction_of_range'] * 100:.1f}% "
            f"| {r['spread_to_noise']:.1f} "
            f"| {abs(r['corr_with_displacement']):.2f} "
            + (f"| {reach:.1f} |" if reach is not None else "| — |")
        )
    lines += [
        "",
        "| axis pair | cells reached by feasible mutants | distinct cells (of 11 gaits) "
        "| \\|corr\\| between axes | worst spread/sd | worst mutation reach "
        "| max \\|corr\\| with displacement |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in pairs[:12]:
        lines.append(
            f"| {p['axes'][0]} x {p['axes'][1]} | {p['probe_cells']} "
            f"| {p['distinct_cells']} "
            f"| {p['abs_correlation']:.2f} | {p['min_ratio']:.1f} "
            f"| {min(p['mutation_reach_sds']):.1f} "
            f"| {max(p['abs_corr_with_fitness']):.2f} |"
        )
    return "\n".join(lines)


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    result = measure(args)
    values = result.pop("_values")
    del values
    if args.mutation_probe:
        result["mutation_probe"] = probe_mutation_reach(args, result)
    pairs = rank_pairs(result, args)
    result.pop("_probe_values", None)
    result["pairs"] = pairs
    result["thresholds"] = {
        "min_ratio": args.min_ratio,
        "max_noise_fraction": args.max_noise_fraction,
        "pad_fraction": args.pad_fraction,
        "grid_dims": list(args.grid_dims),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", result)
    md = to_markdown(result, pairs, args)
    (out / "selection.md").write_text(md + "\n")

    print(f"\nmeasurement set: {len(result['labels'])} gaits, "
          f"{result['replicas']} replicas each")
    for label, surv, disp in zip(
        result["labels"], result["survival_rate"], result["median_displacement_m"]
    ):
        print(f"  {label:<28} survival {surv:5.1%}  median {disp:+.3f} m")
    print()
    print(md)
    print(f"\nwrote {out}/selection.json and {out}/selection.md")

    if pairs:
        best = pairs[0]
        print(
            f"\nbest pair: {best['axes'][0]} x {best['axes'][1]}, "
            f"{best['probe_cells']} cells reached by feasible mutants, "
            f"{best['distinct_cells']} of {len(result['labels'])} gaits in "
            f"distinct cells, ranges "
            f"{[round(v, 4) for v in best['ranges'][0]]} x "
            f"{[round(v, 4) for v in best['ranges'][1]]}"
        )
    else:
        print("\nNO axis pair cleared the thresholds — report this, do not lower them.")


def descriptor_from_pair(pair: dict) -> DescriptorCfg:
    """Turn a ranked pair back into the cfg a run is launched with."""
    (x, y), (rx, ry) = pair["axes"], pair["ranges"]
    return DescriptorCfg(
        axis_x=x, axis_y=y, x_range=tuple(rx), y_range=tuple(ry)
    )


if __name__ == "__main__":
    main()
