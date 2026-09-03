"""Keep the seeds that clear the gate, and say what was dropped.

``qd.seed_crawl`` distils every teacher it is given and scores each one; only
some come out P2'-viable. Handing all of them to the archive wastes the jitter
budget, which is split evenly across the seeds it is given: with one viable
seed in five, four fifths of the opening population is a neighbourhood around
a genome that cannot be inserted at all.

This filters on the measured per-replica viability and writes the survivors,
with the dropped ones named in the JSON rather than silently absent.

    uv run python -m qd.select_seeds --seeds logs/qd/seeds/crawl_seeds.npz \\
        --min-viable 0.5 --out logs/qd/seeds/crawl_seeds_viable.npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from qd.common import write_json


@dataclass
class Args:
    seeds: Path
    out: Path
    min_viable: float = 0.5
    """Per-replica P2' viability a seed must reach to be kept.

    0.5 rather than the gate's 5/8 = 0.625: a seed only has to be *insertable*,
    and the archive re-tests incumbents anyway. A seed at 0.5 is admitted by
    the 5-of-8 gate about 36 % of the time, which across a 240-strong jitter
    family is plenty to open a sub-archive."""

    report: Path | None = None
    """Defaults to the seed file's ``.json`` sibling, as ``qd.seed_crawl`` writes."""


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    with np.load(args.seeds, allow_pickle=False) as f:
        genomes = np.asarray(f["genome"])
        sources = [str(x) for x in f["sources"]] if "sources" in f else [
            str(i) for i in range(len(genomes))
        ]
    report_path = args.report or args.seeds.with_suffix(".json")
    scored = json.loads(Path(report_path).read_text())
    by_source = {r["source"]: r for r in scored}

    keep, kept, dropped = [], [], []
    for i, src in enumerate(sources):
        row = by_source.get(src, {})
        v = float(row.get("per_replica_viable", 0.0))
        entry = {
            "source": src,
            "per_replica_viable": v,
            "median_displacement_m": row.get("median_displacement_m"),
            "label": row.get("label"),
        }
        if v >= args.min_viable:
            keep.append(i)
            kept.append(entry)
        else:
            dropped.append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        genome=genomes[keep] if keep else genomes[:0],
        sources=np.array([sources[i] for i in keep]),
    )
    summary = {
        "from": str(args.seeds),
        "min_viable": args.min_viable,
        "kept": kept,
        "dropped": dropped,
    }
    write_json(args.out.with_suffix(".json"), summary)

    print(f"kept {len(kept)} of {len(sources)} seeds (min viable {args.min_viable})")
    for e in kept:
        print(
            f"   KEEP {e['source']:14s} viable {e['per_replica_viable']:.3f}  "
            f"{e['median_displacement_m']:+.3f} m  {e['label']}"
        )
    for e in dropped:
        print(
            f"   drop {e['source']:14s} viable {e['per_replica_viable']:.3f}  "
            f"{e['median_displacement_m']:+.3f} m  {e['label']}"
        )
    if not keep:
        print(
            "\nNo seed cleared the bar. That is a finding, not a file to fix: "
            "the mode has no starting population and its sub-archive will be "
            "empty by construction."
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
