"""One 20x20 archive per mode, plus the two things a single archive cannot do.

v3's archive is a known good thing — 268 verified elites in 109 resolvable
cells on (trunk height x limb speed) — and a single cross-mode grid throws it
away. Trunk height moves by 0.03 m between modes against 5 mm *within*
walking, so binning all modes on one grid puts every walker in one cell: the v2
duty-factor failure in a new costume. Under the hierarchy the walk sub-archive
keeps v3's axes and ranges byte-for-byte, and each new mode runs its own
resolvable-resolution check instead of inheriting the noisiest mode's.

Two mechanisms live here because they only make sense across sub-archives:

**Per-mode parent budget.** With one shared parent pool, 300 walkers supply
95 % of the offspring while crawl has five elites and never gets varied. The
budget gives every non-empty mode an equal share, so a young mode is not
out-voted by a mature one.

**Incumbent re-testing** (v3's named-but-unbuilt fix). v3 measured the winner's
curse on the *survival* predicate, not only on fitness: archive robustness
decayed 83 % -> 75 % as the search ran, because an elite is in the archive for
having passed unanimously **once** out of ~51,000 attempts, and a genuinely
88 %-robust genome passes unanimous-8 with probability 0.36. Raising N raises
the exponent and never removes the selection. Re-testing a random tenth of the
archive each iteration converts survival from a one-shot maximum into a running
average — precisely what taking the median did for fitness — and an elite whose
running pass rate falls below the gate is evicted.

Nothing here imports mjlab; ``ribs`` and numpy only, so the insertion rule and
the eviction rule are unit-testable on CPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from qd.common import DEFAULT_GRID_DIMS, archive_stats, make_archive, save_archive
from qd.descriptors import DescriptorCfg
from qd.modes import MODES


@dataclass(frozen=True)
class ModeArchiveCfg:
    """One mode's grid: which axes, over what range, at what resolution."""

    descriptor: DescriptorCfg
    grid_dims: tuple[int, int] = DEFAULT_GRID_DIMS
    note: str = ""


V3_WALK_DESCRIPTOR = DescriptorCfg(
    axis_x="torso_height_mean",
    x_range=(0.11667, 0.12184),
    axis_y="joint_speed",
    y_range=(0.89228, 2.50831),
)
"""v3's measured pair and ranges, unchanged. The walk sub-archive is a
continuation of that archive, not a new one, and keeping the geometry identical
is what makes "walk did not regress" a comparison rather than an assertion."""


def default_mode_cfgs(
    crawl_height_range: tuple[float, float] = (0.02, 0.09),
    crawl_speed_range: tuple[float, float] = (0.5, 4.0),
) -> dict[str, ModeArchiveCfg]:
    """Starting geometry for every mode.

    Walk gets v3's measured axes. The others get **trunk height x limb speed
    re-ranged**, which is the design's default-until-measured pair: it separated
    six teacher gaits 6-of-6 within walking, and the expectation — to be
    checked, not assumed — is that it also separates crawls (belly-drag vs
    knee-crawl differ by ~2 cm of trunk height; slow vs fast limbs by the rest).
    "Other" is binned on trunk height x rotation rate instead, because the one
    thing known about an unclassified behaviour is that it is not a walk.

    The crawl ranges are *provisional* until a crawl sub-archive has five
    positive probes of its own to run Stage A on; they are set from the measured
    rest heights (prone settles at 0.035 m, supine 0.048 m) padded outward.
    """
    height_x_speed = lambda lo_h, hi_h, lo_s, hi_s: DescriptorCfg(
        axis_x="torso_height_mean",
        x_range=(lo_h, hi_h),
        axis_y="joint_speed",
        y_range=(lo_s, hi_s),
    )
    return {
        "walk": ModeArchiveCfg(V3_WALK_DESCRIPTOR, note="v3's measured pair, unchanged"),
        "crawl": ModeArchiveCfg(
            height_x_speed(*crawl_height_range, *crawl_speed_range),
            note="provisional: trunk height x limb speed, re-ranged from the "
            "measured prone/supine rest heights",
        ),
        "roll": ModeArchiveCfg(
            height_x_speed(0.02, 0.12, 0.5, 6.0),
            note="provisional; a roll spans every height as it goes over",
        ),
        "hop": ModeArchiveCfg(
            height_x_speed(0.05, 0.13, 0.5, 6.0),
            note="no seed (Appendix A); the grid exists so a bounding gait the "
            "search stumbles on is filed rather than lost",
        ),
        "other": ModeArchiveCfg(
            DescriptorCfg(
                axis_x="torso_height_mean",
                x_range=(0.02, 0.13),
                axis_y="yaw_rate",
                y_range=(0.0, 3.0),
            ),
            note="generic pair until it holds enough elites for its own Stage A",
        ),
    }


def _fingerprint(genome: np.ndarray) -> str:
    """Identity of the genome occupying a cell, so a re-test follows the elite.

    A cell's occupant changes whenever something better is inserted, and the
    running pass count belongs to the *elite*, not to the cell. Hashing the
    genome is the cheapest way to notice the swap without threading an id
    through pyribs, which has nowhere to put one.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(genome, dtype=np.float32).tobytes(), digest_size=16
    ).hexdigest()


@dataclass
class PassRecord:
    """An elite's running gate record, accumulated across re-tests."""

    fingerprint: str
    passes: int = 0
    attempts: int = 0

    @property
    def rate(self) -> float:
        return self.passes / self.attempts if self.attempts else 1.0


@dataclass
class RetestOutcome:
    tested: int = 0
    evicted: int = 0
    evicted_by_mode: dict[str, int] = field(default_factory=dict)
    pass_rate: float = float("nan")


class ModeArchives:
    """``dict[mode, GridArchive]`` with a parent budget and incumbent re-testing."""

    def __init__(
        self,
        solution_dim: int,
        cfgs: dict[str, ModeArchiveCfg] | None = None,
        qd_score_offset: float = -5.0,
        seed: int | None = None,
    ):
        self.cfgs = cfgs or default_mode_cfgs()
        unknown = set(self.cfgs) - set(MODES)
        if unknown:
            raise ValueError(f"archive configured for unknown modes: {sorted(unknown)}")
        self.solution_dim = solution_dim
        self.archives = {
            mode: make_archive(
                solution_dim=solution_dim,
                grid_dims=cfg.grid_dims,
                measure_ranges=cfg.descriptor.ranges,
                qd_score_offset=qd_score_offset,
                seed=seed,
            )
            for mode, cfg in self.cfgs.items()
        }
        self.records: dict[str, dict[int, PassRecord]] = {m: {} for m in self.cfgs}

    # -- insertion ---------------------------------------------------------- #

    def add(
        self,
        mode: str,
        genomes: np.ndarray,
        fitness: np.ndarray,
        axes: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Offer solutions to one mode's archive; returns pyribs' status array.

        Measures are computed from that mode's *own* descriptor, so a crawl is
        binned on the crawl grid's axes even though the caller measured every
        candidate axis once."""
        if mode not in self.archives:
            raise KeyError(f"no sub-archive for mode {mode!r}")
        if len(genomes) == 0:
            return np.zeros(0, dtype=int)
        measures = self.cfgs[mode].descriptor.measures(axes)
        status = self.archives[mode].add(
            np.asarray(genomes), np.asarray(fitness), measures
        )["status"]
        self._sync_records(mode)
        return np.asarray(status)

    def _sync_records(self, mode: str) -> None:
        """Drop records whose cell is empty or has a new occupant."""
        archive = self.archives[mode]
        data = archive.data(return_type="dict")
        live: dict[int, str] = {
            int(idx): _fingerprint(sol)
            for idx, sol in zip(data["index"], data["solution"])
        }
        records = self.records[mode]
        for cell in list(records):
            if cell not in live or records[cell].fingerprint != live[cell]:
                records.pop(cell)
        for cell, fp in live.items():
            records.setdefault(cell, PassRecord(fingerprint=fp))

    # -- parents ------------------------------------------------------------ #

    def sample_parents(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, solution_dim)`` parents, budgeted equally across non-empty modes.

        Without the budget a mature mode owns the offspring: 300 walkers against
        five crawls means 98 % of variation is spent refining walking, and the
        young mode never gets the on-policy data it needs to grow. Remainders go
        to modes in ``MODES`` order rather than to the largest, so the rounding
        does not quietly restore the imbalance.
        """
        live = [m for m in MODES if m in self.archives and len(self.archives[m]) > 0]
        if not live or n <= 0:
            return np.zeros((0, self.solution_dim))
        share, extra = divmod(n, len(live))
        out = []
        for i, mode in enumerate(live):
            k = share + (1 if i < extra else 0)
            if k == 0:
                continue
            sols = self.archives[mode].data("solution")
            pick = rng.integers(0, len(sols), size=k)
            out.append(np.asarray(sols)[pick])
        return np.concatenate(out) if out else np.zeros((0, self.solution_dim))

    def parent_budget(self, n: int) -> dict[str, int]:
        """What :meth:`sample_parents` would spend per mode — for the log."""
        live = [m for m in MODES if m in self.archives and len(self.archives[m]) > 0]
        if not live or n <= 0:
            return {}
        share, extra = divmod(n, len(live))
        return {m: share + (1 if i < extra else 0) for i, m in enumerate(live)}

    # -- incumbent re-testing ----------------------------------------------- #

    def sample_incumbents(
        self, fraction: float, rng: np.random.Generator
    ) -> list[tuple[str, int, np.ndarray]]:
        """A random ``fraction`` of the archive, as ``(mode, cell, genome)``.

        Sampled across all modes together, so the re-test budget follows where
        the elites actually are rather than taxing a five-elite mode as heavily
        as a three-hundred-elite one."""
        pool: list[tuple[str, int, np.ndarray]] = []
        for mode, archive in self.archives.items():
            data = archive.data(return_type="dict")
            for idx, sol in zip(data["index"], data["solution"]):
                pool.append((mode, int(idx), np.asarray(sol)))
        if not pool:
            return []
        k = max(1, round(fraction * len(pool)))
        pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in pick]

    def record_retest(
        self, results: list[tuple[str, int, np.ndarray, bool]], min_pass_rate: float
    ) -> RetestOutcome:
        """Fold re-test verdicts into the running pass rates and evict failures.

        An elite is evicted when its running rate drops **below** the gate it
        was admitted under — not on a single failure. One failed re-test of a
        95 %-robust elite is expected; a rate that settles under the bar is the
        winner's curse being undone.
        """
        outcome = RetestOutcome(tested=len(results))
        evict: dict[str, set[int]] = {m: set() for m in self.archives}
        passes = 0
        for mode, cell, genome, passed in results:
            record = self.records[mode].get(cell)
            fp = _fingerprint(genome)
            if record is None or record.fingerprint != fp:
                continue  # the cell changed hands while the re-test was in flight
            record.attempts += 1
            record.passes += int(passed)
            passes += int(passed)
            if record.rate < min_pass_rate:
                evict[mode].add(cell)
        outcome.pass_rate = passes / len(results) if results else float("nan")
        for mode, cells in evict.items():
            if cells:
                self._evict(mode, cells)
                outcome.evicted += len(cells)
                outcome.evicted_by_mode[mode] = len(cells)
        return outcome

    def _evict(self, mode: str, cells: set[int]) -> None:
        """Rebuild one sub-archive without the named cells.

        pyribs has no removal, so the archive is cleared and refilled from the
        survivors. At 400 cells that is a numpy round-trip per iteration, which
        is nothing against the eight rollouts an offspring batch costs."""
        archive = self.archives[mode]
        data = archive.data(return_type="dict")
        keep = np.array([int(i) not in cells for i in data["index"]], dtype=bool)
        kept_records = {
            int(i): self.records[mode][int(i)]
            for i, k in zip(data["index"], keep)
            if k and int(i) in self.records[mode]
        }
        archive.clear()
        if keep.any():
            archive.add(
                np.asarray(data["solution"])[keep],
                np.asarray(data["objective"])[keep],
                np.asarray(data["measures"])[keep],
            )
        self.records[mode] = kept_records

    def pass_rates(self) -> dict[str, float]:
        """Mean running pass rate per mode, over elites that have been re-tested."""
        out = {}
        for mode, records in self.records.items():
            tested = [r for r in records.values() if r.attempts]
            out[mode] = float(np.mean([r.rate for r in tested])) if tested else float("nan")
        return out

    # -- reporting ---------------------------------------------------------- #

    def __len__(self) -> int:
        return sum(len(a) for a in self.archives.values())

    def stats(self) -> dict[str, dict]:
        return {mode: archive_stats(a) for mode, a in self.archives.items()}

    def occupancy(self) -> dict[str, int]:
        return {mode: len(a) for mode, a in self.archives.items()}

    def save(self, out_dir, meta: dict | None = None) -> dict[str, str]:
        """One ``.npz`` per mode, each carrying its own descriptor in ``meta``.

        Separate files rather than one bundle so every existing v1-v3 tool —
        ``verify_archive``, ``render_gaits``, the viewer — reads a v4 sub-archive
        without knowing v4 exists."""
        from pathlib import Path

        out_dir = Path(out_dir)
        paths = {}
        for mode, archive in self.archives.items():
            payload = dict(meta or {})
            payload["mode"] = mode
            payload["mode_note"] = self.cfgs[mode].note
            payload.update(self.cfgs[mode].descriptor.to_meta())
            payload["retest"] = {
                str(cell): {"passes": r.passes, "attempts": r.attempts}
                for cell, r in self.records[mode].items()
            }
            paths[mode] = str(
                save_archive(archive, out_dir / f"archive_{mode}.npz", payload)
            )
        return paths
