"""Build the interactive archive-gait viewer from rendered clips.

Takes one or more manifests written by :mod:`qd.render_gaits` and emits a
self-contained HTML page: a clickable 20x20 archive heatmap where selecting a
filled cell plays that elite's gait next to its fitness, duty factors and
survival time.

    uv run python -m qd.build_viewer \\
        --manifests logs/qd/gaits/cpg/manifest.json \\
        --labels "MAP-Elites (CPG)" \\
        --out logs/qd/viewer/index.html

Pass two manifests to get an archive switcher (Phase 2 vs Phase 3).

**Why only some cells carry a clip.** Everything is inlined as a data: URI
because a published artifact cannot fetch external media, and the page has a
hard 16 MB ceiling. ``--budget-mb`` (or one ``--budgets-mb`` per manifest)
fills that ceiling deliberately: the top few elites first (``--top-first``, so
a tab squeezed to a sliver still plays its best gaits), then every elite that
survived the full episode (those are the rarest and most informative gaits,
and they are *not* the highest-scoring ones), then the top elites by fitness,
then a spatial sweep across the descriptor space so every region of the
archive is represented. Cells without an embedded clip stay clickable and show
their stats plus the path to the full-resolution file on disk, which
``qd.render_gaits`` wrote for *every* filled cell.

**CVT (v5) manifests.** A ``kind: "cvt"`` manifest has no grid: its cells are
the Voronoi regions of a centroid set in a learned 4-d latent space. The page
projects the centroids onto their first two principal components (computed
here, from the centroid set alone; the explained variance is stated in the
tab's footnote) and draws every cell as a point on that plane — filled ones
coloured by verified median fitness and clickable exactly like grid cells,
arrow keys stepping to the nearest filled neighbour in the arrow's direction.
Clip selection for such a tab is per posture (``--top-per-posture`` best walkers
AND best crawlers) followed by a farthest-point sweep in the full 4-d latent,
so the crawl continuum is sampled along its length rather than at its peak.
Grid tabs are untouched by any of this.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Args:
    manifests: tuple[Path, ...]
    labels: tuple[str, ...] = ()
    """Display name per manifest; defaults to the genome kind."""

    out: Path = Path("logs/qd/viewer/index.html")

    budget_mb: float = 9.0
    """Raw clip bytes to embed per archive. base64 inflates this by ~33%, so
    9 MB of clips is ~12 MB of page — under the 16 MB artifact ceiling."""

    budgets_mb: tuple[float, ...] = ()
    """Per-manifest override of ``budget_mb`` (one entry per manifest)."""

    top: int = 16
    """Highest-fitness elites embedded before the spatial sweep begins."""

    top_first: int = 3
    """Highest-fitness elites embedded before anything else, survivors included."""

    top_per_posture: int = 4
    """CVT tabs: best elites of EACH posture embedded before the latent sweep."""


def _fill(chosen: list[dict], used: float, candidates, budget_bytes: float) -> float:
    """Append candidates in order while they fit; returns the bytes used."""
    for entry in candidates:
        if entry in chosen:
            continue
        if used + entry["bytes"] > budget_bytes:
            break
        chosen.append(entry)
        used += entry["bytes"]
    return used


def _sweep(chosen: list[dict], used: float, entries: list[dict], budget_bytes: float, key) -> None:
    """Farthest-point pass: keep adding the elite furthest (under ``key``) from
    everything already chosen, so coverage fills in evenly instead of
    clustering near the peak."""
    remaining = [e for e in entries if e not in chosen]
    while remaining and used < budget_bytes:
        picked = np.array([key(e) for e in chosen], dtype=float)
        pts = np.array([key(e) for e in remaining], dtype=float)
        if len(picked) == 0:
            idx = 0
        else:
            dist = np.linalg.norm(pts[:, None, :] - picked[None, :, :], axis=-1)
            idx = int(np.argmax(dist.min(axis=1)))
        entry = remaining.pop(idx)
        if used + entry["bytes"] <= budget_bytes:
            chosen.append(entry)
            used += entry["bytes"]


def _select_clips(
    entries: list[dict], budget_bytes: float, top: int, top_first: int = 0
) -> set[int]:
    """Rows to embed: the top few, survivors, then the best elites, then a spatial spread.

    Survivors come before the fitness ranking. On this robot almost nothing
    stays upright for the whole episode, so those clips are the most
    informative ones in the archive — and they are not the highest-scoring, so
    a purely fitness-ranked selection drops exactly the gaits a reader most
    needs to see. ``top_first`` best elites go ahead of even the survivors, so
    a tab given a sliver of the page still plays the gaits its "best" list
    points at. The spatial pass then walks cells in order of distance from the
    already-chosen set.
    """
    chosen: list[dict] = []
    by_fitness = sorted(entries, key=lambda e: -e["archived_fitness"])
    used = _fill(chosen, 0.0, by_fitness[:top_first], budget_bytes)
    used = _fill(
        chosen,
        used,
        sorted((e for e in entries if e.get("survived")), key=lambda e: -e["displacement_m"]),
        budget_bytes,
    )
    used = _fill(chosen, used, by_fitness[:top], budget_bytes)
    _sweep(chosen, used, entries, budget_bytes, key=lambda e: e["cell"])
    return {e["row"] for e in chosen}


def _select_clips_cvt(entries: list[dict], budget_bytes: float, top_per_posture: int) -> set[int]:
    """Rows to embed on a CVT tab: the best of EACH posture, then a latent sweep.

    Survivors-first would spend the whole budget on walkers (every walker
    "survives" under the v1-v3 upright rule, no crawler does), so the ranking
    is per posture instead, and the sweep runs in the archive's own 4-d latent
    rather than on the projected plane — the plane folds two dimensions away.
    """
    chosen: list[dict] = []
    used = 0.0
    for posture in sorted({e["posture"] for e in entries}):
        ranked = sorted(
            (e for e in entries if e["posture"] == posture),
            key=lambda e: -e["archived_fitness"],
        )
        used = _fill(chosen, used, ranked[:top_per_posture], budget_bytes)
    _sweep(chosen, used, entries, budget_bytes, key=lambda e: e["latent"])
    return {e["row"] for e in chosen}


def _project(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """PCA of the centroid set: ``(mean, components (2, D), explained ratio per PC)``.

    Fit on the centroids, not the elites: the plane is a property of the
    archive's geometry, and an elite-fitted plane would rotate every time the
    crawl continuum grew.
    """
    mean = centroids.mean(axis=0)
    _u, s, vt = np.linalg.svd(centroids - mean, full_matrices=False)
    explained = (s**2 / (s**2).sum()).tolist()
    return mean, vt[:2], explained


def _load(manifest_path: Path, label: str, budget_bytes: float, args: Args) -> dict:
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    entries = manifest["elites"]
    kind = manifest.get("kind", "grid")
    if kind == "cvt":
        embed = _select_clips_cvt(entries, budget_bytes, args.top_per_posture)
    else:
        embed = _select_clips(entries, budget_bytes, args.top, args.top_first)

    out_entries = []
    embedded_bytes = 0
    for e in entries:
        item = {k: e[k] for k in e if k != "clip"}
        item["path"] = str((base / e["clip"]).resolve())
        if e["row"] in embed:
            raw = (base / e["clip"]).read_bytes()
            item["src"] = "data:video/mp4;base64," + base64.b64encode(raw).decode()
            embedded_bytes += len(raw)
        out_entries.append(item)

    objective = np.array([e["archived_fitness"] for e in entries])
    dims = manifest["grid_dims"]
    out = {
        "label": label,
        "genome": manifest["genome"],
        "kind": kind,
        "grid": dims,
        # Absent on a v1/v2 manifest; the page falls back to duty factor, which
        # is what those archives were binned on.
        "descriptor": manifest.get("descriptor"),
        "verified": bool(manifest.get("verified", False)),
        "fps": manifest["fps"],
        "episodeSeconds": manifest["episode_seconds"],
        "elites": out_entries,
        "stats": {
            "elites": len(entries),
            "coverage": len(entries) / (dims[0] * dims[1]) if dims else None,
            "best": float(objective.max()),
            "mean": float(objective.mean()),
            "positive": int((objective > 0).sum()),
            "survivors": int(sum(1 for e in entries if e["survived"])),
            "maxUpright": float(max(e["upright_s"] for e in entries)),
            "maxDisplacement": float(max(e["displacement_m"] for e in entries)),
            "embedded": len(embed),
            "embeddedMb": embedded_bytes / 1e6,
        },
    }
    if kind == "cvt":
        centroids = np.asarray(manifest["centroids"], dtype=np.float64)
        mean, components, explained = _project(centroids)
        plane = (centroids - mean) @ components.T
        postures = sorted({e["posture"] for e in entries})
        for item in out_entries:
            # Round the per-elite floats: 970 entries of 17-digit latents are
            # page bytes that could have been clips.
            for k in ("latent", "eval_latent"):
                item[k] = [round(v, 3) for v in item[k]]
            item["xy"] = [round(float(v), 3) for v in plane[item["cell"][0]]]
        out["centroids2d"] = [[round(float(v), 3) for v in p] for p in plane]
        out["projection"] = {
            "method": "PCA of the centroid set",
            "explained": explained,
            "latentDim": int(centroids.shape[1]),
        }
        out["postures"] = postures
        out["replicas"] = int(manifest.get("replicas", 8))
        out["stats"]["coverage"] = len(entries) / len(centroids)
        out["stats"]["cells"] = len(centroids)
        out["stats"]["postureCounts"] = {
            p: int(sum(1 for e in entries if e["posture"] == p)) for p in postures
        }
    return out


PAGE = """<title>Microduck Gait Archive</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #EDF0F5;
  --surface: #FFFFFF;
  --surface-2: #F5F7FA;
  --line: #D6DCE6;
  --ink: #101725;
  --ink-2: #47536A;
  --ink-3: #7B879B;
  --neg: #24799C;
  --zero: #8B93A2;
  --pos: #B37D14;
  --empty: #E2E6ED;
  --ring: #101725;
  --shadow: 0 1px 2px rgba(16,23,37,.06), 0 8px 24px rgba(16,23,37,.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0B1220;
    --surface: #131C2E;
    --surface-2: #18233A;
    --line: #26324B;
    --ink: #E8EDF5;
    --ink-2: #A7B3C7;
    --ink-3: #76829A;
    --neg: #3AA0CC;
    --zero: #6B7486;
    --pos: #E5AC2E;
    --empty: #1A2438;
    --ring: #F5F8FF;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground: #0B1220;
  --surface: #131C2E;
  --surface-2: #18233A;
  --line: #26324B;
  --ink: #E8EDF5;
  --ink-2: #A7B3C7;
  --ink-3: #76829A;
  --neg: #3AA0CC;
  --zero: #6B7486;
  --pos: #E5AC2E;
  --empty: #1A2438;
  --ring: #F5F8FF;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: Archivo, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }

header { display: flex; flex-direction: column; gap: 6px; margin-bottom: 26px; }
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-3);
}
h1 { font-size: 30px; font-weight: 700; letter-spacing: -.02em; margin: 0; text-wrap: balance; }
.lede { color: var(--ink-2); max-width: 68ch; margin: 4px 0 0; }

.switch { display: flex; gap: 8px; margin: 22px 0 18px; flex-wrap: wrap; }
.switch button {
  font: 500 13px/1 Archivo, sans-serif;
  padding: 9px 15px; border-radius: 7px; cursor: pointer;
  background: var(--surface); color: var(--ink-2);
  border: 1px solid var(--line);
}
.switch button[aria-pressed="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
.switch button:focus-visible, .cell:focus-visible, .top-item:focus-visible {
  outline: 2px solid var(--ring); outline-offset: 2px;
}

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; margin-bottom: 26px; }
.stat { background: var(--surface); padding: 13px 15px; }
.stat dt { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 5px; }
.stat dd { margin: 0; font-family: "IBM Plex Mono", monospace; font-size: 20px; font-weight: 600;
  font-variant-numeric: tabular-nums; letter-spacing: -.01em; }

.panel { display: grid; grid-template-columns: minmax(0, 1fr) 372px; gap: 26px; align-items: start; }
@media (max-width: 900px) { .panel { grid-template-columns: 1fr; } }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px; box-shadow: var(--shadow); }
.card h2 { font-size: 13px; font-weight: 600; margin: 0 0 3px; letter-spacing: -.005em; }
.card .hint { font-size: 12.5px; color: var(--ink-3); margin: 0 0 14px; }

.grid-frame { display: grid; grid-template-columns: 30px 1fr; grid-template-rows: 1fr 30px; gap: 7px; }
.ylab, .xlab { font-family: "IBM Plex Mono", monospace; font-size: 10px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--ink-3); display: flex; align-items: center; justify-content: center; }
.ylab { writing-mode: vertical-rl; transform: rotate(180deg); }
.grid { display: grid; gap: 2px; aspect-ratio: 1; }
.cell { border: 0; padding: 0; border-radius: 2px; cursor: pointer; position: relative;
  background: var(--empty); transition: transform .08s ease; }
.cell[data-filled="0"] { cursor: default; }
.cell[data-clip="1"]::after {
  content: ""; position: absolute; inset: auto 2px 2px auto; width: 3px; height: 3px;
  border-radius: 50%; background: var(--ring); opacity: .75;
}
.cell[aria-pressed="true"] { outline: 2px solid var(--ring); outline-offset: 1px; z-index: 2; transform: scale(1.12); }
.cell:hover[data-filled="1"] { transform: scale(1.12); z-index: 1; }

/* CVT tabs: the archive is a point cloud on a projected plane, not a grid. */
.grid.map { display: block; aspect-ratio: auto; }
.grid.map svg { width: 100%; height: auto; display: block; border-radius: 6px; background: var(--surface-2); }
.pt { cursor: pointer; stroke: none; }
.pt-empty { fill: var(--empty); stroke: var(--line); stroke-width: .03; }
.pt[data-clip="1"] { stroke: var(--ring); stroke-width: .07; }
.pt:hover, .pt[aria-pressed="true"] { stroke: var(--ring); stroke-width: .12; }
.pt:focus-visible { outline: none; stroke: var(--ring); stroke-width: .12; }
.ring { width: 11px; height: 11px; border-radius: 50%; border: 2px solid var(--ring); opacity: .8; }

.scale { display: flex; align-items: center; gap: 9px; margin-top: 15px;
  font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-3); }
.ramp { height: 8px; flex: 1; border-radius: 4px; }
.legend-note { display: flex; align-items: center; gap: 7px; margin-top: 9px;
  font-size: 11.5px; color: var(--ink-3); }
.swatch { width: 11px; height: 11px; border-radius: 2px; background: var(--empty); border: 1px solid var(--line); }
.dot { width: 4px; height: 4px; border-radius: 50%; background: var(--ring); opacity: .75; }

video { width: 100%; border-radius: 8px; display: block; background: var(--surface-2); aspect-ratio: 4/3; }
.noclip { width: 100%; aspect-ratio: 4/3; border-radius: 8px; background: var(--surface-2);
  border: 1px dashed var(--line); display: flex; align-items: center; justify-content: center;
  text-align: center; padding: 22px; color: var(--ink-3); font-size: 12.5px; }
.noclip code { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; word-break: break-all; color: var(--ink-2); }

.readout { margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
  background: var(--line); border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
.readout div { background: var(--surface); padding: 9px 11px; }
.readout dt { font-family: "IBM Plex Mono", monospace; font-size: 10px; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 3px; }
.readout dd { margin: 0; font-family: "IBM Plex Mono", monospace; font-size: 14.5px;
  font-weight: 500; font-variant-numeric: tabular-nums; }
.verdict { font-family: Archivo, sans-serif; font-size: 12.5px; font-weight: 500;
  padding: 3px 9px; border-radius: 20px; display: inline-block; }
.verdict[data-ok="1"] { background: color-mix(in srgb, var(--pos) 22%, transparent); color: var(--ink); }
.verdict[data-ok="0"] { background: color-mix(in srgb, var(--neg) 20%, transparent); color: var(--ink); }

.top-list { margin-top: 24px; }
.top-item { display: grid; grid-template-columns: 22px 1fr auto; gap: 11px; align-items: center;
  width: 100%; text-align: left; background: none; border: 0; border-top: 1px solid var(--line);
  padding: 8px 2px; cursor: pointer; color: var(--ink); font: inherit; }
.top-item:hover { background: var(--surface-2); }
.top-item .rank { font-family: "IBM Plex Mono", monospace; font-size: 11px; color: var(--ink-3); }
.top-item .meta { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--ink-3); }
.top-item .val { font-family: "IBM Plex Mono", monospace; font-size: 13px; font-variant-numeric: tabular-nums; }

footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--line);
  color: var(--ink-3); font-size: 12.5px; max-width: 74ch; }
footer p { margin: 0 0 8px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <span class="eyebrow">Quality-Diversity · MuJoCo Warp · Microduck</span>
    <h1>Microduck Gait Archive</h1>
    <p class="lede">Every cell is a different <em>way of walking</em>. MAP-Elites keeps the best gait
    found for each combination of the two behaviour axes named under the grid — so this is a map of
    behaviours, not a single best policy. The axes differ between archives: v1 and v2 are indexed by
    how much of the episode each foot spent on the ground, v3 by axes chosen because that one barely
    moved, v4 keeps one grid per posture, and v5 has no grid at all — its cells live in a learned 4-d
    behaviour space, drawn here on that space's two principal axes. Pick a cell to watch it.</p>
  </header>

  <div class="switch" id="switch" role="group" aria-label="Archive"></div>

  <dl class="stats" id="stats"></dl>

  <div class="panel">
    <section class="card">
      <h2>Behaviour archive</h2>
      <p class="hint" id="gridHint">Colour is fitness: forward metres, minus a penalty for time spent fallen.
        Click a cell, or focus the grid and use the arrow keys.</p>
      <div class="grid-frame">
        <div class="ylab" id="ylab">Right foot duty factor →</div>
        <div class="grid" id="grid" role="grid" aria-label="Archive cells"></div>
        <div></div>
        <div class="xlab" id="xlab">Left foot duty factor →</div>
      </div>
      <div class="scale">
        <span id="scaleMin">—</span>
        <div class="ramp" id="ramp"></div>
        <span id="scaleMax">—</span>
      </div>
      <div class="legend-note"><span class="swatch" id="emptySwatch"></span> <span id="emptyNote">never filled — no gait produced this contact pattern</span></div>
      <div class="legend-note"><span class="dot" id="clipMark"></span> clip embedded in this page</div>
    </section>

    <section class="card">
      <h2 id="selTitle">Select a cell</h2>
      <p class="hint" id="selHint">The gait plays here.</p>
      <div id="player"></div>
      <dl class="readout" id="readout"></dl>
      <div class="top-list">
        <h2>Best gaits</h2>
        <p class="hint">Ranked by archived fitness.</p>
        <div id="topList"></div>
      </div>
    </section>
  </div>

  <footer id="footnotes"></footer>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
let current = 0, selected = null;

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hex = (h) => [1,3,5].map(i => parseInt(h.slice(i,i+2),16));
const mix = (a,b,t) => `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(",")})`;

// Diverging ramp centred on zero fitness: fitness < 0 means the robot lost more
// to falling than it gained in ground covered, so zero is a real midpoint, not
// an arbitrary split. Cool pole = negative, neutral grey = zero, warm = positive.
function colorFor(v, lo, hi) {
  const neg = hex(css("--neg")), zero = hex(css("--zero")), pos = hex(css("--pos"));
  if (v >= 0) return mix(zero, pos, hi > 0 ? Math.min(1, v / hi) : 0);
  return mix(zero, neg, lo < 0 ? Math.min(1, v / lo) : 0);
}
const f3 = (v, unit = "") => (v >= 0 ? "+" : "") + v.toFixed(3) + unit;

const isCvt = (a) => a.kind === "cvt";
const keyOf = (e) => e.cell.join(",");
// A grid cell is named by its (row, col); a CVT cell by its centroid id.
const cellName = (e) => e.cell.length === 2 ? `cell (${e.cell[0]}, ${e.cell[1]})` : `cell ${String(e.cell[0]).padStart(4, "0")}`;
const pct = (v) => (v * 100).toFixed(0) + "%";

function renderStats(a) {
  const s = a.stats;
  const rows = [
    ["Elites", s.elites],
    ["Coverage", (s.coverage * 100).toFixed(1) + "%"],
    ["Best fitness", f3(s.best) + " m"],
    ["Positive fitness", s.positive],
    ["Furthest travelled", f3(s.maxDisplacement) + " m"],
    ["Longest upright", s.maxUpright.toFixed(2) + " s"],
    ["Survived " + a.episodeSeconds + " s", s.survivors],
  ];
  if (isCvt(a)) {
    rows.splice(1, 0, [a.postures.map(cap).join(" / "), a.postures.map(p => s.postureCounts[p]).join(" / ")]);
  }
  document.getElementById("stats").innerHTML = rows
    .map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

// A v1/v2 manifest predates per-archive axes and was binned on duty factor.
function descOf(a) {
  return a.descriptor || {
    axes: ["duty_left", "duty_right"],
    labels: ["left-foot duty factor", "right-foot duty factor"],
  };
}
function mx(e) { return e.measure_x !== undefined ? e.measure_x : e.duty_left; }
function my(e) { return e.measure_y !== undefined ? e.measure_y : e.duty_right; }
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function fmt(v) { return Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toPrecision(3); }
function vec(v) { return "[" + v.map(x => (x >= 0 ? "+" : "") + x.toFixed(1)).join(", ") + "]"; }

function renderAxes(a) {
  const nav = isCvt(a) ? "Click a point, or focus the map and use the arrow keys." : "Click a cell, or focus the grid and use the arrow keys.";
  document.getElementById("gridHint").textContent = a.verified
    ? `Colour is the elite's median forward distance over ${a.replicas || 8} fresh rollouts. The clip beside it is one of those rollouts, and this simulator's chaos means a single one can differ from the median by half a metre. ${nav}`
    : `Colour is fitness: forward metres, minus a penalty for time spent fallen. ${nav}`;
  if (isCvt(a)) {
    const ev = a.projection.explained;
    document.getElementById("xlab").textContent = `PC1 of the ${a.projection.latentDim}-d latent (${pct(ev[0])}) →`;
    document.getElementById("ylab").textContent = `PC2 (${pct(ev[1])}) →`;
    document.getElementById("emptyNote").textContent = "empty cell — no gait was filed at this centroid";
    document.getElementById("clipMark").className = "ring";
  } else {
    const d = descOf(a);
    document.getElementById("xlab").textContent = cap(d.labels[0]) + " →";
    document.getElementById("ylab").textContent = cap(d.labels[1]) + " →";
    document.getElementById("emptyNote").textContent = "never filled — no gait produced this contact pattern";
    document.getElementById("clipMark").className = "dot";
  }
}

function renderScale(lo, hi) {
  document.getElementById("ramp").style.background =
    `linear-gradient(90deg, ${colorFor(lo, lo, hi)}, ${colorFor(0, lo, hi)} ${(-lo/(hi-lo)*100).toFixed(1)}%, ${colorFor(hi, lo, hi)})`;
  document.getElementById("scaleMin").textContent = f3(lo) + " m";
  document.getElementById("scaleMax").textContent = f3(hi) + " m";
}

function renderGrid(a) {
  const objs = a.elites.map(e => e.archived_fitness);
  const lo = Math.min(...objs), hi = Math.max(...objs);
  renderScale(lo, hi);
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  if (isCvt(a)) { grid.classList.add("map"); grid.style.gridTemplateColumns = ""; renderMap(a, lo, hi); return; }
  grid.classList.remove("map");

  const [rows, cols] = a.grid;
  const byCell = new Map(a.elites.map(e => [keyOf(e), e]));
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  // Row 0 of the archive is the bottom of the plot, matching the saved heatmaps.
  for (let r = rows - 1; r >= 0; r--) {
    for (let c = 0; c < cols; c++) {
      const e = byCell.get(r + "," + c);
      const b = document.createElement("button");
      b.className = "cell";
      b.dataset.filled = e ? "1" : "0";
      b.dataset.clip = e && e.src ? "1" : "0";
      b.setAttribute("aria-pressed", "false");
      if (e) {
        b.dataset.key = keyOf(e);
        b.style.background = colorFor(e.archived_fitness, lo, hi);
        b.title = `cell (${r}, ${c}) · ${fmt(mx(e))} / ${fmt(my(e))} · ${f3(e.archived_fitness)} m`;
        b.setAttribute("aria-label", b.title);
        b.onclick = () => select(e);
      } else {
        b.tabIndex = -1;
        b.setAttribute("aria-label", `cell (${r}, ${c}), empty`);
      }
      grid.appendChild(b);
    }
  }
}

// The CVT map: every centroid at its PCA projection, filled cells on top,
// coloured like grid cells. Plot y goes up, SVG y goes down, hence the flip.
function renderMap(a, lo, hi) {
  const pts = a.centroids2d;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = 0.6;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const filled = new Map(a.elites.map(e => [e.cell[0], e]));
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `${x0} ${-y1} ${x1 - x0} ${y1 - y0}`);
  svg.setAttribute("role", "group");
  svg.setAttribute("aria-label", "Archive cells on the projected latent plane");
  const circle = (p, r, cls) => {
    const c = document.createElementNS(svgNS, "circle");
    c.setAttribute("cx", p[0]); c.setAttribute("cy", -p[1]); c.setAttribute("r", r);
    c.setAttribute("class", cls);
    return c;
  };
  pts.forEach((p, i) => { if (!filled.has(i)) svg.appendChild(circle(p, 0.09, "pt-empty")); });
  // Draw in ascending fitness so the best of an overlapping pile is on top.
  [...a.elites].sort((x, y) => x.archived_fitness - y.archived_fitness).forEach(e => {
    const c = circle(e.xy, 0.19, "pt");
    c.setAttribute("fill", colorFor(e.archived_fitness, lo, hi));
    c.setAttribute("role", "button");
    c.setAttribute("tabindex", "0");
    c.setAttribute("aria-pressed", "false");
    c.dataset.key = keyOf(e);
    c.dataset.clip = e.src ? "1" : "0";
    const label = `${cellName(e)} · ${e.posture} · ${f3(e.archived_fitness)} m`;
    c.setAttribute("aria-label", label);
    const t = document.createElementNS(svgNS, "title");
    t.textContent = label;
    c.appendChild(t);
    c.onclick = () => select(e);
    c.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(e); } };
    svg.appendChild(c);
  });
  document.getElementById("grid").appendChild(svg);
}

function renderTop(a) {
  const top = [...a.elites].sort((x, y) => y.archived_fitness - x.archived_fitness).slice(0, 8);
  document.getElementById("topList").innerHTML = top.map((e, i) =>
    `<button class="top-item" data-cell="${keyOf(e)}">
       <span class="rank">${i + 1}</span>
       <span class="meta">${isCvt(a) ? `#${String(e.cell[0]).padStart(4, "0")} · ${e.posture} · up ${e.upright_s.toFixed(1)} s` : `${cellName(e)} · upright ${e.upright_s.toFixed(2)} s`}</span>
       <span class="val">${f3(e.archived_fitness)} m</span>
     </button>`).join("");
  document.querySelectorAll(".top-item").forEach(btn => {
    btn.onclick = () => {
      const key = btn.dataset.cell;
      select(DATA[current].elites.find(e => keyOf(e) === key));
    };
  });
}

function select(e) {
  if (!e) return;
  selected = e;
  const a = DATA[current];
  document.querySelectorAll(".cell, .pt").forEach(c => c.setAttribute("aria-pressed", "false"));
  const idx = [...document.querySelectorAll(".cell, .pt")].find(c => c.dataset.key === keyOf(e));
  if (idx) { idx.setAttribute("aria-pressed", "true"); }

  document.getElementById("selTitle").textContent = cap(cellName(e)) + (isCvt(a) ? ` · ${e.posture}` : "");
  const d = descOf(a);
  document.getElementById("selHint").textContent = (isCvt(a)
    ? `Latent ${vec(e.latent)}. `
    : `${cap(d.labels[0])} ${fmt(mx(e))}, ${d.labels[1]} ${fmt(my(e))}. `) +
    `Left foot down ${(e.duty_left * 100).toFixed(0)}% of the time, right foot ${(e.duty_right * 100).toFixed(0)}%.`;

  const player = document.getElementById("player");
  player.innerHTML = e.src
    ? `<video src="${e.src}" autoplay loop muted playsinline controls></video>`
    : `<div class="noclip"><div>Not embedded — this page ships a subset to stay under the size limit.<br><br>
         Full clip on disk:<br><code>${e.path}</code></div></div>`;

  const ok = e.survived ? 1 : 0;
  const axes = isCvt(a) ? `
    <div><dt>Posture</dt><dd>${cap(e.posture)}</dd></div>
    <div><dt>Viable replicas</dt><dd>${e.viable_count} / ${a.replicas}</dd></div>
    <div><dt>Latent (archive)</dt><dd>${vec(e.latent)}</dd></div>
    <div><dt>Latent (eval space)</dt><dd>${vec(e.eval_latent)}</dd></div>
    <div><dt>Filed at insertion</dt><dd>${f3(e.filed_fitness)} m</dd></div>` : `
    <div><dt>${cap(d.labels[0])}</dt><dd>${fmt(mx(e))}</dd></div>
    <div><dt>${cap(d.labels[1])}</dt><dd>${fmt(my(e))}</dd></div>`;
  document.getElementById("readout").innerHTML = `
    <div><dt>${a.verified ? `Verified median (${a.replicas || 8} rollouts)` : "Archived fitness"}</dt>
        <dd>${f3(e.archived_fitness)} m</dd></div>
    <div><dt>This rollout</dt><dd>${f3(e.replay_fitness)} m</dd></div>
    <div><dt>Distance travelled</dt><dd>${f3(e.displacement_m)} m</dd></div>
    <div><dt>Time upright</dt><dd>${e.upright_s.toFixed(2)} s</dd></div>${axes}
    <div><dt>Duty L / R</dt><dd>${e.duty_left.toFixed(2)} / ${e.duty_right.toFixed(2)}</dd></div>
    <div><dt>Outcome</dt><dd><span class="verdict" data-ok="${ok}">${e.survived ? "Stayed up" : "Fell"}</span></dd></div>`;
}

function show(i) {
  current = i;
  const a = DATA[i];
  document.querySelectorAll("#switch button").forEach((b, j) =>
    b.setAttribute("aria-pressed", String(j === i)));
  renderAxes(a); renderStats(a); renderGrid(a); renderTop(a);
  const best = [...a.elites].sort((x, y) => y.archived_fitness - x.archived_fitness)[0];
  select(best);
  const ev = isCvt(a) ? a.projection.explained : null;
  const cvtNotes = isCvt(a) ? `
    <p><strong>The map.</strong> This archive has no grid: its ${a.stats.cells} cells are the Voronoi regions
    of a centroid set in a ${a.projection.latentDim}-d behaviour space learned from rollouts, and the two
    postures are DBSCAN clusters of the verified elites in that space, named by the v4 label their members
    carry. The plane is ${a.projection.method} (PC1 ${pct(ev[0])}, PC2 ${pct(ev[1])} of the centroids'
    variance — ${pct(ev[0] + ev[1])} together), every cell drawn at its centroid's projection, so distance
    here is approximate: two points that overlap may be far apart in the two dimensions folded away.
    Arrow keys step to the nearest filled cell in the arrow's direction on this plane.</p>
    <p><strong>Verified means.</strong> Every elite shown was re-rolled in ${a.replicas} fresh world-permuted
    replicas and kept only if it passed the viability predicate in at least 5 of them; the colour and the
    "verified median" are the median forward distance over those replicas, and "filed at insertion" is the
    sample the search inserted on. Time upright uses the v1–v3 rule (trunk above 7.5 cm, tilt under 60°),
    so a crawler reads ~0 s by construction — that is its posture, not a failure.</p>` : "";
  document.getElementById("footnotes").innerHTML = `
    <p><strong>What you are watching.</strong> Not a re-simulation: each elite was rolled out in the
    same batched MuJoCo-Warp harness that produced its archived fitness, with joint positions logged
    every control step and replayed through CPU MuJoCo to rasterise. The poses are the ones that were scored.</p>
    <p><strong>Archived vs replay fitness.</strong> MuJoCo-Warp's batched contact solve is
    order-sensitive, so re-running a genome moves its fitness by a few millimetres. MAP-Elites keeps the
    best sample per cell, which makes the archive mildly optimistic — largest at the very top, where a
    gait sits on a tipping point.</p>${cvtNotes}
    <p><strong>${a.stats.embedded} of ${a.stats.elites} clips</strong> are embedded here
    (${a.stats.embeddedMb.toFixed(1)} MB): the highest-fitness gaits plus a sweep across the ${isCvt(a) ? "latent" : "descriptor"}
    space. Every filled cell was rendered to disk at full resolution.</p>`;
}

document.getElementById("switch").innerHTML = DATA.map((a, i) =>
  `<button aria-pressed="${i === 0}">${a.label} · ${a.genome.toUpperCase()}</button>`).join("");
document.querySelectorAll("#switch button").forEach((b, i) => { b.onclick = () => show(i); });
if (DATA.length < 2) document.getElementById("switch").style.display = "none";

document.getElementById("grid").addEventListener("keydown", (ev) => {
  if (!selected || !ev.key.startsWith("Arrow")) return;
  ev.preventDefault();
  const d = { ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[ev.key];
  if (!d) return;
  const a = DATA[current];
  if (isCvt(a)) {
    // Nearest filled cell within 45° of the arrow's direction on the plane.
    const [dy, dx] = d;
    const cands = a.elites.filter(e => {
      const vx = e.xy[0] - selected.xy[0], vy = e.xy[1] - selected.xy[1];
      const along = vx * dx + vy * dy, across = Math.abs(vx * dy - vy * dx);
      return along > 0 && across <= along;
    });
    if (!cands.length) return;
    const dist = (e) => Math.hypot(e.xy[0] - selected.xy[0], e.xy[1] - selected.xy[1]);
    const next = cands.sort((x, y) => dist(x) - dist(y))[0];
    select(next);
    document.querySelector(`.pt[data-key="${keyOf(next)}"]`)?.focus();
    return;
  }
  // Step to the nearest filled cell in that direction — the archive has holes,
  // so a naive +1 would dead-end on an empty cell.
  const cands = a.elites.filter(e =>
    Math.sign(e.cell[0] - selected.cell[0]) === d[0] &&
    Math.sign(e.cell[1] - selected.cell[1]) === d[1] ||
    (d[0] !== 0 && e.cell[1] === selected.cell[1] && Math.sign(e.cell[0] - selected.cell[0]) === d[0]) ||
    (d[1] !== 0 && e.cell[0] === selected.cell[0] && Math.sign(e.cell[1] - selected.cell[1]) === d[1]));
  if (!cands.length) return;
  const dist = (e) => Math.hypot(e.cell[0] - selected.cell[0], e.cell[1] - selected.cell[1]);
  select(cands.sort((x, y) => dist(x) - dist(y))[0]);
});

show(0);
</script>
"""


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    labels = list(args.labels) or [p.parent.name for p in args.manifests]
    if len(labels) != len(args.manifests):
        raise SystemExit("--labels must have one entry per manifest")

    budgets = list(args.budgets_mb) or [args.budget_mb] * len(args.manifests)
    if len(budgets) != len(args.manifests):
        raise SystemExit("--budgets-mb must have one entry per manifest")

    archives = [
        _load(p, label, budget * 1e6, args)
        for p, label, budget in zip(args.manifests, labels, budgets)
    ]
    html = PAGE.replace("__DATA__", json.dumps(archives, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    size = args.out.stat().st_size / 1e6
    for a in archives:
        print(f"{a['label']}: {a['stats']['elites']} elites, "
              f"{a['stats']['embedded']} clips embedded "
              f"({a['stats']['embeddedMb']:.1f} MB raw)")
    print(f"wrote {args.out} ({size:.1f} MB)")
    if size > 15.5:
        print("WARNING: over the 16 MB artifact ceiling — lower --budget-mb")


if __name__ == "__main__":
    main()
