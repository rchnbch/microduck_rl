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
hard 16 MB ceiling. ``--budget-mb`` fills that ceiling deliberately: every
elite that survived the full episode first (those are the rarest and most
informative gaits, and they are *not* the highest-scoring ones), then the top
elites by fitness, then a spatial sweep across the descriptor space so every
region of the archive is represented. Cells without an embedded clip stay
clickable and show their stats plus the path to the full-resolution file on
disk, which ``qd.render_gaits`` wrote for *every* filled cell.
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

    top: int = 16
    """Highest-fitness elites embedded before the spatial sweep begins."""


def _select_clips(entries: list[dict], budget_bytes: float, top: int) -> set[int]:
    """Rows to embed: survivors, then the best elites, then a spatial spread.

    Survivors come FIRST regardless of fitness. On this robot almost nothing
    stays upright for the whole episode, so those clips are the most
    informative ones in the archive — and they are not the highest-scoring, so
    a purely fitness-ranked selection drops exactly the gaits a reader most
    needs to see. The spatial pass then walks cells in order of distance from
    the already-chosen set, so coverage fills in evenly instead of clustering
    near the peak.
    """
    chosen: list[dict] = []
    used = 0.0
    for entry in sorted(
        (e for e in entries if e.get("survived")), key=lambda e: -e["displacement_m"]
    ):
        if used + entry["bytes"] > budget_bytes:
            break
        chosen.append(entry)
        used += entry["bytes"]

    by_fitness = sorted(entries, key=lambda e: -e["archived_fitness"])
    for entry in by_fitness[:top]:
        if entry in chosen:
            continue
        if used + entry["bytes"] > budget_bytes:
            break
        chosen.append(entry)
        used += entry["bytes"]

    remaining = [e for e in entries if e not in chosen]
    while remaining and used < budget_bytes:
        picked = np.array([e["cell"] for e in chosen], dtype=float)
        cells = np.array([e["cell"] for e in remaining], dtype=float)
        if len(picked) == 0:
            idx = 0
        else:
            dist = np.linalg.norm(cells[:, None, :] - picked[None, :, :], axis=-1)
            idx = int(np.argmax(dist.min(axis=1)))
        entry = remaining.pop(idx)
        if used + entry["bytes"] <= budget_bytes:
            chosen.append(entry)
            used += entry["bytes"]
    return {e["row"] for e in chosen}


def _load(manifest_path: Path, label: str, budget_bytes: float, top: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    entries = manifest["elites"]
    embed = _select_clips(entries, budget_bytes, top)

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
    return {
        "label": label,
        "genome": manifest["genome"],
        "grid": dims,
        "fps": manifest["fps"],
        "episodeSeconds": manifest["episode_seconds"],
        "elites": out_entries,
        "stats": {
            "elites": len(entries),
            "coverage": len(entries) / (dims[0] * dims[1]),
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
    <p class="lede">Every cell is a different <em>way of walking</em>, indexed by how much of the
    episode each foot spent on the ground. MAP-Elites keeps the best gait found for each
    combination — so this is a map of behaviours, not a single best policy. Pick a cell to watch it.</p>
  </header>

  <div class="switch" id="switch" role="group" aria-label="Archive"></div>

  <dl class="stats" id="stats"></dl>

  <div class="panel">
    <section class="card">
      <h2>Behaviour archive</h2>
      <p class="hint">Colour is fitness: forward metres, minus a penalty for time spent fallen.
        Click a cell, or focus the grid and use the arrow keys.</p>
      <div class="grid-frame">
        <div class="ylab">Right foot duty factor →</div>
        <div class="grid" id="grid" role="grid" aria-label="Archive cells"></div>
        <div></div>
        <div class="xlab">Left foot duty factor →</div>
      </div>
      <div class="scale">
        <span id="scaleMin">—</span>
        <div class="ramp" id="ramp"></div>
        <span id="scaleMax">—</span>
      </div>
      <div class="legend-note"><span class="swatch"></span> never filled — no gait produced this contact pattern</div>
      <div class="legend-note"><span class="dot"></span> clip embedded in this page</div>
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
  document.getElementById("stats").innerHTML = rows
    .map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

function renderGrid(a) {
  const [rows, cols] = a.grid;
  const objs = a.elites.map(e => e.archived_fitness);
  const lo = Math.min(...objs), hi = Math.max(...objs);
  const byCell = new Map(a.elites.map(e => [e.cell[0] + "," + e.cell[1], e]));

  document.getElementById("ramp").style.background =
    `linear-gradient(90deg, ${colorFor(lo, lo, hi)}, ${colorFor(0, lo, hi)} ${(-lo/(hi-lo)*100).toFixed(1)}%, ${colorFor(hi, lo, hi)})`;
  document.getElementById("scaleMin").textContent = f3(lo) + " m";
  document.getElementById("scaleMax").textContent = f3(hi) + " m";

  const grid = document.getElementById("grid");
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  grid.innerHTML = "";
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
        b.style.background = colorFor(e.archived_fitness, lo, hi);
        b.title = `cell (${r}, ${c}) · duty ${e.duty_left.toFixed(2)} / ${e.duty_right.toFixed(2)} · ${f3(e.archived_fitness)} m`;
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

function renderTop(a) {
  const top = [...a.elites].sort((x, y) => y.archived_fitness - x.archived_fitness).slice(0, 8);
  document.getElementById("topList").innerHTML = top.map((e, i) =>
    `<button class="top-item" data-cell="${e.cell[0]},${e.cell[1]}">
       <span class="rank">${i + 1}</span>
       <span class="meta">cell (${e.cell[0]}, ${e.cell[1]}) · upright ${e.upright_s.toFixed(2)} s</span>
       <span class="val">${f3(e.archived_fitness)} m</span>
     </button>`).join("");
  document.querySelectorAll(".top-item").forEach(btn => {
    btn.onclick = () => {
      const key = btn.dataset.cell;
      select(DATA[current].elites.find(e => e.cell.join(",") === key));
    };
  });
}

function select(e) {
  if (!e) return;
  selected = e;
  document.querySelectorAll(".cell").forEach(c => c.setAttribute("aria-pressed", "false"));
  const idx = [...document.querySelectorAll(".cell")].find(c =>
    c.getAttribute("aria-label")?.startsWith(`cell (${e.cell[0]}, ${e.cell[1]})`));
  if (idx) { idx.setAttribute("aria-pressed", "true"); }

  document.getElementById("selTitle").textContent = `Cell (${e.cell[0]}, ${e.cell[1]})`;
  document.getElementById("selHint").textContent =
    `Left foot down ${(e.duty_left * 100).toFixed(0)}% of the time, right foot ${(e.duty_right * 100).toFixed(0)}%.`;

  const player = document.getElementById("player");
  player.innerHTML = e.src
    ? `<video src="${e.src}" autoplay loop muted playsinline controls></video>`
    : `<div class="noclip"><div>Not embedded — this page ships a subset to stay under the size limit.<br><br>
         Full clip on disk:<br><code>${e.path}</code></div></div>`;

  const ok = e.survived ? 1 : 0;
  document.getElementById("readout").innerHTML = `
    <div><dt>Archived fitness</dt><dd>${f3(e.archived_fitness)} m</dd></div>
    <div><dt>Replay fitness</dt><dd>${f3(e.replay_fitness)} m</dd></div>
    <div><dt>Distance travelled</dt><dd>${f3(e.displacement_m)} m</dd></div>
    <div><dt>Time upright</dt><dd>${e.upright_s.toFixed(2)} s</dd></div>
    <div><dt>Duty L / R</dt><dd>${e.duty_left.toFixed(2)} / ${e.duty_right.toFixed(2)}</dd></div>
    <div><dt>Outcome</dt><dd><span class="verdict" data-ok="${ok}">${e.survived ? "Stayed up" : "Fell"}</span></dd></div>`;
}

function show(i) {
  current = i;
  const a = DATA[i];
  document.querySelectorAll("#switch button").forEach((b, j) =>
    b.setAttribute("aria-pressed", String(j === i)));
  renderStats(a); renderGrid(a); renderTop(a);
  const best = [...a.elites].sort((x, y) => y.archived_fitness - x.archived_fitness)[0];
  select(best);
  document.getElementById("footnotes").innerHTML = `
    <p><strong>What you are watching.</strong> Not a re-simulation: each elite was rolled out in the
    same batched MuJoCo-Warp harness that produced its archived fitness, with joint positions logged
    every control step and replayed through CPU MuJoCo to rasterise. The poses are the ones that were scored.</p>
    <p><strong>Archived vs replay fitness.</strong> MuJoCo-Warp's batched contact solve is
    order-sensitive, so re-running a genome moves its fitness by a few millimetres. MAP-Elites keeps the
    best sample per cell, which makes the archive mildly optimistic — largest at the very top, where a
    gait sits on a tipping point.</p>
    <p><strong>${a.stats.embedded} of ${a.stats.elites} clips</strong> are embedded here
    (${a.stats.embeddedMb.toFixed(1)} MB): the highest-fitness gaits plus a sweep across the descriptor
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

    archives = [
        _load(p, label, args.budget_mb * 1e6, args.top)
        for p, label in zip(args.manifests, labels)
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
