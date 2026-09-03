#!/usr/bin/env bash
# Checkpoint 4: turn a finished v4 run into the numbers and the page.
#
# Per mode, in order:
#   1. independent 8-replica P2' verification, scored at the gate's own k
#      (5-of-8) with the full strictness sweep printed alongside;
#   2. render every filled cell of the VERIFIED archive, so the colour on the
#      page is a median over eight rollouts rather than one lucky sample;
#   3. one viewer with a tab per mode.
#
# An empty sub-archive is skipped with a line saying so, not with silence:
# roll is expected empty (no seed exists on this machine, and no open-loop
# probe rotates), and "the roll tab is missing" should never be something the
# reader has to infer.
#
#   MUJOCO_GL=glfw bash qd/finalize_v4.sh logs/qd/modes_v4/final
set -eu

RUN_DIR="${1:-logs/qd/modes_v4/final}"
OUT="${2:-logs/qd/v4}"
MODES="walk crawl roll hop other"

mkdir -p "$OUT"
MANIFESTS=()
LABELS=()

for mode in $MODES; do
  archive="$RUN_DIR/archive_${mode}.npz"
  if [ ! -f "$archive" ]; then
    echo "== $mode: no archive at $archive — skipping"
    continue
  fi
  n=$(uv run --no-sync python -c "
from qd.common import load_archive
print(len(load_archive('$archive')['objective']))
" 2>/dev/null | tail -1)
  if [ "$n" -eq 0 ]; then
    echo "== $mode: archive is EMPTY (0 elites) — skipping render, reporting the zero"
    continue
  fi

  echo "== $mode: verifying $n elites under P2'/5-of-8"
  uv run --no-sync python -m qd.verify_modes \
      --archive "$archive" \
      --out "$OUT/verified_${mode}.npz" \
      2>&1 | tee "$OUT/verify_${mode}.log"

  verified="$OUT/verified_${mode}.npz"
  [ -f "$verified" ] || { echo "== $mode: nothing survived verification"; continue; }

  echo "== $mode: rendering the verified archive"
  uv run --no-sync python -m qd.render_gaits \
      --archive "$verified" --out "$OUT/gaits/$mode" \
      2>&1 | tail -3
  MANIFESTS+=("$OUT/gaits/$mode/manifest.json")
  LABELS+=("$mode")
done

if [ ${#MANIFESTS[@]} -eq 0 ]; then
  echo "no mode produced a renderable archive; not building a viewer"
  exit 0
fi

echo "== building the viewer with ${#MANIFESTS[@]} mode tabs: ${LABELS[*]}"
uv run --no-sync python -m qd.build_viewer \
    --manifests "${MANIFESTS[@]}" \
    --labels "${LABELS[@]}" \
    --out "$OUT/viewer/index.html"
echo "wrote $OUT/viewer/index.html"
