#!/usr/bin/env bash
# v5 verification: v5's final archive and v4's two archives as ONE set, both on
# the frozen evaluation space and the fixed centroids, at the same 5-of-8 bar.
#   bash qd/finalize_v5.sh logs/qd/aurora_v5/final.npz logs/qd/v5
set -euo pipefail
FINAL=${1:-logs/qd/aurora_v5/final.npz}
OUT=${2:-logs/qd/v5}
V4=${V4_DIR:-qd-run-archives/j007/modes_v4/final}
SPACE=${SPACE:-logs/qd/v5/space_eval/space_ae.npz}
CENT=${CENT:-logs/qd/v5/space_eval/centroids_ae.npz}

uv run --group qd python -m qd.verify_aurora --space "$SPACE" --centroids "$CENT" \
    --archives v5 "$FINAL" --out "$OUT/verify_v5" 2>&1 | tee "$OUT/verify_v5.log"
uv run --group qd python -m qd.verify_aurora --space "$SPACE" --centroids "$CENT" \
    --archives v4_walk "$V4/archive_walk.npz" v4_crawl "$V4/archive_crawl.npz" \
    --out "$OUT/verify_v4" 2>&1 | tee "$OUT/verify_v4.log"
