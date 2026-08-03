#!/usr/bin/env bash
# Sweep GPU_STREAM_BATCH_SIZE x NUM_GPU_STREAMS, print the batchedForward timing lines.
#
# Defaults reproduce the original broad sweep. Override either axis via env vars, e.g.
# a focused stream sweep at the sweet-spot batch sizes:
#   BS_VALUES="50 75 100" STREAM_VALUES="3 4 5 6 7 8" scripts/grid_search_gpu.sh
#
# Writes a CSV to results/ as well as printing, so a sweep can be re-analysed
# without re-running it (a full grid is ~30-50 min, see cost model below).
#
# ── REPEATS: why each point runs more than once ───────────────────────────────
# GPU throughput on this project is BIMODAL per process — two tight, disjoint
# clusters ~23% apart, fixed at process startup. Locking clocks does not remove
# it; see the "GPU throughput is BIMODAL" entry in TODO.md for the evidence and
# what has been ruled out. A single sample per config is therefore a coin flip
# between modes, which is far larger than the ~1% differences a knee search is
# trying to resolve.
#
# So each config runs REPEATS times and we keep the BEST. Best-of-N, not median:
# the low mode is a defect state rather than a slower-but-valid configuration, so
# averaging it in would just blur two different things together. All individual
# samples go to the CSV so the mode split stays visible and this decision can be
# revisited without re-running.
#
# ── Cost model (measured 2026-08-02, RX 7600) ─────────────────────────────────
#   first run at a new bs : ~123 s  (JIT compile dominates; the bench itself is ~10 ms)
#   subsequent runs at same bs : ~14 s  (JIT cache hit)
# bs stays comptime (-D, re-keys the JIT cache) so it's the outer loop: each bs
# compiles once, then the inner streams loop reuses that binary via the runtime
# --num-streams arg (no recompile per stream value). Total is roughly
#   n_bs * (123 + (REPEATS * n_streams - 1) * 14) seconds.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BS_VALUES="${BS_VALUES:-25 50 75 100 125 150 200 250 320 400 500 625 750 1000 2000}"
STREAM_VALUES="${STREAM_VALUES:-2 3 5}"
REPEATS="${REPEATS:-3}"

mkdir -p results
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/grid_search_gpu_${STAMP}.csv"
echo "bs,streams,rep,eff_batch,fps,ns_per_img,median_ms,min_ms,max_ms,accuracy_pct" > "$OUT"
echo "Writing results to $OUT"
echo "REPEATS=$REPEATS (best-of-N per config; see header for why)"
echo

best_overall=-1; best_bs=""; best_streams=""

for bs in $BS_VALUES; do
  for streams in $STREAM_VALUES; do
    best_fps=-1
    samples=()
    for (( rep = 1; rep <= REPEATS; rep++ )); do
      line="$(pixi run mojo -D GPU_STREAM_BATCH_SIZE="$bs" src/main.mojo \
                --bench-only --num-streams "$streams" 2>&1 \
              | grep -E "batchedForwardMultiStream" | tail -1)"

      if [[ -z "$line" ]]; then
        echo "$bs,$streams,$rep,,,,,,," >> "$OUT"
        samples+=("ERR")
        continue
      fi

      # batchedForwardMultiStream[s=12]: eff_batch=1200, 9682/10000 (96%) correct, 12ms (1204ns/img), 830315 fps [min=11ms max=14ms]
      eff="$(grep -oE 'eff_batch=[0-9]+'  <<<"$line" | cut -d= -f2)"
      acc="$(grep -oE '\([0-9]+%\)'       <<<"$line" | grep -oE '[0-9]+')"
      med="$(grep -oE ' [0-9]+ms \('      <<<"$line" | grep -oE '[0-9]+')"
      nsi="$(grep -oE '[0-9]+ns/img'      <<<"$line" | grep -oE '[0-9]+')"
      fps="$(grep -oE '[0-9]+ fps'        <<<"$line" | grep -oE '[0-9]+')"
      mn="$(grep -oE 'min=[0-9]+ms'       <<<"$line" | grep -oE '[0-9]+')"
      mx="$(grep -oE 'max=[0-9]+ms'       <<<"$line" | grep -oE '[0-9]+')"

      echo "$bs,$streams,$rep,$eff,$fps,$nsi,$med,$mn,$mx,$acc" >> "$OUT"
      samples+=("$fps")
      [[ "$fps" =~ ^[0-9]+$ ]] && (( fps > best_fps )) && best_fps="$fps"
    done

    printf '=== bs=%-4s streams=%-2s -> best %8s fps   (samples: %s)\n' \
      "$bs" "$streams" "$best_fps" "${samples[*]}"

    if [[ "$best_fps" =~ ^[0-9]+$ ]] && (( best_fps > best_overall )); then
      best_overall="$best_fps"; best_bs="$bs"; best_streams="$streams"
    fi
  done
done

echo
echo "================================================"
if [[ -n "$best_bs" ]]; then
  echo "BEST: GPU_STREAM_BATCH_SIZE=$best_bs --num-streams $best_streams  ($best_overall fps)"
else
  echo "No successful runs."
fi
echo "Full CSV: $OUT"
