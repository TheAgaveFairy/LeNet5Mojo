#!/usr/bin/env bash
# Sweep DIV_CHANS_CONV2 / DIV_CHANS_CONV3 to find the fastest combination.
#
# Metric: median fps from the batchedForwardMultiStream line of `--bench-only`.
# Default mode is coordinate descent (cheap): sweep conv2 with conv3 fixed, pick
# the best, then sweep conv3 at that best conv2. Pass --grid for a full cross
# product (slower: every conv2 x conv3 pair).
#
# Prereqs:
#   - models/deleteme.test must exist (run `pixi run doit` once to train+save).
#   - For stable numbers, lock clocks first: `pixi run gpulock` (sudo, real term).
#
# Usage:
#   scripts/sweep_div_chans.sh                # coordinate descent, defaults
#   scripts/sweep_div_chans.sh --grid         # full cross product
#   CONV2_VALUES="2 4 8" CONV3_VALUES="4 8" scripts/sweep_div_chans.sh
set -euo pipefail

# --- locate repo root (script lives in <root>/scripts) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# conv2 must divide LAYER3=16 AND keep threads <=1024 ((16/div)*100<=1024 => div>=2)
CONV2_VALUES="${CONV2_VALUES:-2 4 8 16}"
# conv3 must be a factor of 120. Two failure modes:
#   - LOW div (2,3): num_ocs=120/div is large -> huge comptime-for unroll ->
#     compile-time blowup / build fail (caught by BUILD_TIMEOUT or nonzero rc).
#   - HIGH div (60,120): num_ocs tiny but grid.y huge -> many layer4 reloads +
#     launch overhead -> the slow "worst case" end (compiles, just slow).
# Default set skips the known-explosive 2,3 but includes the high end to observe
# the slow tail. Add them back with e.g. CONV3_VALUES="2 3 4 ... 60".
CONV3_VALUES="${CONV3_VALUES:-4 5 6 8 10 12 15 20 30 60}"
CONV2_DEFAULT=8
CONV3_DEFAULT=8
# Per-config wall-clock cap. Compile-time explosion (low div) hangs instead of
# erroring, so we kill it. Override with BUILD_TIMEOUT=900 etc.
BUILD_TIMEOUT="${BUILD_TIMEOUT:-600}"

# GPU batch/stream config to measure AT. Unset = the compiled-in constants.mojo
# defaults, i.e. the RTX 3070 tuning — unchanged behaviour on that box.
#
# Set these on any card whose tuning differs, because the measurement is only as
# good as the config it runs in. On the RX 7600 the 3070 defaults (bs=100 s=12)
# sit in the BIMODAL region documented in TODO.md — ~18.5% run-to-run spread,
# two disjoint modes. A div_chans effect of a few percent is invisible under
# that; you would be sampling which mode you landed in, not which divisor is
# faster. At bs=250 s=3 the same card measures 0.12% spread, so the signal is
# recoverable. Example:
#   GPU_BS=250 GPU_STREAMS=3 pixi run bash scripts/sweep_div_chans.sh
GPU_BS="${GPU_BS:-}"
GPU_STREAMS="${GPU_STREAMS:-}"

EXTRA_D=()
[[ -n "$GPU_BS" ]] && EXTRA_D+=(-D "GPU_STREAM_BATCH_SIZE=$GPU_BS")
EXTRA_ARGS=()
[[ -n "$GPU_STREAMS" ]] && EXTRA_ARGS+=(--num-streams "$GPU_STREAMS")

GRID=0
[[ "${1:-}" == "--grid" ]] && GRID=1

if [[ ! -f models/deleteme.test ]]; then
  echo "ERROR: models/deleteme.test not found. Run 'pixi run doit' once to train+save." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/sweep_div_chans_${STAMP}.csv"
mkdir -p results
echo "conv2,conv3,fps,ns_per_img,accuracy_pct,status" > "$OUT"
echo "Writing results to $OUT"
echo "Measuring at: bs=${GPU_BS:-<constants.mojo default>} streams=${GPU_STREAMS:-<constants.mojo default>}"
echo

# Run one config. Args: conv2 conv3.
#   - progress + diagnostics -> stderr (shown live)
#   - one CSV row appended to $OUT (every path: ok / fail / timeout / no_metric)
#   - stdout: the fps integer on success, or "ERR" on any failure (for capture)
run_one() {
  local c2="$1" c3="$2"
  local log rc=0
  log="$(mktemp)"
  # `timeout` returns 124 on kill (compile-time explosion); other nonzero = build/run error.
  timeout "$BUILD_TIMEOUT" \
    mojo -D DIV_CHANS_CONV2="$c2" -D DIV_CHANS_CONV3="$c3" \
      ${EXTRA_D[@]+"${EXTRA_D[@]}"} src/main.mojo --bench-only \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"$log" 2>&1 || rc=$?

  if (( rc == 124 )); then
    printf "  conv2=%-2s conv3=%-2s -> TIMEOUT after %ss (likely compile blowup; log %s)\n" \
      "$c2" "$c3" "$BUILD_TIMEOUT" "$log" >&2
    echo "$c2,$c3,,,,timeout" >> "$OUT"; echo "ERR"; return
  fi
  if (( rc != 0 )); then
    printf "  conv2=%-2s conv3=%-2s -> BUILD/RUN FAILED (rc=%s; log %s)\n" \
      "$c2" "$c3" "$rc" "$log" >&2
    echo "$c2,$c3,,,,fail" >> "$OUT"; echo "ERR"; return
  fi

  local line fps nsimg acc
  line="$(grep 'batchedForwardMultiStream' "$log" | tail -1 || true)"
  if [[ -z "$line" ]]; then
    printf "  conv2=%-2s conv3=%-2s -> no timing line (log %s)\n" "$c2" "$c3" "$log" >&2
    echo "$c2,$c3,,,,no_metric" >> "$OUT"; echo "ERR"; return
  fi
  fps="$(grep -oE '[0-9]+ fps'    <<<"$line" | grep -oE '[0-9]+' | head -1 || true)"
  nsimg="$(grep -oE '[0-9]+ns/img' <<<"$line" | grep -oE '[0-9]+' | head -1 || true)"
  acc="$(grep -oE '\([0-9]+%\)'    <<<"$line" | grep -oE '[0-9]+' | head -1 || true)"
  rm -f "$log"
  printf "  conv2=%-2s conv3=%-2s -> %6s fps  %7s ns/img  acc=%s%%\n" \
    "$c2" "$c3" "$fps" "$nsimg" "$acc" >&2
  echo "$c2,$c3,$fps,$nsimg,$acc,ok" >> "$OUT"
  echo "$fps"
}

best_fps=-1; best_c2=""; best_c3=""
note_best() { # fps c2 c3
  if [[ "$1" =~ ^[0-9]+$ ]] && (( $1 > best_fps )); then
    best_fps="$1"; best_c2="$2"; best_c3="$3"
  fi
}

if (( GRID )); then
  echo "== Full grid sweep =="
  for c2 in $CONV2_VALUES; do
    for c3 in $CONV3_VALUES; do
      out="$(run_one "$c2" "$c3" | tail -1)"
      [[ "$out" != "ERR" ]] && note_best "$(awk '{print $1}' <<<"$out")" "$c2" "$c3"
    done
  done
else
  echo "== Phase 1: sweep conv2 (conv3=$CONV3_DEFAULT) =="
  p1_best_fps=-1; p1_best_c2="$CONV2_DEFAULT"
  for c2 in $CONV2_VALUES; do
    out="$(run_one "$c2" "$CONV3_DEFAULT" | tail -1)"
    if [[ "$out" != "ERR" ]]; then
      f="$(awk '{print $1}' <<<"$out")"
      note_best "$f" "$c2" "$CONV3_DEFAULT"
      if [[ "$f" =~ ^[0-9]+$ ]] && (( f > p1_best_fps )); then p1_best_fps="$f"; p1_best_c2="$c2"; fi
    fi
  done
  echo
  echo "== Phase 2: sweep conv3 (conv2=$p1_best_c2, best from phase 1) =="
  for c3 in $CONV3_VALUES; do
    [[ "$c3" == "$CONV3_DEFAULT" ]] && continue  # already measured in phase 1
    out="$(run_one "$p1_best_c2" "$c3" | tail -1)"
    [[ "$out" != "ERR" ]] && note_best "$(awk '{print $1}' <<<"$out")" "$p1_best_c2" "$c3"
  done
fi

echo
echo "================================================"
if [[ -n "$best_c2" ]]; then
  echo "BEST: DIV_CHANS_CONV2=$best_c2 DIV_CHANS_CONV3=$best_c3  ($best_fps fps)"
else
  echo "No successful runs."
fi
echo "Full CSV: $OUT"
