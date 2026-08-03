#!/usr/bin/env bash
# Sweep GPU_STREAM_BATCH_SIZE x NUM_GPU_STREAMS, print the batchedForward timing lines.
#
# Defaults reproduce the original broad sweep. Override either axis via env vars, e.g.
# a focused stream sweep at the sweet-spot batch sizes:
#   BS_VALUES="50 75 100" STREAM_VALUES="3 4 5 6 7 8" scripts/grid_search_gpu.sh
#
# Tip: tee to a results file:  scripts/grid_search_gpu.sh | tee results/grid_search_<stamp>.txt
BS_VALUES="${BS_VALUES:-25 50 75 100 125 150 200 250 320 400 500 625 750 1000 2000}"
STREAM_VALUES="${STREAM_VALUES:-2 3 5}"

# bs stays comptime (-D, re-keys the JIT cache) so it's the outer loop: each bs
# compiles once, then the inner streams loop reuses that binary via the runtime
# --num-streams arg (no recompile per stream value).
for bs in $BS_VALUES; do
  for streams in $STREAM_VALUES; do
    echo "=== bs=$bs streams=$streams ==="
    pixi run mojo -D GPU_STREAM_BATCH_SIZE=$bs src/main.mojo --bench-only --num-streams $streams 2>&1 | grep -E "batchedForward"
  done
done
