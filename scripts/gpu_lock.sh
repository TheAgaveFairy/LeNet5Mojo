#!/usr/bin/env bash
# Lock the GPU into a steady state for reproducible benchmarking / profiling.
# Consumer cards (e.g. RTX 3070) boost + thermal/power throttle aggressively,
# which adds run-to-run timing variance. Pinning a SUSTAINABLE graphics clock
# (below max boost, so it won't throttle mid-run) + persistence mode removes most
# of it. Needs root.
#
#   Usage:  sudo scripts/gpu_lock.sh [graphics_clock_mhz]
#   Default clock 1500 MHz — check valid values first:
#           nvidia-smi -q -d SUPPORTED_CLOCKS
#   Undo:   sudo scripts/gpu_unlock.sh
set -euo pipefail

CLK="${1:-1500}"

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root: sudo scripts/gpu_lock.sh [mhz]" >&2
    exit 1
fi

nvidia-smi -pm 1                  # persistence mode: keep driver resident
nvidia-smi -lgc "${CLK},${CLK}"   # pin graphics clock to a fixed value

echo "Locked graphics clock to ${CLK} MHz, persistence ON."
echo "Pick a clock the card sustains thermally — if it still throttles, lower it."
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,temperature.gpu,power.draw \
    --format=csv
