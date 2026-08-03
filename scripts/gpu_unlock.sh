#!/usr/bin/env bash
# Restore default GPU clock behavior after benchmarking/profiling.
# Reverses scripts/gpu_lock.sh. Needs root.
#
#   Usage:  sudo scripts/gpu_unlock.sh [--no-persistence]
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root: sudo scripts/gpu_unlock.sh" >&2
    exit 1
fi

nvidia-smi -rgc                  # reset graphics clock to default boost behavior

if [[ "${1:-}" == "--no-persistence" ]]; then
    nvidia-smi -pm 0             # also turn persistence mode off
    echo "Reset clocks AND disabled persistence mode."
else
    echo "Reset graphics clock to default. Persistence mode left ON"
    echo "(pass --no-persistence to also disable it)."
fi

nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,temperature.gpu --format=csv
