#!/usr/bin/env bash
# Restore default GPU clock behavior after benchmarking/profiling.
# Reverses scripts/gpu_lock.sh. Needs root.
#
#   Usage:  sudo scripts/gpu_unlock.sh [--no-persistence]
#   Force a vendor (skips autodetect):  GPU_VENDOR=amd sudo -E scripts/gpu_unlock.sh
#
# --no-persistence is NVIDIA-only (AMD has no equivalent knob); it is accepted
# and ignored on AMD so the same command works everywhere.
#
# The AMD side undoes all three lock paths unconditionally, in reverse order of
# how much state they touch. It does not try to work out which one gpu_lock.sh
# actually landed on — resetting a knob that was never set is harmless, and an
# unlock that only half-reverses is worse than one that over-reverses.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root: sudo scripts/gpu_unlock.sh" >&2
    exit 1
fi

detect_vendor() {
    if [[ -n "${GPU_VENDOR:-}" ]]; then echo "$GPU_VENDOR"; return; fi
    if command -v nvidia-smi >/dev/null 2>&1 \
       && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        echo nvidia; return
    fi
    if command -v rocm-smi >/dev/null 2>&1 && [[ -e /dev/kfd ]]; then
        echo amd; return
    fi
    echo unknown
}

unlock_nvidia() {
    nvidia-smi -rgc                  # reset graphics clock to default boost behavior

    if [[ "${1:-}" == "--no-persistence" ]]; then
        nvidia-smi -pm 0             # also turn persistence mode off
        echo "Reset clocks AND disabled persistence mode."
    else
        echo "Reset graphics clock to default. Persistence mode left ON"
        echo "(pass --no-persistence to also disable it)."
    fi

    nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,temperature.gpu --format=csv
}

unlock_amd() {
    local rc_smi="rocm-smi --autorespond y"

    $rc_smi --resetperfdeterminism >/dev/null 2>&1 || true  # if determinism was used
    $rc_smi --resetclocks          >/dev/null 2>&1 || true  # clocks + OverDrive to default
    $rc_smi --setperflevel auto    >/dev/null 2>&1 || true  # hand DPM back to the driver

    echo "Reset clocks, cleared determinism, performance level back to 'auto'."
    rocm-smi --showperflevel --showclocks 2>/dev/null \
        | grep -viE "^$|^====|^WARNING" || true
}

case "$(detect_vendor)" in
    nvidia) echo "Detected NVIDIA."; unlock_nvidia "${1:-}" ;;
    amd)    echo "Detected AMD (ROCm)."; unlock_amd ;;
    *)      echo "No supported GPU management tool found (need nvidia-smi or rocm-smi)." >&2
            exit 1 ;;
esac
