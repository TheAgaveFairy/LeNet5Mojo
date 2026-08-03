#!/usr/bin/env bash
# Lock the GPU into a steady state for reproducible benchmarking / profiling.
# Consumer cards (e.g. RTX 3070, RX 7600) boost + thermal/power throttle
# aggressively, which adds run-to-run timing variance. Pinning a SUSTAINABLE
# clock removes most of it. Needs root.
#
#   Usage:  sudo scripts/gpu_lock.sh [clock_mhz]
#   Undo:   sudo scripts/gpu_unlock.sh
#   Force a vendor (skips autodetect):  GPU_VENDOR=amd sudo -E scripts/gpu_lock.sh
#
# Why it measurably matters — five IDENTICAL runs on the RX 7600 with clocks
# UNLOCKED (bs=100, s=12) spanned 804k-911k fps, a 13% spread, while adjacent
# configs in a stream sweep differ by ~1%. Unlocked, the noise is an order of
# magnitude larger than the effect a knee search is trying to resolve.
#
# ── NVIDIA ────────────────────────────────────────────────────────────────────
# Default clock 1500 MHz — check valid values first:
#         nvidia-smi -q -d SUPPORTED_CLOCKS
#
# ── AMD ───────────────────────────────────────────────────────────────────────
# Not a straight port: nvidia-smi's `-lgc` pins an arbitrary MHz, but the
# equivalent AMD knobs (--setextremum, --setsclk at an arbitrary frequency) go
# through OverDrive, which many consumer cards do not expose. On the RX 7600 here
# `rocm-smi --showsclkrange` returns "get_od_volt, Not supported", and the DPM
# table has exactly two states (255 MHz idle / 2250 MHz), so there is no
# sub-maximum frequency to pin directly. Hence a cascade, best-first:
#
#   1. --setperfdeterminism MHZ  — purpose-built for this ("minimal performance
#      variation"), and the only option that caps BELOW max. Frequently
#      unsupported on consumer parts; we try it and check whether it took.
#   2. --setperflevel manual + --setsclk <top level>  — pins the top DPM state.
#   3. --setperflevel high  — coarsest, most portable.
#
# 2 and 3 pin at MAXIMUM, so unlike the NVIDIA path they leave no thermal
# headroom. If the card still throttles mid-run, pass an explicit lower MHz to
# force path 1, or lower the power cap (--setpoweroverdrive WATTS).
#
# Note: an idle AMD card sits in runtime-suspend ("low-power state" warning from
# rocm-smi). The first pass after idle pays the wake-up, which is a plausible
# source of the one-off `max=23ms` outlier seen in the variance runs above. Any
# benchmark should discard its first iteration regardless of clock locking.
set -euo pipefail

CLK="${1:-}"

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root: sudo scripts/gpu_lock.sh [mhz]" >&2
    exit 1
fi

# ── Vendor detection ──────────────────────────────────────────────────────────
# Presence of the CLI is not enough (a machine can have both installed, and the
# conda env ships nvidia bits), so require the tool to actually enumerate a GPU.
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

VENDOR="$(detect_vendor)"

# ── NVIDIA ────────────────────────────────────────────────────────────────────
lock_nvidia() {
    local clk="${CLK:-1500}"
    nvidia-smi -pm 1                  # persistence mode: keep driver resident
    nvidia-smi -lgc "${clk},${clk}"   # pin graphics clock to a fixed value

    echo "Locked graphics clock to ${clk} MHz, persistence ON."
    echo "Pick a clock the card sustains thermally — if it still throttles, lower it."
    nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,temperature.gpu,power.draw \
        --format=csv
}

# ── AMD ───────────────────────────────────────────────────────────────────────

# Top DPM state index from sysfs, e.g. "1" from "1: 2250Mhz". Falls back to 1.
top_sclk_level() {
    local f
    for f in /sys/class/drm/card*/device/pp_dpm_sclk; do
        [[ -r "$f" ]] || continue
        awk -F: '/^[0-9]+:/ { lvl = $1 } END { if (lvl != "") print lvl }' "$f"
        return
    done
    echo 1
}

amd_perf_level() {
    rocm-smi --showperflevel 2>/dev/null \
        | grep -oiE "Performance Level:[[:space:]]*[a-z]+" \
        | awk '{print tolower($NF)}' | head -1
}

lock_amd() {
    local rc_smi="rocm-smi --autorespond y"   # these prompt for confirmation

    if [[ -n "$CLK" ]]; then
        echo "Trying performance determinism at ${CLK} MHz..."
        if $rc_smi --setperfdeterminism "$CLK" >/dev/null 2>&1 \
           && [[ "$(amd_perf_level)" == "determinism" ]]; then
            echo "Locked via performance determinism at ${CLK} MHz."
            show_amd_state; return
        fi
        echo "  determinism not supported here (or did not take) — falling back." >&2
    fi

    local lvl; lvl="$(top_sclk_level)"
    echo "Pinning top DPM sclk state (level ${lvl})..."
    if $rc_smi --setperflevel manual >/dev/null 2>&1 \
       && $rc_smi --setsclk "$lvl" >/dev/null 2>&1 \
       && [[ "$(amd_perf_level)" == "manual" ]]; then
        echo "Locked to DPM sclk level ${lvl} (perf level: manual)."
        show_amd_state; return
    fi
    echo "  manual sclk pin failed — falling back to perf level 'high'." >&2

    if $rc_smi --setperflevel high >/dev/null 2>&1; then
        echo "Set performance level to 'high' (coarsest lock; clocks pinned at max)."
        show_amd_state; return
    fi

    echo "ERROR: could not lock clocks. Is amdgpu OverDrive/DPM control available?" >&2
    echo "       Some consumer cards need amdgpu.ppfeaturemask set on the kernel cmdline." >&2
    exit 1
}

show_amd_state() {
    echo "Pinned at MAXIMUM unless determinism took — watch for thermal throttling."
    rocm-smi --showperflevel --showclocks --showtemp --showpower 2>/dev/null \
        | grep -viE "^$|^====|^WARNING" || true
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$VENDOR" in
    nvidia) echo "Detected NVIDIA."; lock_nvidia ;;
    amd)    echo "Detected AMD (ROCm)."; lock_amd ;;
    *)      echo "No supported GPU management tool found (need nvidia-smi or rocm-smi)." >&2
            echo "Override detection with GPU_VENDOR=nvidia|amd if this is wrong." >&2
            exit 1 ;;
esac
