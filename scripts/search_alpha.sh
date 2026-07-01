#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# act_fn × ALPHA (learning-rate) search.
#
# Per activation: coarse log sweep (phase 1) → linear refine around the peak
# (phase 2) → FINE refine around that peak (phase 3, recovers precision the
# step-10 grid loses). Runs are deduped across phases via the CSV.
#
# Default flow (no args): full search for the PRIMARY activations, then the GELU
# VARIANTS as phase-2/3 only, centered on plain GELU's best (they share a regime,
# so a fresh coarse sweep each is wasteful). Sigmoid is intentionally excluded —
# it collapses under the default short schedule (see docs/activation_tuning.md).
#
# Run it (mojo must be on PATH → go through pixi):
#   pixi run bash scripts/search_alpha.sh                 # default flow
#   pixi run bash scripts/search_alpha.sh GELU            # one activation, full
#   CENTER_ALPHA=700 pixi run bash scripts/search_alpha.sh GELUFast   # phase 2/3 only
#
# Knobs (env-overridable): PHASE2_HALF_WIDTH PHASE2_STEP PHASE3_HALF_WIDTH
# PHASE3_STEP CENTER_ALPHA.
# ══════════════════════════════════════════════════════════════════════════════

# All known activation functions. Add new ones here; they become valid args.
ALL_ACT_FNS=(ReLU GELU GELUFast GELUTanh Sigmoid Tanh)

# Default-flow split: PRIMARY get a full search; VARIANTS reuse GELU's best.
PRIMARY=(ReLU GELU Tanh)
VARIANTS=(GELUTanh GELUFast)

# Mojo invocation. This script is meant to be launched via `pixi run bash ...`,
# so `mojo` is already on PATH — no `pixi run` prefix here.
MOJO_CMD="mojo"
MOJO_FILE="src/main.mojo"

# CSV output
CSV_SEP=","
OUTDIR="results"

# Phase 1: coarse log-scale sweep (integer ALPHA values, 1–1000)
PHASE1_POINTS=(8 32 64 128 256 384 512 768 1000)

# Phase 2: linear sweep around the phase 1 peak
PHASE2_HALF_WIDTH="${PHASE2_HALF_WIDTH:-120}"   # ± this many ALPHA units around the peak
PHASE2_STEP="${PHASE2_STEP:-10}"                # step size

# Phase 3: FINE sweep around the phase 2 peak — recovers the precision the
# step-10 phase-2 grid can't resolve (the "lost perfection").
PHASE3_HALF_WIDTH="${PHASE3_HALF_WIDTH:-16}"
PHASE3_STEP="${PHASE3_STEP:-4}"

# If set, SKIP phase 1 for every activation and center phase 2 on this ALPHA.
CENTER_ALPHA="${CENTER_ALPHA:-}"

# ══════════════════════════════════════════════════════════════════════════════

mkdir -p "$OUTDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="${OUTDIR}/results_${TIMESTAMP}.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

# Extract the value of a key=VALUE field from a CSV-style output line.
parse_field() {
    local line="$1" key="$2"
    echo "$line" | grep -oP "${key}=\K[^,]+"
}

# Best ALPHA (highest correct) recorded so far for an activation, from the CSV.
best_from_csv() {
    awk -F"$CSV_SEP" -v fn="$1" '
        NR > 1 && $1 == fn && $4 ~ /^[0-9]+$/ { if ($4 + 0 > max) { max = $4 + 0; ba = $2 } }
        END { print ba }
    ' "$OUTFILE"
}

# Build a clamped [1,1000] integer point list: build_points HALF STEP CENTER.
build_points() {
    local half="$1" step="$2" center="$3"
    local start=$(( center - half )) end=$(( center + half )) a
    (( start < 1 )) && start=1
    (( end > 1000 )) && end=1000
    local pts=()
    for (( a = start; a <= end; a += step )); do pts+=("$a"); done
    echo "${pts[*]}"
}

# Run one (act_fn, alpha_int) pair. Writes one CSV row and sets LAST_CORRECT.
# Dedups: if the pair is already in the CSV, reuse it instead of re-training.
LAST_CORRECT=-1
run_alpha() {
    local act_fn="$1" alpha_int="$2"
    LAST_CORRECT=-1

    local existing
    existing=$(awk -F"$CSV_SEP" -v fn="$act_fn" -v a="$alpha_int" \
        'NR > 1 && $1 == fn && $2 == a { print $4; exit }' "$OUTFILE")
    if [[ -n "$existing" ]]; then
        LAST_CORRECT="$existing"
        printf '  [cached] %s ALPHA=%s correct=%s\n' "$act_fn" "$alpha_int" "$existing" >&2
        return
    fi

    local output exit_code=0
    output=$("$MOJO_CMD" "-D${act_fn}" "-DALPHA=${alpha_int}" "$MOJO_FILE" 2>/dev/null) \
        || exit_code=$?

    if [[ $exit_code -ne 0 ]] || [[ -z "$output" ]]; then
        local alpha_float
        alpha_float=$(awk "BEGIN { printf \"%.3f\", ${alpha_int}/1000 }")
        local row="${act_fn}${CSV_SEP}${alpha_int}${CSV_SEP}${alpha_float}${CSV_SEP}-1${CSV_SEP}-1${CSV_SEP}-1${CSV_SEP}-1${CSV_SEP}error"
        printf '%s\n' "$row" | tee -a "$OUTFILE"
        printf '  [ERROR] %s ALPHA=%s exited with code %s\n' "$act_fn" "$alpha_int" "$exit_code" >&2
        return
    fi

    local alloc correct total_count training_ms testing_ms alpha_float
    alloc=$(parse_field "$output" "alloc")
    correct=$(parse_field "$output" "correct")
    total_count=$(parse_field "$output" "total_count")
    training_ms=$(parse_field "$output" "training_ms")
    testing_ms=$(parse_field "$output" "testing_ms")
    alpha_float=$(parse_field "$output" "ALPHA")

    local row="${act_fn}${CSV_SEP}${alpha_int}${CSV_SEP}${alpha_float}${CSV_SEP}${correct}${CSV_SEP}${total_count}${CSV_SEP}${training_ms}${CSV_SEP}${testing_ms}${CSV_SEP}${alloc}"
    printf '%s\n' "$row" | tee -a "$OUTFILE"

    LAST_CORRECT="${correct:--1}"
}

# Sweep a list of ALPHA integers for one activation function.
sweep() {
    local act_fn="$1"; shift
    local alpha_int
    for alpha_int in "$@"; do
        run_alpha "$act_fn" "$alpha_int"
    done
}

banner() {
    printf '══════════════════════════════════════════════\n'
    printf '  %s\n' "$1"
    printf '══════════════════════════════════════════════\n'
}

# Full or centered search for one activation.
# process ACT_FN [CENTER]  — CENTER present ⇒ skip phase 1.
process() {
    local act_fn="$1" center="${2:-}"
    banner "$act_fn"

    if [[ -z "$center" ]]; then
        printf '  Phase 1: coarse log sweep\n'
        sweep "$act_fn" "${PHASE1_POINTS[@]}"
        center=$(best_from_csv "$act_fn")
    else
        printf '  Phase 1 skipped; centering on ALPHA=%s\n' "$center"
    fi

    printf '  Phase 2: ±%s step %s around ALPHA=%s\n' "$PHASE2_HALF_WIDTH" "$PHASE2_STEP" "$center"
    sweep "$act_fn" $(build_points "$PHASE2_HALF_WIDTH" "$PHASE2_STEP" "$center")
    center=$(best_from_csv "$act_fn")

    printf '  Phase 3: FINE ±%s step %s around ALPHA=%s\n' "$PHASE3_HALF_WIDTH" "$PHASE3_STEP" "$center"
    sweep "$act_fn" $(build_points "$PHASE3_HALF_WIDTH" "$PHASE3_STEP" "$center")
    printf '\n'
}

# ── Validate explicit arguments ────────────────────────────────────────────────

if [[ $# -gt 0 ]]; then
    for fn in "$@"; do
        valid=0
        for known in "${ALL_ACT_FNS[@]}"; do
            [[ "$fn" == "$known" ]] && valid=1 && break
        done
        if (( valid == 0 )); then
            printf 'Unknown activation function: %s\nKnown: %s\n' "$fn" "${ALL_ACT_FNS[*]}" >&2
            exit 1
        fi
    done
fi

# ── Initialise output file ─────────────────────────────────────────────────────

printf '%s\n' "act_fn${CSV_SEP}alpha_int${CSV_SEP}alpha_float${CSV_SEP}correct${CSV_SEP}total_count${CSV_SEP}training_ms${CSV_SEP}testing_ms${CSV_SEP}alloc" \
    > "$OUTFILE"

printf 'Writing results to: %s\n\n' "$OUTFILE"

# ── Main ───────────────────────────────────────────────────────────────────────

if [[ $# -gt 0 ]]; then
    # Explicit activations: full search each (honor CENTER_ALPHA if set).
    for fn in "$@"; do process "$fn" "$CENTER_ALPHA"; done
else
    # Default flow: PRIMARY full, then VARIANTS centered on GELU's best.
    for fn in "${PRIMARY[@]}"; do process "$fn" "$CENTER_ALPHA"; done

    gelu_best=""
    [[ -z "$CENTER_ALPHA" ]] && gelu_best=$(best_from_csv "GELU")
    center_for_variants="${CENTER_ALPHA:-$gelu_best}"

    for fn in "${VARIANTS[@]}"; do process "$fn" "$center_for_variants"; done
fi

# ── Summary (derived from CSV so it reflects all phases) ───────────────────────

printf '══════════════════════════════════════════════\n'
printf '  Summary (best ALPHA per activation)\n'
printf '══════════════════════════════════════════════\n'

summarised=("$@")
(( ${#summarised[@]} == 0 )) && summarised=("${PRIMARY[@]}" "${VARIANTS[@]}")

for act_fn in "${summarised[@]}"; do
    best_row=$(awk -F"${CSV_SEP}" -v fn="$act_fn" '
        NR > 1 && $1 == fn && $4 ~ /^[0-9]+$/ {
            if ($4 + 0 > max) { max = $4 + 0; row = $0 }
        }
        END { print row }
    ' "$OUTFILE")

    if [[ -n "$best_row" ]]; then
        best_alpha_int=$(printf '%s' "$best_row" | cut -d"${CSV_SEP}" -f2)
        best_alpha_f=$(printf '%s' "$best_row"   | cut -d"${CSV_SEP}" -f3)
        best_correct=$(printf '%s' "$best_row"   | cut -d"${CSV_SEP}" -f4)
        printf '  %-12s  best ALPHA=%s (%s)  correct=%s\n' \
            "$act_fn" "$best_alpha_int" "$best_alpha_f" "$best_correct"
    else
        printf '  %-12s  no successful runs\n' "$act_fn"
    fi
done

printf '\nResults saved to: %s\n' "$OUTFILE"
