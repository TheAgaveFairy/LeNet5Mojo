#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Configuration — all tuneable knobs live here
# ══════════════════════════════════════════════════════════════════════════════

# All known activation functions. Add new ones here; they become valid args.
ALL_ACT_FNS=(ReLU GELU GELUFast GELUTanh Sigmoid Tanh)

# Mojo invocation
MOJO_CMD="mojo"
MOJO_FILE="main.mojo"

# CSV output
CSV_SEP=","
OUTDIR="."

# Phase 1: coarse log-scale sweep (integer ALPHA values, 1–1000)
PHASE1_POINTS=(8 32 64 128 256 384 512 768 1000)

# Phase 2: linear sweep around the phase 1 peak
PHASE2_HALF_WIDTH=40   # sweep ± this many ALPHA units around the peak
PHASE2_STEP=10         # step size

# ══════════════════════════════════════════════════════════════════════════════

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTFILE="${OUTDIR}/results_${TIMESTAMP}.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

# Extract the value of a key=VALUE field from a CSV-style output line
parse_field() {
    local line="$1" key="$2"
    echo "$line" | grep -oP "${key}=\K[^,]+"
}

# Run one (act_fn, alpha_int) pair. Writes one CSV row and sets LAST_CORRECT.
LAST_CORRECT=-1
run_alpha() {
    local act_fn="$1" alpha_int="$2"
    LAST_CORRECT=-1

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
# Sets LAST_BEST_ALPHA to whichever point had the highest correct count.
LAST_BEST_ALPHA=-1
sweep() {
    local act_fn="$1"; shift
    local points=("$@")
    local best_alpha=-1 best_correct=-1

    for alpha_int in "${points[@]}"; do
        run_alpha "$act_fn" "$alpha_int"
        if [[ "$LAST_CORRECT" =~ ^[0-9]+$ ]] && (( LAST_CORRECT > best_correct )); then
            best_correct=$LAST_CORRECT
            best_alpha=$alpha_int
        fi
    done

    LAST_BEST_ALPHA=$best_alpha
}

# ── Validate arguments ─────────────────────────────────────────────────────────

if [[ $# -gt 0 ]]; then
    ACT_FNS=("$@")
else
    ACT_FNS=("${ALL_ACT_FNS[@]}")
fi

for fn in "${ACT_FNS[@]}"; do
    valid=0
    for known in "${ALL_ACT_FNS[@]}"; do
        [[ "$fn" == "$known" ]] && valid=1 && break
    done
    if (( valid == 0 )); then
        printf 'Unknown activation function: %s\nKnown: %s\n' "$fn" "${ALL_ACT_FNS[*]}" >&2
        exit 1
    fi
done

# ── Initialise output file ─────────────────────────────────────────────────────

printf '%s\n' "act_fn${CSV_SEP}alpha_int${CSV_SEP}alpha_float${CSV_SEP}correct${CSV_SEP}total_count${CSV_SEP}training_ms${CSV_SEP}testing_ms${CSV_SEP}alloc" \
    > "$OUTFILE"

printf 'Writing results to: %s\n\n' "$OUTFILE"

# ── Main loop ──────────────────────────────────────────────────────────────────

for act_fn in "${ACT_FNS[@]}"; do
    printf '══════════════════════════════════════════════\n'
    printf '  %s — Phase 1: coarse log sweep\n' "$act_fn"
    printf '══════════════════════════════════════════════\n'

    sweep "$act_fn" "${PHASE1_POINTS[@]}"
    p1_best=$LAST_BEST_ALPHA

    printf '\n'
    printf '══════════════════════════════════════════════\n'
    printf '  %s — Phase 2: linear sweep around ALPHA=%s\n' "$act_fn" "$p1_best"
    printf '══════════════════════════════════════════════\n'

    # Clamp sweep range to [1, 1000]
    p2_start=$(( p1_best - PHASE2_HALF_WIDTH ))
    p2_end=$(( p1_best   + PHASE2_HALF_WIDTH ))
    (( p2_start < 1    )) && p2_start=1
    (( p2_end   > 1000 )) && p2_end=1000

    # Build phase 2 point list, skipping points already run in phase 1
    phase2_points=()
    for (( a = p2_start; a <= p2_end; a += PHASE2_STEP )); do
        skip=0
        for p1 in "${PHASE1_POINTS[@]}"; do
            (( a == p1 )) && skip=1 && break
        done
        (( skip == 0 )) && phase2_points+=("$a")
    done

    sweep "$act_fn" "${phase2_points[@]}"
    printf '\n'
done

# ── Summary (derived from CSV so it reflects both phases) ─────────────────────

printf '══════════════════════════════════════════════\n'
printf '  Summary\n'
printf '══════════════════════════════════════════════\n'

for act_fn in "${ACT_FNS[@]}"; do
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
