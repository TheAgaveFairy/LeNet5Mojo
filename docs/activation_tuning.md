# Activation function tuning — suggested defaults

Starting points for **which learning rate (`ALPHA`) to pair with each activation
function**. The optimal `ALPHA` is activation-dependent (that's the act_fn ×
learning-rate interaction), so a value that trains ReLU well can stall or diverge
GELU/Sigmoid.

Pick an activation with a bare `-D` flag and a learning rate with `-D ALPHA=N`
(integer, interpreted as `N / 1000`; default `500` → `0.5`):

```bash
pixi run mojo -D GELUFast -D ALPHA=300 src/main.mojo   # example
```

## Suggested combos

> ⚠️ **Preliminary.** Most rows are TBD. The numbers present are single-run,
> tuned on the test set, and not from a rigorous search — treat them as rough
> starting points, not recommendations. The real table comes from the improved
> `scripts/search_alpha.sh` sweep (multi-seed averaging + adaptive refinement +
> a validation split — see TODO.md "act_fn × ALPHA search").

| Activation | `-D` flag     | Suggested `ALPHA` (int → float) | Best observed acc | Notes / source |
|------------|---------------|---------------------------------|-------------------|----------------|
| ReLU       | *(default)*   | `500`–`1000` → 0.5–1.0          | ~9691–9721/10000  | single-run: α=50→8726, α=500→9691, α=1000→9721 (2026-06-30) |
| GELU       | `-D GELU`     | TBD                             | TBD               | exact erf form |
| GELUTanh   | `-D GELUTanh` | TBD                             | TBD               | tanh approximation |
| GELUFast   | `-D GELUFast` | `500` → 0.5 *(unconfirmed)*     | ~9003/10000       | single short train (2026-06-30), not converged |
| Sigmoid    | `-D Sigmoid`  | ⚠️ none found                   | ~1028–1801/10000  | **COLLAPSES** — see warning below |
| Tanh       | `-D Tanh`     | TBD                             | TBD               | |

> ⚠️ **Sigmoid collapses (~10–18%, near chance) under the default short schedule**
> at every `ALPHA` tried (50 / 500 / 1000 → 1748 / 1028 / 1801 correct). Presumed
> vanishing gradients through this depth with a saturating activation, not a
> learning-rate that just needs nudging. Needs a different setup (more epochs,
> different init/schedule) before an `ALPHA` recommendation is meaningful — don't
> read the numbers above as "Sigmoid's best".

## How to (re)generate this table

> **Status:** `scripts/search_alpha.sh` is prepped and validated but **not yet run
> in anger** — the table above is still preliminary. Running the full search is the
> next step (it's slow: each `(act, ALPHA)` point recompiles + trains, ~35 s, so a
> full multi-phase multi-activation sweep is roughly **1.5–2 hours**). Do it on a
> free machine, then fill in the rows.

The harness does a 3-phase search per activation:

1. **Phase 1** — coarse log sweep over `ALPHA` (integer, `/1000`): `8 … 1000`.
2. **Phase 2** — linear refine `±PHASE2_HALF_WIDTH` (default 120) step `PHASE2_STEP`
   (10) around the phase-1 peak.
3. **Phase 3** — FINE refine `±PHASE3_HALF_WIDTH` (default 16) step `PHASE3_STEP`
   (4) around the phase-2 peak. This is what recovers the precision the step-10
   phase-2 grid can't resolve (the earlier "lost perfection" at `HALF_WIDTH=100`).

Runs are **deduped across phases** via the CSV, and results land in `results/results_<ts>.csv`.

```bash
# Default flow: full search for ReLU, GELU, Tanh, then the GELU VARIANTS
# (GELUTanh, GELUFast) as phase-2/3 only, centered on plain GELU's best.
# Sigmoid is intentionally excluded (it collapses — see warning above).
pixi run bash scripts/search_alpha.sh

# One activation, full 3-phase search:
pixi run bash scripts/search_alpha.sh GELU

# Phase-2/3 only, centered on a known-good ALPHA (skips the coarse sweep):
CENTER_ALPHA=700 pixi run bash scripts/search_alpha.sh GELUFast

# Tune the search resolution:
PHASE2_HALF_WIDTH=150 PHASE3_STEP=2 pixi run bash scripts/search_alpha.sh Tanh
```

Read the printed **Summary** (best ALPHA per activation) or the CSV, then update
the rows above. State the seed (runs are deterministic at `--seed 42`), the split
used (currently test set — a validation split is the planned improvement, see
TODO), and the git SHA alongside any number promoted from "preliminary" to
"suggested". For real rigor, average a few seeds per point (the search is
single-seed today; multi-seed averaging is in the TODO rewrite).
