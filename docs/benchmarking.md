# Benchmarking protocol & record format

The durable methodology for every performance number that ends up in `ideas.typ`
(the blog / conference writeup). The goal is **attribution**: each number must
trace to an exact code version, a known clock state, and a named experiment step,
so deltas can be decomposed cleanly (e.g. "SoA layout alone: +X%; GEMM on top:
+Y%").

## Why the old artifacts aren't enough

- Per-run CSVs (`results/gpu_infer_bs*_run=*.csv`) record only median `elapsed_ns`
  — no min/max, no git SHA, no clock-lock state, no experiment label.
- `results/stream_sweep_*.txt` are human-readable console dumps — not programmatic.

Neither lets you say "this number is from commit abc123, clocks locked, the SoA
step." That attribution is the whole point.

## Source of truth: `results/benchmarks.csv` (append-only master log)

One row per measurement (device × batch_size × num_streams × step). Append-only —
the entire experiment history lives in one file, loads straight into
pandas/typst for tables and plots.

### Schema

| column | source | notes |
|--------|--------|-------|
| `timestamp` | auto | run time |
| `step` | CLI `--bench-step` | `baseline` \| `soa` \| `reorder` \| `gemm` \| … (the variable under test) |
| `git_sha` | auto (`git rev-parse --short HEAD`) | exact code version |
| `git_dirty` | auto (`git status --porcelain` non-empty → 1) | **headline numbers require 0** |
| `device` | auto | `CPU` \| `GPU` |
| `gpu_name` | auto (`ctx.name()`) | empty for CPU |
| `clocks_locked` | CLI `--clocks-locked` | the trust gate (see below) |
| `mojo_version` | auto (`mojo --version`) | toolchain drift |
| `dtype` | auto | e.g. `Float32` |
| `activation` | auto | e.g. `ReLU` |
| `scenario` | const | `streaming` (headline) \| `resident` (future) |
| `norm_mode` | const | `per_image` (current) \| `const` (parity, future) |
| `batch_size` | auto | stream batch size |
| `num_streams` | auto | |
| `eff_batch` | auto | `batch_size * num_streams` |
| `n_warmup` | auto | `N_WARMUP` |
| `n_passes` | auto | `N_PASSES` |
| `count` | auto | images processed (`n_proc`) |
| `correct` | auto | the 9648 oracle |
| `accuracy_pct` | auto | `correct/count` |
| `median_ns` | auto (`_timing_stats`) | central metric |
| `min_ns` | auto | |
| `max_ns` | auto | |
| `spread_pct` | auto | `(max-min)/median*100` — variance red-flag |
| `ns_per_img` | auto | `median_ns/count` |
| `fps` | auto | `count*1e9/median_ns` |

Auto = captured by the harness; CLI = a flag the operator sets; const = fixed in
code until a future feature makes it variable.

## Fixed control config (comparability)

Hold everything constant except the variable under test:

- `batch_size = 50` — divides 10,000 and 60,000 → full dataset coverage, no
  dropped tail. (Differs from the shipped default of 100; set `-D
  GPU_STREAM_BATCH_SIZE=50` for these runs.)
- Run **both** `--num-streams 5` (tuned default) **and** `--num-streams 1`
  (fair single-stream — the honesty number vs single-stream libraries). Report
  both; never quote only the tuned one.
- `N_PASSES >= 20` for headline runs (tighter median), `N_WARMUP = 3`.
- `dtype = Float32`, `scenario = streaming`, `norm_mode = per_image`.

State every one of these next to any number quoted in the writeup.

## A/B protocol (run at each checkpoint: baseline → soa → reorder → gemm)

1. **Commit the change first.** A headline number requires a clean tree
   (`git_dirty = 0`). Dirty-tree runs are exploratory only — fine to take, but
   tag them mentally as not-for-publication.
2. **Lock clocks once per session:** `pixi run gpulock` (sudo, real terminal).
   Then pass `--clocks-locked` so the row records it. Restore later with
   `pixi run gpuunlock`. The consumer 3070 boosts/throttles → ~10% jitter that
   swamps a small layout delta. Locking is non-negotiable for published numbers.
3. **Same-session A/B.** Build + run the *baseline* commit, then build + run the
   *new* commit, back-to-back, same clocks, same session. Both rows append to
   `benchmarks.csv`, distinguished by `step` + `git_sha`. Same-session is the
   gold standard.
4. **Reject high-variance rows.** If `spread_pct` is large, re-run. Never compare
   a warm run to a cold one (a prior pool-rewrite measurement got burned exactly
   this way — the recorded baseline was warm/unlocked at different conditions, so
   only the same-day A/B delta was real).
5. **Cross-session comparison** is allowed *only* if both rows have
   `clocks_locked = 1` on the same `gpu_name`. Otherwise re-measure in one
   session.

## Per-session notes

Keep a short human log in `results/bench_notes.md` (free-form): what changed this
session, any anomalies, the gpulock clock value used, hardware temp/conditions if
notable. The CSV is the data; this is the context.

## Harness changes required (to auto-capture the new columns)

Currently `ResultLogger.logInferenceResult` writes the old per-run schema. To emit
the master log:

- Add a `--bench-step <label>` and `--clocks-locked` flag to `cli.mojo`
  (`CliArgs.parse` + `printHelp`).
- Capture `git_sha` / `git_dirty` / `mojo_version` via `std.subprocess.run`
  (prefer subprocess over Python interop for these shell-outs).
- Extend the logger to append a full row (all columns above) to
  `results/benchmarks.csv`, pulling `median/min/max` from the existing
  `_timing_stats`, and `n_warmup`/`n_passes`/`count`/`correct` already in scope
  in `runGPUTest` (and the CPU bench path).
- Keep the per-run CSVs if desired, but treat `benchmarks.csv` as canonical.
