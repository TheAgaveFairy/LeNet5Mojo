# GPU kernel baseline — 2026-07-02

Config: bs=100, streams=5, ftype=float32, ReLU, clocks locked 1500 MHz (persistence on),
60k train images (601 launches/kernel incl. warmup), `pixi run nsysprofile_gpu`.

## Per-kernel (nsys cuda_gpu_kern_sum) — MEDIANS are the comparison numbers

| kernel           | time % | med (µs) | avg (µs) | min (µs) |
|------------------|--------|----------|----------|----------|
| conv2Fused       | 29.6   | 59.5     | 74.5     | 52.8     |
| conv3Fused       | 27.6   | 65.0     | 69.5     | 24.7     |
| maxPool1         | 12.7   | 22.6     | 32.0     | 3.9      |
| conv1Fused       | 9.8    | 19.2     | 24.6     | 16.5     |
| matMulFused      | 9.7    | 20.6     | 24.5     | 5.5      |
| normalizeInputs  | 6.2    | 12.0     | 15.7     | 3.1      |
| maxPool2         | 4.4    | 5.8      | 11.0     | 3.1      |

Max/stddev columns are inflated by multi-stream contention (maxes 0.6–1.0 ms);
ignore them for kernel-vs-kernel comparisons. memcpy: H2D 4.4 µs avg ×615,
D2H 1.4 µs ×601.

## End-to-end (main --bench-only, median of 10 passes, 10k test images)

- Config grid bs ∈ {50,100,200,500,1000} × streams ∈ {3,4,5,6,8}: best cells
  ~880–950k fps, bs=100 row strongest, bs=1000 clearly worse at low streams.
- **Run-to-run variance dominates**: bs=100 s=5 repeated ×5 spanned
  676k–926k fps (±16%) despite locked clocks and per-run median-of-10.
  Configs within (bs 50–500) × (s 4–6) are statistically indistinguishable
  with one run each. Default bs=100 s=5 stands.

## Methodology rules going forward

E2e fps is the project metric — it's what gets benchmarked against JAX / ONNX-RT /
PyTorch. The rules below are about *signal quality per question*, not priority.

1. **Developing a kernel change** → nsys per-kernel median is the signal.
   E2e can't resolve it: a 20% win on a 28%-share kernel is ~5.5% e2e, under
   the ±16% single-run noise floor, and multi-stream overlap hides kernel
   deltas entirely until the kernel is the critical path (more streams = more
   hiding — "best at higher streams" is a pipeline fact, not a kernel fact).
2. **Accepting a change / quoting a number** → e2e fps, but earn it: ≥5
   process reps (min or mean-of-mins), or lengthen the timed region
   (`-D N_PASSES=50`); the current ~10 ms/pass region is scheduler-noise prey.
   Every framework-comparison number follows this rule (and same-precision:
   disable TF32 in torch/JAX or state it).
3. Sweep stream count at runtime (`--num-streams`), batch size needs
   `-D GPU_STREAM_BATCH_SIZE` rebuild.
