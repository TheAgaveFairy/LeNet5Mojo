# LeNet5Mojo — TODO / FIXME Checklist

Generated from all TODO and FIXME comments across the repo, plus design review notes.
Check items off as they are completed.

---

## Architecture / Design

- [ ] **Refactor `batchedForward` to take one pre-sliced batch** (`accel/ops.mojo`)
  - Currently takes the full dataset and loops internally. Should take exactly `batch_size` images.
  - Move the loop to the caller in `main.mojo`.
  - Lets the caller own the GPU pixel staging buffer (allocated once, reused), enables ping-pong
    streaming (submit next H2D copy before syncing current compute), and makes the function
    reusable for training loops with per-batch logic.
  - Suggested caller pattern: `for i in range(0, total, batch_size): batchedForward(ctx, data.slice(i, batch_size), ...)`

- [x] **`MNISTBatch` lifetime not tracked — callers must `keep(data_repo)`** (`dataloader.mojo`, `profile_gpu.mojo`)
  - `MNISTBatch` now parameterized on `origin: ImmutOrigin`. `getTrainBatch`/`getTestBatch` return
    `MNISTBatch[origin=origin_of(self)]` via `rebind` on the arena's `MutAnyOrigin` pointer.
    Borrow checker enforces repo outlives the batch. `keep(data_repo)` removed; accuracy verified.

- [x] **Rename `MNISTBatch` — name is misleading when used as a full-dataset view** (`dataloader.mojo`)
  - Renamed to `MNISTDataView`. Added `__getitem__(start, end)` returning a sub-view.
    Updated all call sites in `accel/ops.mojo`, `tests/profile_gpu.mojo`.

- [x] **Define `AcceptsAllocator` trait** (`accel/model.mojo:54`, `accel/model.mojo:91`)
  - Added `ArenaSizable` trait to `cpu/arena.mojo` (`@staticmethod sizeInBytes() -> Int`).
    Renamed `_calcArenaSize` → `sizeInBytes` on CPU structs. `LeNet5`, `Feature`,
    `FeatureGPUBuffers`, `LeNet5GPUBuffers`, `LeNet5GPU` all conform.

- [ ] **Centralize weight + feature layouts into structs in `constants.mojo`** (`constants.mojo`, `accel/ops.mojo`, `accel/feature.mojo`)
  - Today the per-layer dims (K/M of each weight, feature flat sizes) are re-derived inline in each
    kernel (e.g. conv3 flattening `LAYER4*5*5=400`, weight4_5 as `[400,120]`, weight5_6 as `[120,10]`).
  - Hoist into named layout structs so the GEMM-style ops (conv3, matmul5_6) can read K, M, and the
    `Layout` straight off a shared definition instead of open-coding shapes. Makes the generic
    `gemm[B,K,M,...]` reuse clean and keeps CPU/GPU shape definitions in one place.

- [ ] **(Future) Kill `List[Image]` from CPU hot path; share SoA spans for both CPU and GPU** (`dataloader.mojo`)
  - CPU path currently uses `List[Image]` (AoS, 785-byte stride with label interleaved).
  - GPU path already uses SoA pixel/label arenas exposed via `MNISTBatch`.
  - If CPU profiling shows packing cost, migrate CPU ops to consume `raw_pixels`/`raw_labels` spans
    directly. `Image` becomes a lightweight view into the arena rather than an owned struct.

---

## GPU Pipeline

- [x] **Ping-pong streaming: overlap H2D copy with compute** (`accel/ops.mojo`, `main.mojo`)
  - Already implemented via `batchedForwardMultiStream` with configurable `NUM_GPU_STREAMS`.

- [x] **Wire up GPU inference logger** (`main.mojo:281`)
  - `logger.logInferenceResult(device_name, elapsed, correct, COUNT_TRAIN, batch_size, ftype)`
    is commented out after `batchedForward`. Uncomment and hook up.

- [x] **`singleForward`: `gpu_guess` sentinel should be `Optional[Int]`** (`accel/ops.mojo:639`) — `singleForward` removed
  - `var gpu_guess = 10` uses magic number 10 as "invalid". Typed `Optional` is cleaner.

- [x] **Confirm matmul output: skip or apply `act_fn.simdForward()`?** (`accel/ops.mojo:160`)
  - Decision: skip activation after final FC layer (raw logits → softmax/argmax at output). Both CPU and GPU consistent.

- [x] **Kernel wrapper functions may be unnecessary now that `ctx` is passed** (`accel/ops.mojo:76`)
  - `conv1Forward`, `normalizeInputs`, etc. are thin wrappers around `ctx.enqueue_function`.
  - Consider force-inlining or calling kernels directly. Evaluate if the abstraction is worth keeping.

- [ ] **conv2 image loading could be more efficient** (`accel/ops.mojo:406`)
  - Comment: "could make this much more efficient." Candidate for shared-mem load optimization.

- [x] **Profile conv2 with ncu — it's now the #1 kernel (29.5%)** (`accel/ops.mojo`) — DONE 2026-06-08
  - Verdict: NO easy win. conv2 is L1/TEX-bound (91.5% SOL) but on LEGITIMATE reuse (weights
    reused across 100 spatial outputs/channel, image across output channels) — not a wasteful
    reduction we can delete. Already per-thread register accum ~3.0 ps/MAC, occupancy 62% (theo
    81%, capped by 38 reg/thread). Tail (1.45 waves) already mitigated by the streams 3->5 bump.
    Only real lever = register tiling / thread coarsening (Tier-B-class rewrite, medium effort,
    low ROI now). DEPRIORITIZED. Full read: ignoreme/ncu_conv2_notes.md.

- [ ] **(LAST PRIORITY) conv3 Tier B — tiled GEMM for single-stream occupancy** (`accel/ops.mojo`)
  - Tier A works but under-occupies (ncu: 8.8% occupancy, 0.11 waves/SM — 50 blocks of 120
    threads can't fill the GPU). It leans on stream concurrency to hit peak throughput.
  - Tier B = real tiled GEMM mapping batch×M onto MANY warp-multiple blocks, so one launch fills
    the GPU on its own (high single-stream throughput, less reliance on 5+ streams). See the
    GEMM design notes / `ignoreme/conv3_tierA_writeup.md` §4 caveat + §7.
  - Explicitly deferred: do this ONLY after conv2 is understood and the rest of this list is done.

- [ ] **Full image coverage for ANY batch size (pad the tail)** (`accel/ops.mojo`, `main.mojo`)
  - Today eff_batch must divide the dataset or the remainder images are dropped (e.g. bs=75 →
    eff 375, 10000 % 375 = 25 dropped → 9975/10000). That's why the default is pinned to bs=50
    (divides 10000 & 60000). Want: any batch size covers all 10k/60k.
  - Approach: pad the final partial batch to full `eff_batch` with zeros (pixels and/or a masked
    label), run it, then ignore the padded slots when tallying correct/total. Keeps the kernels
    fixed-size (no special last-batch path) and frees batch size to be a pure perf knob.
  - Mirrors the CPU-side remainder handling already done in `testingParallel` (cpu/ops.mojo).

- [ ] **Auto-heuristic for `num_streams` from compile-time batch size** (`main.mojo`, `cli.mojo`)
  - Today the user must hand-tune `--num-streams` per `GPU_STREAM_BATCH_SIZE` to hit peak (grid
    search). Most standard libraries (PyTorch/ORT eager) use a single stream and just take a
    `batch_size` — so Mojo looks harder to drive. Want a sensible runtime default so good numbers
    come out-of-the-box, while keeping the grid search for per-card peak tuning.
  - Approach: when `--num-streams` is unset, pick `streams = clamp(round(TARGET_EFF / bs), 1, 6)`
    with `TARGET_EFF ≈ 500` (the eff_batch sweet spot on the RTX 3070; knee at 5–6 streams per
    `results/stream_sweep_*.txt`). Caveat: the saturation point is hardware-dependent (bigger GPUs
    need larger eff_batch to fill) — so `TARGET_EFF` is itself a knob; document "default heuristic
    tuned on RTX 3070, run `scripts/grid_search_gpu.sh` for your hardware."
  - `num_streams` is already a runtime arg, so this is host-side logic only (no kernel changes).
  - Cross-ref: the CNNTesting benchmark reports two Mojo series for honesty — `mojo` (tuned 5-stream)
    and `mojo-s1` (single-stream, fair vs the single-stream libraries).

- [ ] **Audit compile-time `-D` vars: which should be runtime instead?** (`constants.mojo`)
  - Several knobs are `comptime` via `defines.get_defined_int[...]` (NUM_GPU_STREAMS,
    GPU_STREAM_BATCH_SIZE, DIV_CHANS_CONV2/3, ftype, etc). Some genuinely need comptime (drive
    kernel `block_dim`/unrolling/layouts → must be known at build to specialize the kernel).
    Others probably don't: e.g. NUM_GPU_STREAMS just sizes a host-side loop / slot count — making
    it runtime would let one binary sweep stream counts without recompiling per value (the grid
    search rebuilds every cell today). Go knob-by-knob: tag each "needs comptime (why)" vs
    "could be runtime (why)", then migrate the runtime-safe ones. Big payoff for benchmarking.
  - DONE: NUM_GPU_STREAMS → runtime `--num-streams N` (1..10); BENCH_ONLY → runtime
    `--bench-only`. Both bundled in `cli.mojo` (`CliArgs.parse`, `printHelp`), reused by
    main.mojo + tests/profile_gpu.mojo. grid_search now sweeps streams via runtime arg (bs
    outer/comptime → one compile per bs). `-D NUM_GPU_STREAMS` / `-D BENCH_ONLY` kept as
    defaults for back-compat.
  - CLEANUP (later): drop the `-D BENCH_ONLY` comptime path entirely once nothing depends on
    it — runtime `--bench-only` covers it; the define is just clutter for a minor feature.

- [ ] **conv1 kernel: `INPUT > 1` not handled** (`accel/ops.mojo:501`)
  - Kernel hardcodes single-channel input. If ever extended beyond MNIST (grayscale), this breaks.

- [x] **Implement `LeNet5GPUBuffers.__del__`** (`accel/model.mojo:115`)
  - Stale. `DeviceBuffer` fields are RAII-managed by `DeviceContext`; placeholder comment removed.

- [x] **Compile-time explosion in StreamSlot/buffer setup** (`accel/ops.mojo`) — RESOLVED 2026-06-05
  - Root cause was NOT a `comptime for`: it was the `InlineArray[FeatureGPU, batch_size]` *field* on
    StreamSlot — the synthesized struct move unrolled N element-moves at comptime (batch 256 = compiler
    timeout, 1024 = never). Fixed by making the host handles a local in `__init__` (only needed to seed
    the device copy), so StreamSlot's move is a pointer shuffle. Build now batch-independent (~60s).

---

## Benchmarking / Profiling

- [x] **Hoist StreamSlot construction OUT of the timed region — kills run-to-run variance** (`accel/ops.mojo`, `main.mojo`)
  - `runGPUTest` times the whole `batchedForwardMultiStream` call, which `alloc`s + `free`s all
    StreamSlots every pass. `cudaMalloc`/`cudaFree`/pinned-host-alloc are synchronous, pool-dependent,
    and scale with eff_batch → ~10% variance, worst at eff_batch 1500. You're timing setup+teardown,
    not steady-state compute.
  - Fix: split `batchedForwardMultiStream` into setup (build N slots once) + run (loadBatch/doWork/
    getResults). Benchmark loop times only `run` over the reused slots. This is the apples-to-apples
    discipline every framework (PyTorch/JAX/ONNX-RT/MAX) uses.
  - Note: a `DeviceContext` host-callback (cudaLaunchHostFunc-style) is NOT the tool here — that orders
    host work *within* a stream; the prep just needs to move to `__init__`, before the timed loop.

- [ ] **Sweep NUM_GPU_STREAMS past 5; re-tune the default (currently 3)** (`scripts/grid_search_gpu.sh`, `constants.mojo`)
  - After the Tier A conv3 rewrite, a single launch only fills ~9% of the GPU (ncu: 8.8% occupancy,
    0.11 waves/SM). That idle headroom is why more streams now help where they capped at ~3 before
    (block.sum conv3 filled the GPU per launch at 93% occ). 8jun grid search: small batch + s=5 wins
    (bs=75 s=5 ≈ 986k fps vs old ~598k peak), and the curve is still RISING at s=5 for bs=50–100.
  - Action: extend `grid_search_gpu.sh` to s=6,7,8 at bs∈{50,75,100}; find where streams saturate
    (GPU full, or CPU launch overhead dominates). Bump `NUM_GPU_STREAMS` default toward the winner.
  - RESULT (2026-06-08, results/stream_sweep_to8_*.txt): knee at **~5–6 streams**, flat/down after.
    bs50 peaks s=6 (~882k), bs75 s=7 (~882k), bs100 s=5 (~890k). Past ~6, extra streams stop
    finding idle SMs and CPU launch overhead dominates. Default 3 -> **5** (6 marginal); 8 wasted.
    (Absolute fps lower than 8jun grid due to no clock lock + warm GPU; trend is what matters.)
    `grid_search_gpu.sh` now takes BS_VALUES / STREAM_VALUES env overrides.
  - DONE: `NUM_GPU_STREAMS` default 3 -> 5. Kept `GPU_STREAM_BATCH_SIZE`=50 (divides 10000 & 60000
    evenly = full test coverage; ~2.5% under the bs=75 peak — chose clean coverage over peak fps).
  - Caveat: this is mitigating under-occupied kernels with concurrency. Tier B conv3 (many blocks)
    would raise single-stream occupancy and reduce the reliance on high stream counts — compare both.

- [ ] **Lock GPU clocks for benchmark/profile runs** (`scripts/gpu_lock.sh`, `scripts/gpu_unlock.sh`)
  - Consumer 3070 boosts/throttles → timing jitter. `pixi run gpulock` (sudo, real terminal) pins a
    sustainable graphics clock + persistence mode; `pixi run gpuunlock` restores. Lock once per session.
  - NOT auto-prepended to `nsysprofile_gpu`/`ncuprofile_gpu`: sudo needs an interactive password (breaks
    non-interactive pixi), and you want to lock once per session, not per profile.

- [ ] **Test fp64 and fp16 paths; document dtype parity vs other libs** (`constants.mojo` `ftype`)
  - Try `ftype = DType.float64` and `float16`/`bfloat16`; note where the project struggles vs PyTorch/JAX.
  - Apples-to-apples gotcha: PyTorch/JAX default to **TF32** matmuls on Ampere, not true FP32 — disable
    (`torch.backends.cuda.matmul.allow_tf32=False`) or compare same-precision. State dtype in every number.
  - Honesty-of-shortcomings is fine: if a dtype isn't easily supported, "hand it to the libraries" and
    say so — that's legitimate framing for the writeup.

- [ ] **Pre-staging (data-resident / compute-only) GPU benchmark option** (`accel/ops.mojo`, `main.mojo`)
  - Today `_batchRun` does H2D every batch via `StreamSlot.loadBatch` (`device_inputs.enqueue_copy_from`)
    inside the timed region — this measures the **streaming** scenario (data arrives from host), which
    is the apples-to-apples default the Python harness uses.
  - Want a second mode: upload all `COUNT_TEST` pixels to the device **once** before the timed loop, then
    have the timed loop run kernels straight off the resident device buffer (no per-batch H2D). Measures
    **compute-only** throughput — the "dataset already in VRAM" scenario (offline/batch inference, training).
  - Why it matters for the writeup: the streaming number includes Mojo's transfer advantage (raw uint8,
    ~4× less PCIe than the libraries' pre-normalized fp32). A compute-only run **removes** that edge and
    exposes the pure kernel/compiler comparison (Mojo vs XLA/cuDNN/TensorRT) — the ranking may shift.
    Mirrors the CNNTesting plan to add `pytorch-resident` / `jax-resident` variants; report BOTH scenarios,
    clearly labeled, streaming as the headline.
  - Sketch: a flag (e.g. `--resident` / `-D RESIDENT`) that swaps `loadBatch` for a one-time bulk
    `enqueue_copy_from` of the whole pixel arena into a persistent device buffer, then indexes batches
    off it. Normalization currently happens on-GPU per batch — decide whether to pre-normalize the
    resident buffer once too (closer to how the libs pre-normalize on host) or keep it in the timed loop.

- [ ] **Accept already-normalized (pre-normalized fp32) uploaded images** (`accel/ops.mojo` StreamSlot)
  - Today the GPU path uploads raw **uint8** and runs `normalizeInputsKernel` on-device (the ~4×-less-PCIe
    + fused-normalize advantage). Want a second input mode that accepts **pre-normalized fp32** images
    directly — i.e. the same input the Python libraries get — and skips the on-GPU normalize.
  - Why: lets us run the apples-to-apples comparison that *removes* Mojo's uint8/upload edge, isolating
    pure kernel quality (Mojo conv/pool/fc vs cuDNN/XLA/TensorRT) from the smart data path. Pairs with
    the pre-staging/resident item and the CNNTesting `*-resident` variants.
  - Approach: comptime-parameterize `StreamSlot[batch_size]` on the input format (e.g. an `InputKind`
    enum or a `normalize: Bool` / `InputDType` param) so the slot's `loadBatch` either (a) copies uint8 +
    enqueues `normalizeInputsKernel`, or (b) copies fp32 straight into `device_inputs` and skips the norm
    kernel. Keeps both paths in one kernel pipeline, selected at compile time — no runtime branch in the
    hot loop.
  - Honesty payoff: report Mojo BOTH ways (uint8+on-GPU-norm = the real/optimized path; pre-normalized
    fp32 = same-input-as-libs) so the writeup can separately credit "good data path" vs "good kernels."

---

## CPU Pipeline

- [x] **`predict` / `predictNew` could be methods of `LeNet5`** (`cpu/ops.mojo:691`)
  - Already methods on `LeNet5` in `cpu/model.mojo`. TODO was stale.

- [ ] **`trainBatchParallel` accumulation is single-threaded** (`cpu/ops.mojo:751`)
  - The loop that calls `buffer.accumulateFromOther(deltas[i], 1.0)` runs serially after `parallelize`.
  - Should use atomics or a critical section, or restructure to reduce into a tree.

- [x] **`testingParallel`: handle `len(data) % batch_size != 0`** (`cpu/ops.mojo:942`)
  - Added remainder pass after main loop — sequential, avoids race condition, handles any dataset size.

- [ ] **`convoluteBackward` requires explicit `kernel_size=` — investigate why** (`cpu/ops.mojo:649,664,679`)
  - Three call sites need `convoluteBackward[kernel_size=LENGTH_KERNEL](...)` explicitly.
  - File a Mojo bug report if this is a compiler inference failure.

- [ ] **`CPUBumpArenaAllocator.alloc`: consider returning `Span` instead of raw pointer** (`cpu/arena.mojo:39`)
  - Would make ownership and bounds clearer at call sites.

- [ ] **`accumulateFromOther`: needs compiler/stdlib fix** (`cpu/model.mojo:231`)
  - Direct `tensor *= scalar` LayoutTensor math was removed because it explodes compile times.
  - Re-evaluate when Mojo LayoutTensor math performance improves.

- [ ] **`_randHelper`: FIXME compile times from LayoutTensor math** (`cpu/model.mojo:253`)
  - `tensor *= sftype(sqrt(6.0)) / scale` is commented out for the same reason as above.

- [ ] **`bytesToFType`: comptime unrolling may slow compilation for large tensors** (`cpu/model.mojo:303`)
  - Currently uses `for i in range(comptime(tensor.layout.size()))`. Profile compile time impact.

---

## Data Loading

- [x] **Remove or fix dead `MNISTBatch.__init__(images: List[Image], ...)` overload** (`dataloader.mojo`)
  - Fixed when wiring origins: updated to use parameterized `origin`, correct memcpy count,
    rebind for field assignment. Kept as a nice-to-have utility (no callers, not dead weight).

- [x] **`image.mojo`: implement or remove `padded: Bool` flag** (`image.mojo:64`, `image.mojo:97`)
  - Already wired: `comptime off = PADDING if padded else 0` used in both methods.

- [x] **Delete `Image._normalize` static method** (`image.mojo`)
  - Decision: keep as nice-to-have utility (manual PixelStorage→DataTensor path). No callers but not dead weight.

- [x] **Consider merging `getTrainBatch`/`getTestBatch` into one method** (`dataloader.mojo:137`)
  - Decision: skip. Two 10-line methods, callers are already explicit, no real duplication cost.

- [x] **`_readData` (deprecated): skip `InlineArray` intermediate** (`dataloader.mojo:193`)
  - Already removed; only `_readTrainData`/`_readTestData` remain.

---

## Logging (`resultlogger.mojo`)

- [x] **Implement JSON log format** (`resultlogger.mojo:12`, `resultlogger.mojo:29`)
  - Decision: skip. CSV sufficient for current benchmarking needs.

- [x] **`LeNet5Logger.test_size` field — clarify or improve** (`resultlogger.mojo:96`)
  - `TrainingResult.test_size` renamed to `sample_size`. `InferenceResult.test_size` kept
    (correct name for inference count). Stale comment removed.

- [x] **`headers_written: Bool` — make this better** (`resultlogger.mojo:191`)
  - Removed field; `_writeResult` now checks `os.path.exists` per write. All log methods drop `mut self`.

- [x] **Collapse the `runTrain`/`runTest` logger overloads into one `Optional[Logger]` param** (`main.mojo`)
  - Done: `runTrain`, `runTest`, `benchCPUInference` unified to `logger: Optional[MultiFileLogger] = None`.
    `trainingParallel`/`training` in `cpu/ops.mojo` same. Added `ImplicitlyCopyable` to logger structs.

---

## Cleanup / Dead Code

- [x] **Remove unused `reflect` import in `main.mojo`** (`main.mojo:11`) — still used for `act_fn_name`; TODO was stale

- [x] **FIXME: `CPUSession` — tie arena + model lifetimes together** (`main.mojo:224`)
  - `CPUSession` added to `cpu/model.mojo` (mirrors `DeviceSession`). Holds `arena` + `model`.
    `benchmark.keep(arena)` removed. `CPUBumpArenaAllocator` import dropped from `main.mojo`.

- [x] **`FeatureGPUBuffers` still exists in `accel/model.mojo` after split to `accel/feature.mojo`**
  - Resolved: `accel/feature.mojo` is canonical; `accel/model.mojo` copy removed.

---

## Upstream Bug Reports to File

- [ ] **`convoluteForward` slice syntax — API bug** (`cpu/ops.mojo:389`)
  - Slice call requires specific form that differs from what docs describe. File Mojo issue.

- [ ] **`convoluteForward` slice IndexList vs Int — docs bug** (`cpu/ops.mojo:405`)
  - Docs say `IndexList` is expected but passing `Int` is needed. File Mojo issue.
