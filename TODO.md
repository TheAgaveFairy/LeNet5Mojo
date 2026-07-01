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

- [x] **Centralize weight + feature layouts into structs in `constants.mojo`** — DONE 2026-06-25
  - The `Layout.row_major(...)` defs were duplicated across BOTH CPU (`cpu/model.mojo` `LeNet5`,
    `Feature`) and GPU (`accel/feature.mojo` `FeatureGPU`, `accel/model.mojo` `LeNet5GPU`) with
    byte-identical shapes, only the member names differing (`w01_layout` vs `w0_1_layout`).
  - Hoisted into three namespace structs in `constants.mojo`: `FeatureLayouts` (input, layer1..5,
    output), `WeightLayouts` (w01/w23/w45/w56), `BiasLayouts` (b01/b23/b45/b56). Dropped the
    `_layout` suffix — the struct prefix reads well (`FeatureLayouts.layer4`). Standardized the
    `w01` (no-underscore) naming per Paul; left the `weight0_1`/`bias0_1` *field* names alone
    (already consistent everywhere). All four structs + the kernels in `accel/ops.mojo` now point
    at the central defs; single source of truth for CPU + GPU.
  - Pure no-op refactor: build green, accuracy exact CPU 9648/10000 + GPU 9648/10000 (1.32M fps s=5).
  - NOTE: did NOT add GEMM K/M descriptors (the original blurb's `gemm[B,K,M,...]` angle) — Paul
    scoped this as clarity/organization only. If Tier B wants named K/M, add a thin descriptor
    layer over these later. Also dovetails with the explicit `[1,28,28]` single-channel TODO
    (would just edit `FeatureLayouts.input` + the MNIST loader in one place now).

- [ ] **(Future) Kill `List[Image]` from CPU hot path; share SoA spans for both CPU and GPU** (`dataloader.mojo`)
  - CPU path currently uses `List[Image]` (AoS, 785-byte stride with label interleaved).
  - GPU path already uses SoA pixel/label arenas exposed via `MNISTBatch`.
  - If CPU profiling shows packing cost, migrate CPU ops to consume `raw_pixels`/`raw_labels` spans
    directly. `Image` becomes a lightweight view into the arena rather than an owned struct.

- [ ] **(Long-term) Batched feature layout — SoA across images, not AoS `FeatureGPU` per image** (`accel/feature.mojo`, `accel/ops.mojo`)
  - Today every kernel is `grid=(batch_size, ...)` with one image's small feature tensors per block
    (`feats[img].layerN`). A batched `[N, C, H, W]` tensor per layer would let conv2/conv3 become real
    GEMMs over the whole eff_batch (better coalescing, fewer/larger launches). This is the natural
    generalization of conv3 Tier B — note it here so Tier B is designed with batched layouts in mind.

- [ ] **GPU ops should take weight/bias FIELDS directly, not the `lenet` holder** (`accel/ops.mojo`, `accel/model.mojo`, `main.mojo`)
  - DECISION 2026-07-01: pass each kernel only the tensors it uses (e.g. `matMulFused(weight5_6, bias5_6,
    ...)`, `conv3(weight4_5, bias4_5, ...)`) instead of the whole `LeNet5GPU`. Rationale: the signature
    becomes the real dependency contract (standard cuDNN/CUTLASS style); it's a SMALLER by-value param
    footprint (2-3 fat-pointer tensors vs an 8-field struct); and with `//`-inferred `Layout` params the
    dims fold to comptime cleanly (`weight5_6.shape[0]()`) — no `type_of()` workaround, and the compiler
    checks the passed tensor. Makes kernels unit-testable in isolation → pairs with the PyTorch-parity
    suite (#7). `LeNet5GPU` STAYS as the host-side owning aggregate (lifetimes via `DeviceSession`); it just
    stops needing `DevicePassable`/`TrivialRegisterPassable` (drop that machinery — individual LayoutTensors
    are already device-passable). Writeup: keep the DevicePassable-model trick as an exploration→refinement
    arc, ship the cleaner API.
  - SCOPE: WEIGHTS-FIRST (mechanical: ~7 kernel sigs + their `enqueue_function`/`CompiledKernels` call sites
    + drop `lenet` arg). The `feats`/`FeatureGPU` side is per-image AoS pointer-passed on purpose (param
    ceiling) → leave it to the batched-SoA refactor above. SUPERSEDES the `type_of(lenet.weightX).shape[N]()`
    half-measure for GPU kernels (see the shape[N]() item + Discord follow-up). Do after the low-hanging
    fruit; it's a real refactor, not a style tweak.

- [ ] **Style guide: index via `tensor.shape[N]()`, not hardcoded constants** (all ops, `cpu/ops.mojo` + `accel/ops.mojo`)
  - Marker at `accel/ops.mojo:76`. Prefer `input.shape[0]()`-style queries over `IMAGE_SIZE`/`LAYER5`
    etc. so kernels/ops stay correct if a layout changes. Sweep CPU + accel ops; pure clarity/robustness.
  - IN PROGRESS 2026-07-01: `normalizeInputsKernel` converted (reads `raw_pixels.shape[1/2]()` — a DIRECT
    param). BLOCKER found: `shape[N]()` is only comptime on a direct `LayoutTensor` param, NOT on a struct
    field — so the GPU conv/matmul kernels (take `lenet`/`feats` structs; dims live in `lenet.weightX` /
    `feats[i].layerN`) can't adopt it for their `comptime`/`comptime for` dims. See the Discord follow-up +
    MWE (`ignoreme/shape_comptime_mwe.mojo`) in the Upstream section. Realistic scope until resolved:
    ops/kernels that take the relevant tensor as a DIRECT param (CPU `convolute*` already do; GPU
    `normalizeInputsKernel` done). The struct-field kernels stay on named constants (single source =
    the `*Layouts` structs) for now.

- [ ] **Parameterize + fuse the `act_fn` epilogue (CPU + GPU)** (`accel/ops.mojo:158`, `cpu/ops.mojo:418`, `cpu/ops.mojo:582`)
  - Three markers, one theme: make the post-op activation a compile-time knob (enable/disable) and fuse
    it into the preceding accumulation loop instead of a separate `act_fn.forward` pass — `matMulFusedKernel`
    (GPU, raw-logits epilogue), `convoluteForward` (fuse `simdForward()` into the bias add loop), and
    `matmulForward` (CPU, enable/disable + fuse). Pairs with the existing matmul-act decision (skip after
    final FC). Supersedes the loose `# TODO: look into if this is good or bad` / `# FIXME: just a louder
    reminder` cluster at `cpu/ops.mojo:581-583`.

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
  - CLEANUP DONE 2026-06-24: dropped the `-D BENCH_ONLY` comptime path entirely. `comptime
    BENCH_ONLY` + its `defines` import removed from `cli.mojo`; `bench_only` defaults to `False`;
    `-D BENCH_ONLY` line dropped from `printHelp`; main.mojo error message no longer mentions it.
    Migrated the three callers to runtime `--bench-only`: pixi `bench` + `nsysprofile` tasks,
    `scripts/grid_search_gpu.sh`. Build green, `--help` + `--bench-only` verified.

- [x] **conv1 kernel: `INPUT > 1` not handled** (`accel/ops.mojo`) — DONE 2026-06-20
  - Kernel hardcodes single-channel input. Added `comptime assert INPUT == 1` at the top of
    `conv1FusedKernel` so bumping `INPUT` for a multi-channel set fails at compile time with a clear
    message instead of silently producing wrong results. Verified the assert fires (INPUT=2 →
    "constraint failed: conv1FusedKernel hardcodes INPUT==1"). Actually implementing multi-channel
    conv1 remains future work (the staging + MAC loop need an INPUT loop).

- [x] **Implement `LeNet5GPUBuffers.__del__`** (`accel/model.mojo:115`)
  - Stale. `DeviceBuffer` fields are RAII-managed by `DeviceContext`; placeholder comment removed.

- [x] **Compile-time explosion in StreamSlot/buffer setup** (`accel/ops.mojo`) — RESOLVED 2026-06-05
  - Root cause was NOT a `comptime for`: it was the `InlineArray[FeatureGPU, batch_size]` *field* on
    StreamSlot — the synthesized struct move unrolled N element-moves at comptime (batch 256 = compiler
    timeout, 1024 = never). Fixed by making the host handles a local in `__init__` (only needed to seed
    the device copy), so StreamSlot's move is a pointer shuffle. Build now batch-independent (~60s).

- [x] **Fix `loadBatch` short-batch OOB read** (`accel/ops.mojo` `StreamSlot.loadBatch`) — DONE 2026-06-12
  - `hosted_inputs.enqueue_copy_from(batch.unsafe_ptr())` copies the *buffer's* full length from the
    span pointer — a short span reads past its end. The preceding `enqueue_fill(0)` is then fully
    overwritten anyway. Latent today (`_batchRun` only sends full batches) but armed the moment the
    pad-the-tail item lands. Fixed: host `memcpy` of `len(batch)` bytes into the pinned buffer,
    `memset_zero` only the remainder. Unblocks pad-the-tail.

- [ ] **Fuse normalize into conv1** (`accel/ops.mojo`)
  - conv1 already stages the input in shared memory; it could load raw uint8 pixels, do the block.sum
    mean/std reduction, and normalize in shared before convolving. Kills one launch per batch AND a
    full global round-trip of the 32×32 fp32 input. Pairs with the normalization-parity item below.

- [ ] **Fuse matmul → outputs (+ GPU argmax); delete gather kernel** (`accel/ops.mojo`)
  - `matMulFusedKernel` can write straight into the batched `outputs` tensor — `gatherOutputsKernel`
    disappears (1 of 8 launches gone; launch overhead matters at these kernel sizes). Stretch: do
    argmax on-device and D2H 1 byte/img instead of `OUTPUT` floats; `getResults` becomes a byte compare.
  - DONE 2026-06-12 (both parts):
    - Base fusion: `gatherOutputsKernel` deleted, `matMulFusedKernel` takes the batched `outputs`
      tensor and writes logits directly (7 launches/batch, was 8). RESULT (unlocked, bs=100, vs
      same-day post-pool-rewrite): s=5 1.289M → **1.319M fps** (+2.3%), s=1 906k → **927k** (+2.3%).
    - GPU argmax: thread 0 tracks running max while writing logits, writes `guesses[img]`; hot path
      D2H is now 1 byte/img (was OUTPUT floats), `getResults` is a byte compare. Matches the
      PyTorch/JAX device-argmax pattern (output-judging parity item). RESULT: s=5 flat (~1.319M),
      s=1 927k → **934k** (+0.7%, noise-level) — D2H was already tiny; value is parity + cleaner
      `getResults`. Logits still land in `outputs` on device for debugging; `batchedArgMax` kept
      as host-side fallback. Accuracy 9648/10000, exact CPU match throughout.

- [ ] **`getResults`: SIMD the guess-vs-label byte compare** (`accel/ops.mojo`)
  - The scalar `for j in range(batch_size)` loop compares two UInt8 buffers. Candidates: load
    `SIMD[DType.uint8, nelts]` chunks from both (`UnsafePointer.load[width=...]`), `(g == l).cast[DType.uint8]().reduce_add()`
    to count matches; or `comptime for` unroll since `batch_size` is comptime. ~50–100 bytes/batch so
    perf impact tiny — mostly a SIMD exercise. `batchedArgMax` kept around as the host-side fallback.

- [ ] **Remove now-unused `FeatureGPU.output` buffer — or fold into the SoA migration** (`accel/feature.mojo`)
  - After the matmul→outputs fusion no kernel reads/writes `feats[img].output`; the field, its
    layout, the `FeatureGPUBuffers.output` alloc, and its share of `sizeInBytes()` are dead weight
    (small: OUTPUT floats/img, but misleading to readers). Two options: (a) quick delete now, or
    (b) leave it for the batched-SoA layout migration (see the Architecture item above) which
    replaces per-image feature structs wholesale — the batched `outputs` tensor in StreamSlot is
    effectively the first SoA-ified layer. Don't do (a) in a way that makes (b) harder.

- [x] **Pool kernels: drop shared memory** (`accel/ops.mojo` `maxPool1Kernel`, `maxPool2Kernel`) — DONE 2026-06-12
  - 2×2 non-overlapping pooling has ZERO data reuse — staging layer1/layer3 in shared then reading it
    back is pure overhead (extra latency + a barrier). maxPool1 also launched 28×28 threads and idled
    75% of them after the load. Now one thread per *output* (14×14 / 16×5×5), 4 global reads, no
    shared, no barrier. RESULT (same-day A/B, clocks UNLOCKED both sides, bs=100): old kernels
    1.169M fps s=5 / 794k s=1 → new **1.289M fps s=5 (+10%) / 906k s=1 (+14%)**. Accuracy 9648/10000,
    exact CPU match. NOTE: vs the *recorded* ~890k baseline the jump looks like +45%, but that
    recording was warm/unlocked at different conditions — only the same-day A/B delta is real.
    Re-baseline with locked clocks (`pixi run gpulock`) before quoting numbers in the writeup.

- [ ] **conv3 Tier A: pad block_dim 120 → 128** (`accel/ops.mojo` `conv3FusedKernel`)
  - 120 threads = 3.75 warps; the partial warp wastes a scheduler slot. Cheap experiment: launch 128,
    guard `oc < LAYER5` (or give the 8 spare threads shared-load duty).

- [ ] **FC matmul as a real GPU GEMM** (`accel/gemm.mojo` — new WIP file)
  - New untracked scaffold: `gemm3` kernel stub + a copy of `matMulFusedKernel`. Goal is a tiled GEMM
    for the final FC layer (`LAYER5`×`OUTPUT`) instead of the per-block reduction. Separate from conv3
    Tier B. Marker `# TODO: dram to local call possible` (`accel/gemm.mojo:68`). NOTE: file is untracked
    (`git add` when ready).

- [ ] **GPU signature/cleanup nits** (`accel/ops.mojo`)
  - Pre-existing uncatalogued markers: take `Span`s in the conv kernels (`:232`); make `stream_slots` a
    `Span` (`:771`); bypass an intermediate (`:587`); replace ad-hoc print with proper logging (`:675`).
    Low priority; grouped here so they're tracked.

---

## Benchmarking / Profiling

- [ ] **Add a deliberately-bad allocator to benchmark against** (`cpu/arena.mojo`, `accel/arena.mojo`)
  - The bump-vs-system swap is ~noise on inference (both allocate once, then `zero()` a small buffer).
    A "BadAllocator" that emulates bad-design practices would give the benchmark real contrast and show
    WHY the arena matters. E.g. `zero()` frees the old buffer, allocates a fresh one, and memsets it in a
    dumb (e.g. byte-at-a-time / unaligned / per-element) way; `alloc()` could over-allocate or never reuse.
    Conforms to `CPUAllocator`/`GPUAllocator` so it drops straight into the existing `-D *_SYSTEM_ALLOC`-style
    toggle (add a `-D *_BAD_ALLOC` arm to the `ConditionalType` in `constants.mojo`). Pairs with the
    "Benchmark the GPU arena allocator" item — the bad allocator is the pessimal baseline the arena beats.

- [ ] **Run the act_fn × ALPHA search + finish the rigor pass** (`scripts/search_alpha.sh`)
  - DONE 2026-06-30 (script prep/improvements): fixed stale paths (`src/main.mojo`, `results/` output);
    added **phase 3** FINE sweep (±16 step 4) around the phase-2 peak — recovers the precision the step-10
    grid lost (the `HALF_WIDTH=100` concern); bumped phase-2 half-width 100→120; **dedup** across phases
    via the CSV; **CENTER_ALPHA** / default PRIMARY-full-then-VARIANTS-centered-on-GELU flow so the GELU
    variants skip the coarse sweep. Helper logic offline-validated. Usage documented in
    `docs/activation_tuning.md`.
  - REMAINING:
    - **RUN it** — ~1.5–2 h (each point recompiles + trains ~35 s). Paul to hammer it on a free machine,
      then fill `docs/activation_tuning.md`.
    - **Noise**: still single-seed (deterministic at `--seed 42`). Average N seeds per point / report
      mean±sd so "best ALPHA" isn't chasing jitter.
    - **Leakage**: still tuned on the TEST set — carve a validation split from train.
    - **Refinement (optional)**: the fixed coarse→linear→fine phases could become adaptive
      (golden-section / successive-halving / small Bayesian) for fewer evals to a tighter optimum.
  - Output feeds the per-act_fn suggested-defaults doc (`docs/activation_tuning.md`).

- [ ] **Fill in per-act_fn suggested ALPHA defaults** (`docs/activation_tuning.md`)
  - Doc scaffold created 2026-06-30 with a table + the few data points we have (mostly TBD). Populate it
    from the improved search above so people have sane starting `-D <ACT> -D ALPHA=N` combos per activation.

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

- [ ] **Normalization parity vs libraries** (`accel/ops.mojo` `normalizeInputsKernel`)
  - Mojo does **per-image** mean/std standardization (two block.sum reductions per image, on GPU).
    Standard torchvision preprocessing is **fixed dataset constants** (0.1307/0.3081) — much cheaper,
    usually host-side, and numerically different (different accuracy too, not just speed).
  - For apples-to-apples: either add a fixed-constant normalize mode here, or make every framework
    harness do per-image standardize. Whichever way, state it explicitly in the writeup — right now
    Mojo is doing MORE preprocessing work than the libs while also being timed for it.

- [ ] **Output-judging parity: device argmax + device compare** (`accel/ops.mojo` `getResults`)
  - Library harnesses do `logits.argmax(dim=1)` and `(preds == labels).sum()` on device, D2H one
    scalar per batch. Mojo D2Hs `OUTPUT` floats/img then does host argmax + host label compare inside
    the timed region — strictly more transfer + host work. Move argmax/compare on-device (pairs with
    the matmul-fusion item) and transfer just the correct-count, or document the asymmetry.

---

## CPU Pipeline

- [x] **`predict` / `predictNew` could be methods of `LeNet5`** (`cpu/ops.mojo:691`)
  - Already methods on `LeNet5` in `cpu/model.mojo`. TODO was stale.

- [x] **`trainBatchParallel` grad accumulation is serial — WON'T FIX (low ROI)** (`cpu/ops.mojo:640`)
  - The loop calling `buffer.accumulateFromOther(deltas[i], 1.0)` runs serially after the
    `parallelize(work, batch_size)` — but the heavy fwd/bwd per image is ALREADY parallel across
    cores (each writes its own `deltas[tid]`, race-free). Only the reduction is serial, and
    `accumulateFromOther` is ALREADY SIMD-vectorized (`_accumHelper` → `vectorize[nelts]`).
  - Cost is tiny: reduction streams ~52k floats/img (~208 KB; w4_5 dominates at 48k), memory-bound,
    ~4% of batch flops (fwd/bwd ≈ 60M flop/batch). Amdahl ceiling on any fix = that small slice.
  - Training is NOT the project headline (inference benchmark is) → low ROI. KEEP SERIAL.
  - If ever revisited: profile FIRST; and the old "atomics / critical section" note is WRONG for this
    shape (per-element atomic-add on 52k floats serializes worse). Right design = strided tree reduce
    into `deltas[0]` (race-free, batch_size/2-way first pass), then `model.accumulateFromOther(deltas[0], k)`
    dropping `buffer` entirely. No atomics.

- [x] **`testingParallel`: handle `len(data) % batch_size != 0`** (`cpu/ops.mojo:942`)
  - Added remainder pass after main loop — sequential, avoids race condition, handles any dataset size.

- [x] **`convoluteBackward` requires explicit `kernel_size=` — ROOT CAUSE FOUND (upstream)** (`cpu/model.mojo:293,302,311`)
  - DIAGNOSED 2026-06-25 (reproducer `ignoreme/probe_conv_infer.mojo`): Mojo binds params
    left-to-right with NO deferred unification. `input` binds in_chan/feat_size; the next arg
    `outerror` (layout `(out_chan, feat_size-kernel_size+1, ...)`) is checked immediately while
    out_chan/kernel_size are still unbound. `feat_size-kernel_size+1` is non-invertible arithmetic →
    kernel_size unsolvable there, and the compiler does NOT skip ahead to `weight` (kernel_size
    appears directly as dims 2,3). Error: "types parameters include unfolded expression at parser
    time." Probe variant `fB` (weight BEFORE outerror) infers fine; `fA` (real order) fails. So it's
    an inference-ordering limitation, not user error.
  - RESOLUTION (#1): keep the explicit `kernel_size=LENGTH_KERNEL` (harmless, arguably clearer);
    root-cause comment added at `convoluteBackward` in `cpu/ops.mojo`. Paul to ask Discord first
    (known? expected?) before filing an upstream issue — reproducer is ready to paste.
  - FUTURE OPTION #2 (only if the explicit param becomes annoying): reorder the fn args so
    `weight, wdeltas` come BEFORE `outerror` → kernel_size + out_chan infer, drop the explicit param
    at all 3 call sites. Cost: arg order reads worse (weight ahead of the output it helps produce).
  - FUTURE OPTION #3 (most robust, biggest change): drop the int params entirely; take generic
    `Layout` params for weight/outerror/wdeltas and derive sizes inside via `.shape[N]()` + comptime
    asserts — the pattern already used by `convoluteValid`/`convoluteFull` in the same file. Decouples
    from param inference completely. Do this if a future refactor touches the conv backward path anyway.

- [ ] **`CPUBumpArenaAllocator.alloc`: consider returning `Span` instead of raw pointer** (`cpu/arena.mojo:39`)
  - Would make ownership and bounds clearer at call sites. NOT low-hanging: `alloc` returns
    `UnsafePointer[T, MutUntrackedOrigin]`; a `Span` return adds an origin param that colors every
    caller (feature arenas in `cpu/ops.mojo`, `cpu/model.mojo`) doing pointer math / `LayoutTensor`
    construction off the result. Likely one of the larger refactors, not a quick swap. Defer.

- [x] **`accumulateFromOther` / `_randHelper`: direct LayoutTensor math — UPSTREAM, WON'T FIX** (`cpu/model.mojo`)
  - Both keep a hand-rolled `vectorize[nelts]` workaround instead of direct elementwise LayoutTensor
    math (`tensor *= scalar`, `accum += other * lr`, `tensor *= sqrt(6.0)/scale`). The direct form
    explodes compile times — it appears to unroll an op per element at comptime.
  - MEASURED 2026-06-25 (probes in `ignoreme/probe_lt_direct.mojo` vs `probe_lt_vectorize.mojo`,
    single N=48000 tensor = real w4_5 size): vectorize workaround builds in **2.1s / 417 MB**; the
    direct-math build did **NOT finish in 10 min** (killed). Same machine, same nightly.
  - Verdict: this is a Mojo compiler/stdlib limitation, OUT OF OUR CONTROL — nothing to fix in this
    repo. The vectorize path is correct AND fast; keep it. Re-test only if a future nightly is
    reported to fix LayoutTensor elementwise math (rerun the two probes). Not blocking anything.

- [x] **`bytesToFType`: comptime unrolling — STALE FIXME, no blowup** (`cpu/model.mojo`)
  - False alarm. The outer `for i in range(comptime(tensor.layout.size()))` is a RUNTIME loop —
    `comptime(...)` just materializes the size as a value, it does NOT unroll. The only unrolled
    loop is the inner `comptime for bi in range(f_sz)` over f_sz (4 for Float32 / 8 for Float64) —
    tiny, size-independent. Corrected the comment in-place; nothing to profile.

### New markers (added 2026-06-29 pass)

- [ ] **`matmulForward` is not production-grade — port the real CPU matmul** (`cpu/ops.mojo:553`)
  - Marker: "this is not production grade, i have one somewhere to copy over...". CPU-only; bring over
    Paul's existing CPU matmul and replace the naive triple-loop. Unrelated to the GPU `gemm.mojo` /
    conv3 work below.

- [ ] **`predict`: make `feat` an explicit `Optional[Feature]` and combine the two paths** (`cpu/model.mojo:321`)
  - One predict that takes/builds the feature arena instead of two near-duplicate methods.

- [x] **`crossEntropyLoss` returns `Float32` not `sftype` — confirm intent** (`cpu/ops.mojo:90`)
  - RESOLVED: intentional. Loss is a *reporting metric*, not part of model compute (backward uses
    `softmax - onehot` directly, never flows through this). Accumulators (`max_val`, `exp_sum`) stay
    `sftype` so the sum widens if `ftype`→fp64; only the final report truncates to `Float32`.
    Report-narrow / accumulate-wide is the right shape. Comment updated to say so.

- [ ] **`convoluteBackward`: rebind helper for slicing, or eliminate rebinds entirely** (`cpu/ops.mojo:175`)
  - The per-slice `rebind[...]` calls are noisy; factor a helper or restructure layouts to drop them.

- [x] **`maxPoolBackward`: add shape asserts** (`cpu/ops.mojo:438`)
  - RESOLVED: `comptime assert in_feat_size % out_feat_size == 0` (clean pooling — floor-div `len`
    silently drops trailing rows otherwise) + `out_feat_size > 0`. Write index is provably in-bounds
    (`(out-1)*len + (len-1) < out*len <= in`), so this guards *ignored rows / garbage calls*, not OOB.
    Also noted the scatter precondition: caller must pre-zero `inerror` (backward only writes argmax cells).

- [ ] **Benchmark "branchless" maxpool vs a normal one** (`cpu/ops.mojo:450`, `cpu/ops.mojo:462`)
  - Two markers: verify the branchless `maxPoolBackward` inner loop is actually faster, and A/B the
    branchless `maxPoolForward` against a straightforward max. May not be worth the obfuscation.

- [ ] **`loadFromFile`: kill the extra copy** (`cpu/model.mojo:397`)
  - Reads into an `InlineArray` then `memcpy`s; load straight into the destination buffer.

- [x] **`CPUSession`: offer constructors for other allocators** (`cpu/model.mojo:537`) — DONE 2026-06-30
  - Now `CPUSession[Allocator: CPUAllocator = CPUBumpArenaAllocator]` — parameterized on the allocator,
    default keeps every `CPUSession()` call working. Required aligning the `CPUAllocator` trait to
    `GPUAllocator` (uniform `__init__`/`zero`/`wipe` + `ImplicitlyDeletable, Movable` supertraits) so the
    allocator can be a session field and swapped with no branching. Verified: generic `exercise[A:
    CPUAllocator]` runs Bump + System identically; main still 9691/10000.

- [ ] **Drop the `benchmark.keep()` calls in train loops** (`cpu/ops.mojo:642`, `cpu/ops.mojo:698`)
  - `trainBatchParallel` + `trainBatch` both `keep()` arenas to dodge DCE; check if the new
    Session/origin lifetimes make them unnecessary now.

- [ ] **`bytesToFType` big-endian: `from_bytes` flag had compiler issues; parameterize the swap** (`cpu/model.mojo:364`, `cpu/model.mojo:433`)
  - FIXME at :364 — `Scalar.from_bytes(buffer, big_endian=...)` errored, fell back to default; investigate
    or file upstream (see Upstream section). TODO at :433 — make the manual `is_big_endian()` swap a
    function parameter / arg check instead of comptime-only.

### New markers (added 2026-06-30 pen-and-paper audit)

- [x] **`MNISTDataView.__getitem__(start, end)`: add range validation** (`dataloader.mojo:58`) — DONE 2026-06-30
  - Added loud `raise` guard (`start<0 or end<=start or end>len(self)`). Also refactored the body to Span
    slice syntax (`raw_pixels[start*image_size:end*image_size]`) + rebind to `ImmutUntrackedOrigin`,
    dropping the manual pointer math. NOTE: Span slice CLAMPS OOB (`slc.indices()`) rather than raising,
    so the explicit guard is the real validation — slicing alone would silently give a wrong-size view.

- [x] **`_timing_stats`: drop the hand-rolled `less_than` sort closure** (`main.mojo:148`) — DONE 2026-06-30
  - Replaced the `@parameter less_than` closure + `sort[cmp_fn=less_than]` with bare `sort(Span(times))`.
    Default ascending overload (`builtin/sort.mojo:585`, `T: Copyable & Comparable`) covers `UInt`. 5→1 line.

- [x] **Seed: one mature default + expose via the LIVE cli** (`dataloader.mojo:206`, `main.mojo:107`, `cli.mojo`) — DONE 2026-06-30
  - Added `comptime DEFAULT_SEED = 42` to `constants.mojo` (single source). Wired `--seed N` into the LIVE
    `cli.mojo` (`CliArgs.seed` field + fail-loud parse + `printHelp` line); `main.mojo` now `seed(cli.seed)`
    (was `42069`); `dataloader.seed_default` now aliases `DEFAULT_SEED` (was `69`). Verified: `--help` shows
    `--seed` (default 42), `--seed abc` fails loud. Dormant `cliparser.mojo` left alone. Audit item 4.

- [x] **Relocate `origin_util.mojo`'s `untrack` / `untrack_imm`** — AUDITED, NO MOVE 2026-06-30
  - `origin_util` is imported by 7 files across BOTH packages (`image`, `cpu/model`, `cpu/arena`,
    `accel/feature`, `accel/ops`, `accel/gemm`, `accel/model`). Folding into `cpu/ops.mojo` would make
    `accel/*` import the heavy CPU-training module — wrong dependency direction; rejected. And there's no
    pile of generic helpers to justify a `utils.mojo` (cpu/ops free-fns are all domain ops; only
    `showProgress` is generic). Verdict: `origin_util.mojo` is already the correct home — minimal, shared,
    descriptively named. Kept as-is. Audit item 5.

- [x] **`act_fn.forward` should call `simdForward()` internally** (`activation_fn.mojo`) — DONE 2026-06-30
  - Went further than delegation: since every `forward` was just its `simdForward` mapped elementwise,
    hoisted ONE default `forward` into the `ActivationFunction` trait (load → `Self.simdForward` → store)
    and DELETED all 6 per-struct overrides (ReLU/Sigmoid/Tanh/GELU/GELUTanh/GELUFast). Trait-default method
    with a nested `vectorize` closure + `Self.simdForward` compiles fine. Verified: build green; bench ReLU
    9691/10000 (CPU==GPU); full train+test `-D GELUFast` 9003/10000 (CPU==GPU) — forward AND backward good.
    Also fixed `cli.mojo printHelp` (advertised bogus `-D ACT_FN`; real flags are bare `-D GELU` etc.).
    Surfaced dead imports → new cleanup item above. Audit item 6.

- [ ] **PyTorch parity test suite via Mojo/Python interop** (`tests/`)
  - Compare each op (conv/pool/fc/activation, fwd + bwd) against a PyTorch reference through Python interop.
    SKILL: `mojo-python-interop`. New test target. Audit item 7.

- [x] **GPU arena trait audit** (`accel/arena.mojo`) — AUDITED 2026-06-30
  - Generic consumers (`DeviceSession[Allocator: GPUAllocator]`, `LeNet5GPUBuffers`) call only `alloc`
    (+ `__init__` for `DeviceSession(ctx)`). No production call sites for `.free_all()`/`.zero()`/`.wipe()`/
    `.base_address()` — test-only. `GPUSystemAllocator` is production-dead (test-only), same as CPU's.
  - Parity vs CPU `CPUAllocator`: `__init__`-in-trait is JUSTIFIED (GPU constructs the allocator generically;
    CPU doesn't) — cost is `GPUSystemAllocator.__init__` ignoring `capacity_bytes` (`:101`). `zero`/`wipe`
    in the trait is an UNJUSTIFIED divergence (CPU keeps them out as "arena-specific") but Paul chose to
    LEAVE AS-IS for symmetry/future use — no code change.
  - USEFUL FOLLOW-UP: `GPUSystemAllocator` is the arena-vs-system BENCHMARK baseline, not dead weight.
    Wired a `-D GPU_SYSTEM_ALLOC` comptime toggle in `runGPUTest` (`ConditionalType` picking
    `DeviceSession[GPUSystemAllocator]` vs `[GPUBumpArenaAllocator]`; label via `reflect[GPUAllocT].base_name()`).
    Verified 2026-06-30: `GPUSystemAllocator` runs cleanly through `DeviceSession` (safe — no `free_all`/`wipe`
    mid-run, so system buffers persist).
  - CAVEAT (matters for the "Benchmark the GPU arena allocator" item, ideas.typ §GPU Pinned Memory): the
    session allocator only allocates the 8 weight/bias buffers ONCE at setup, OUTSIDE the timed loop, so
    `-D GPU_SYSTEM_ALLOC` moves fps by ~noise (measured ~1.12M both ways). The HOT-PATH per-batch feature
    buffers use a SEPARATE arena — `device_arena`, hardcoded `GPUBumpArenaAllocator` in `accel/ops.mojo:554,579`
    (`StreamSlot`). To actually benchmark arena-vs-system on throughput, parameterize `StreamSlot`'s
    `device_arena` on the allocator too (or add a matching `-D`), not just the session. Audit item 8.

- [x] **`_randHelper`: misleading commented reference math** (`cpu/model.mojo:243-248`) — VOID 2026-06-30
  - Paul reviewed: code is fine as-is, no change wanted. Not a bug. Audit item 9 closed.

- [ ] **Docstrings audit** (repo-wide) — open-ended
  - Sweep public structs/methods/free-fns for missing, stale, or misleading docstrings. Prioritize the
    public API surface (`cpu/model.mojo`, `accel/model.mojo`, `dataloader.mojo`, `cli.mojo`). Ongoing.

- [ ] **Error-path / abort audit** (repo-wide) — open-ended
  - Sweep for silent failures, clamps, and hard aborts: functions that should `raise` but swallow (cf. the
    `__getitem__` clamp finding), `debug_assert` that vanishes in release, and any `abort`/unreachable that
    should be a recoverable error with a message. Decide raise-vs-assert per site. Ongoing.

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

- [x] **`activation_fn.mojo`: remove unused test/random imports** (`activation_fn.mojo:4-10`) — DONE 2026-06-30
  - Dropped the dead `std.testing` block (`TestSuite, assert_equal, assert_true, assert_almost_equal`) and
    `from std.random import seed, randn` — each symbol appeared exactly once (the import itself), no file
    tests/main. Build green. (Restoring an actual activation test suite comparing forward/simdForward/backward
    still belongs with the PyTorch-parity test item #7.)

- [x] **Remove unused `reflect` import in `main.mojo`** (`main.mojo:11`) — still used for `act_fn_name`; TODO was stale

- [x] **FIXME: `CPUSession` — tie arena + model lifetimes together** (`main.mojo:224`)
  - `CPUSession` added to `cpu/model.mojo` (mirrors `DeviceSession`). Holds `arena` + `model`.
    `benchmark.keep(arena)` removed. `CPUBumpArenaAllocator` import dropped from `main.mojo`.

- [x] **`FeatureGPUBuffers` still exists in `accel/model.mojo` after split to `accel/feature.mojo`**
  - Resolved: `accel/feature.mojo` is canonical; `accel/model.mojo` copy removed.

- [ ] **Bundle the 8 `DeviceFunction`s into a `CompiledKernels` struct** (`accel/ops.mojo`, `main.mojo`)
  - `doWork`, `_batchRun`, `batchedForwardMultiStream` all thread 8 positional params
    (`norm, conv1, pool1, ...`). One struct built next to the `compile_function` calls in
    `runGPUTest`, passed as one arg. Pure signature hygiene; also makes adding/removing a
    kernel (e.g. deleting `gather` after the matmul fusion) a one-site change.
  - DONE 2026-06-12. Working recipe (proven in `ignoreme/mvp_compiled_kernels.mojo` first):
    field type = `type_of(DeviceContext().compile_function[kernel[Self.batch_size]]())` — the
    CHECKED return type, compiler-spelled, so launch sites keep compile-time arg validation.
    Dead ends for the record: opaque inferred per-field type params (old skeleton — compiled,
    launches failed); bare `DeviceFunction` fields ("is not concrete" — bare works only as
    inferred *args*); `DeviceFunction[kernel, None]` (`compile_function_unchecked` doesn't exist
    in this nightly, and the checked return embeds the arg list so it can't convert).
    `doWork`/`_batchRun`/`batchedForwardMultiStream` now take one `kernels` arg; call sites in
    main.mojo + profile_gpu.mojo are one-liners. Accuracy + fps unchanged.

- [x] **CLI: warn on unknown args** (`cli.mojo`) — DONE 2026-06-20
  - A typo (`--num-stream 5`, `--benchonly`) was silently ignored and the default silently used —
    worst failure mode for a benchmarking knob. `CliArgs.parse` now raises on any unrecognized
    `--`-prefixed flag (fail loud). `cli.mojo` is the live parser (main.mojo imports it);
    `cliparser.mojo` is dormant (no importers).
  - NOT porting the MojoLLM `cliparser.mojo` (TokenizerParser) pattern now — see the reflection
    rewrite below. The fail-loud behavior is parser-agnostic and portable to whatever wins.

- [x] **`MNISTDataRepository.__init__` swallows read errors** (`dataloader.mojo`) — DONE 2026-06-12
  - Constructor is now `raises`; the try/except-print is gone, read failures propagate.

- [x] **`getResults`: bounds-check `len(labels)` vs `batch_size`** (`accel/ops.mojo`) — DONE 2026-06-12
  - Raises with both lengths in the message if the labels span is short.

- [x] **`printerGPU`: drop the inner `DeviceContext`** (`accel/ops.mojo`) — DONE 2026-06-12
  - `map_to_host` already syncs; spurious ctx + synchronize removed.

---

## Upstream Bug Reports to File

- [x] **`tensor.shape[N]()` dynamic on a struct-FIELD LayoutTensor — EXPECTED (Discord)** (`ignoreme/shape_comptime_mwe.mojo`) — RESOLVED 2026-07-01
  - `shape[N]()` folds to COMPTIME on a direct `LayoutTensor` PARAMETER but is DYNAMIC through a runtime
    struct field (`lenet.weight5_6.shape[0]()` → "cannot use a dynamic value in comptime initializer";
    definition-time error, `comptime if` / parametric gating don't hide it). Discord verdict: EXPECTED —
    the compiler can't prove a field reached through a runtime instance is the statically-known one, so it's
    conservatively dynamic. FIX = read from the TYPE: `type_of(lenet.weight5_6).shape[0]()` or the layout-
    tensor alias `.shape[0]()` (both fold to comptime — verified in the MWE). NOT a bug; nothing to file.
    Superseded for the GPU kernels by the "take fields directly" refactor above (direct params dodge it).

- [ ] **Ask Discord: `trait_downcast` lint not silenced by `comptime assert conforms_to`** (`resultlogger.mojo:31`)
  - Nightly `1.0.0b3.dev2026062906` warns "use `conforms_to(type_of(src), Trait)` instead in a `where`
    clause or `comptime assert`" on `trait_downcast[Writable](fr)` in `reflectCSV`. Adding
    `comptime assert conforms_to(type_of(fr), Writable), ...` does NOT silence it — the stdlib silences
    via `_constrained_conforms_to` (`builtin/constrained.mojo:87`), which takes `conforms_to(...)` as a
    comptime PARAMETER (not a statement). Ask: is a statement-form `comptime assert` supposed to satisfy
    the lint, or must the guard be in parameter/`where` position? Keeping the assert for its clean
    diagnostic; living with the one warning until resolved.

- [ ] **`convoluteForward` slice syntax — API bug** (`cpu/ops.mojo:389`)
  - Slice call requires specific form that differs from what docs describe. File Mojo issue.

- [ ] **`convoluteForward` slice IndexList vs Int — docs bug** (`cpu/ops.mojo:405`)
  - Docs say `IndexList` is expected but passing `Int` is needed. File Mojo issue.

- [ ] **`Scalar.from_bytes` big-endian flag — compiler issue** (`cpu/model.mojo:364`)
  - FIXME: passing a `big_endian=` flag to `from_bytes` had compiler issues, so `bytesToFType` falls back
    to the default + a manual swap (`cpu/model.mojo:433`). Make a minimal reproducer; investigate or file.

---

## From Blog Draft (`ideas.typ`)

Surfaced while editing the writeup. Code items first; "(writeup)" items are research/verification
needed so the prose is accurate, not necessarily code changes.

- [ ] **Parameterize `Image` on origin** (`image.mojo`) — `PixelTensor`/`DataTensor` use `MutAnyOrigin`
  today; carry a real `origin` so the borrow checker enforces the repo outliving the image instead of
  relying on "the MNIST repo stays alive long enough." Related to the Session pattern. (ideas.typ §Images, §Model)

- [ ] **(Optional, low priority) `MNISTDataView`: split into TWO origin params** (`dataloader.mojo`)
  - Today both `raw_pixels`/`raw_labels` are `Span[UInt8, Self.origin]` — ONE abstract mutable origin.
    That made `__getitem__`'s sub-view trip the exclusivity checker (two same-origin mutable spans to
    one ctor look like aliasing even though they index disjoint buffers). Patched 2026-06-25 by making
    the sub-view immutable + untracked (`ImmutUntrackedOrigin`) — fine because the slicer is read-only
    and was unused. See [[project-register-passable-origin]].
  - The principled version (only if sub-views get real use, or to show origins off in the writeup):
    give the struct distinct `p_origin`/`l_origin` params so pixels/labels provably don't alias →
    exclusivity error gone, sub-view stays MUTABLE + tracked. Fully compatible with
    `TrivialRegisterPassable` (which IS the right trait here: it's a tiny non-owning 2-span view, so
    trivial bitwise copy + no-op destroy are correct). Cost: mildly viral — every `MNISTDataView[origin]`
    annotation (`accel/ops.mojo`, the two repo builders) becomes `[p_origin, l_origin]`, and
    `origin_of(self.raw_pixels, self.raw_labels)` unions become useful for any method returning a ptr
    into either buffer. Pairs naturally with the `Image`-on-origin coloring above. (ideas.typ §origins)

- [x] **Load MNIST into `[1, 28, 28]` (explicit channel dim)** — DONE 2026-06-25
  - `Image.PixelLayout` `[28,28]` → `[INPUT,28,28]`; `DataLayout` now ALIASES `FeatureLayouts.input`
    (`[1,32,32]`) since the padded/normalized image *is* the feature input (PADDED_SIZE ==
    LENGTH_FEATURE0). `normalized()`/`_normalize()` write `tensor[0, r, c]`; deprecated debug
    `FeatureGPUBuffers.loadInput` indexes `pixels[0, i, j]`. `.size()` unchanged (784/1024) so all
    dataloader byte math + memcpy counts hold — dataloader needed ZERO changes (all `.size()`/flat
    `.ptr`). C = INPUT (parameterized for the conv1 multi-channel TODO), not literal 1.
  - GPU batch-pixel staging left as flat `[N,H,W]` transport (note added in `ops.mojo`); the
    normalize kernel already writes the channel into `feats[img].input[0,...]`.
  - Pure rank/type change, no memory layout change: CPU 9648/10000 + GPU 9648/10000 (1.32M fps) match.

- [ ] **`test_data` / `train_data` as `Span`s natively** (`dataloader.mojo`) — instead of `List[Image]`.
  Narrower cousin of "Kill `List[Image]` from CPU hot path." (ideas.typ §Data Loading)

- [x] **Use `with open(...)` context manager for MNIST files** (`dataloader.mojo`) — DONE 2026-06-20
  - Both handles now opened via nested `with open(...)`; manual `.close()` calls gone (auto-closed on
    scope exit, including the error path). Done together with the reader collapse below.

- [x] **Collapse `_readData` into one private `@staticmethod`** (`dataloader.mojo`) — DONE 2026-06-20
  - `_readTrainData`/`_readTestData` (near-identical) replaced by one `_readSplit(image_file,
    label_file, count, mut pixels_arena, mut labels_arena, mut data, split)` staticmethod. Callers in
    `__init__` pass the matching destination fields — disjoint, so separate `mut` args avoid a
    whole-`self` borrow. Build green, accuracy unchanged.

- [x] **Finalize reflection usage in the Logger** (`resultlogger.mojo`) — DONE 2026-06-25
  - Both `getHeaders` AND `toCSV` are now reflection-driven via two free helpers: `reflectHeaders[T]`
    (field NAMES) and `reflectCSV[T](ref s)` (field VALUES via `reflect[T].field_ref[i](s)` +
    `trait_downcast[Writable](...)` — both builtins). Adding a CSV column = adding a struct field;
    the hand-written concatenation is gone. The one blocker — `ftype: DType` reflects as lowercase
    `float32` — was fixed by materializing it to a `String` field ("Float32"/"Float64") in `__init__`,
    same trick already used for `activation_fn` (`reflect[act_fn].base_name()`). Output is
    byte-identical to the old format; full project builds; CSVs verified. Clean to show in the writeup.
  - Gotcha for the writeup: pass `trait_downcast[Writable](fr)` DIRECTLY into `String(...)` — binding
    it to a `var` first fails (the downcast existential isn't `ImplicitlyCopyable`).

- [ ] **Add a simple divide-by-constant MNIST normalization option** (`accel/ops.mojo`
  `normalizeInputsKernel`, `image.mojo`) — alongside the per-image mean/std path; closer to what the
  other libs do and a cleaner apples-to-apples. (ideas.typ §Fixing Old Mistakes aside)

- [ ] **CLI: reflection-driven parser (clap-derive / moclap style)** (`cliparser.mojo`) — replace the
  per-type `get[T]` overloads; or evaluate adopting `moclap`. (ideas.typ §CLI Parsing)
  - BLOCKED ON `__extension__` (coming to Mojo, not yet landed). The clean design needs a
    `ConvertibleFromString` trait that built-in types (`Int`, `Float64`, `DType`, `Bool`) conform to;
    today you can't add that conformance to types you don't own, forcing the per-type `get[T]`
    overload workaround. With extensions you retroactively conform them, then a reflection pass fills
    a `CliArgs`-shaped struct field-by-field from flags (clap-derive). Don't build the awkward
    pre-extension version — it'd be thrown away. Until then `cli.mojo`'s explicit parse stays; the
    fail-loud unknown-flag fix already covers the worst failure mode.

- [ ] **Benchmark the GPU arena allocator** (`accel/arena.mojo`) — CPU arena gave ~20%; the GPU-side
  arena gain is asserted but unmeasured ("benchmarks haven't been done"). (ideas.typ §GPU Pinned Memory)

- [ ] **(writeup) Re-verify old-version benchmark numbers** (C, first CUDA, first Mojo) before citing
  them in the "Old Versions" comparison — current numbers are "ones I don't trust." (ideas.typ §Old Versions)

- [ ] **(writeup) Document GPU thread/warp → hardware mapping** — answer concretely: if a block doesn't
  use a full 32-lane warp, what happens to the idle lanes; can leftover warps form another block?
  Needed for the Custom Kernels section to be correct. (ideas.typ §Custom Kernels)

- [ ] **(writeup) Pin down when `comptime(N)` / materialize is required** around `vectorize` — so the
  SIMD aside explains it rather than hand-waving. (ideas.typ §SIMD Aside)

- [ ] **(writeup) Find the nsys marker for CUDA-graph stream capture** — only relevant if/when graphs are
  added (currently in the "passed on" list). (ideas.typ §Custom Kernels)
