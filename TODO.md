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

- [ ] **`MNISTBatch` lifetime not tracked — callers must `keep(data_repo)`** (`dataloader.mojo`, `profile_gpu.mojo`)
  - `MNISTBatch` holds `Span[UInt8, ImmutAnyOrigin]` into `MNISTDataRepository`'s arena. `ImmutAnyOrigin`
    erases the borrow link, so Mojo may destroy the repo before GPU copies finish. Workaround: `keep(data_repo)`
    at call sites (already done in `profile_gpu.mojo`, `main.mojo` avoids this by calling `batchedForward` first).
  - Fix: parameterize `MNISTBatch` on the arena origin (the commented-out `is_mutable/origin` params are the
    right shape — just wire them up). Then `getTrainBatch` can return `MNISTBatch[origin=origin_of(self)]`,
    which the compiler will enforce outlives its spans.

- [ ] **Rename `MNISTBatch` — name is misleading when used as a full-dataset view** (`dataloader.mojo`)
  - `getTrainBatch(0, 60000)` returns the whole dataset, not a batch. Consider `MNISTData` / `MNISTDataView`.
  - Add `.slice(i, batch_size) -> MNISTBatch` method so batch-level access is explicit at the call site.

- [ ] **Define `AcceptsAllocator` trait** (`accel/model.mojo:54`, `accel/model.mojo:91`)
  - `LeNet5GPUBuffers` and `LeNet5GPU` both carry an `allocator_owns_memory: Bool` workaround
    and duplicate `sizeInBytes()` statics. A trait would unify this pattern.

- [ ] **(Future) Kill `List[Image]` from CPU hot path; share SoA spans for both CPU and GPU** (`dataloader.mojo`)
  - CPU path currently uses `List[Image]` (AoS, 785-byte stride with label interleaved).
  - GPU path already uses SoA pixel/label arenas exposed via `MNISTBatch`.
  - If CPU profiling shows packing cost, migrate CPU ops to consume `raw_pixels`/`raw_labels` spans
    directly. `Image` becomes a lightweight view into the arena rather than an owned struct.

---

## GPU Pipeline

- [ ] **Ping-pong streaming: overlap H2D copy with compute** (`accel/ops.mojo`, `main.mojo`)
  - Depends on `batchedForward` refactor above. Once the caller owns the loop, allocate two pixel
    staging buffers and two `DeviceContext` streams; submit next batch H2D before syncing compute.
  - Reference pattern in `GPU_STREAMS_REFERENCE.md` (Step 2 — two streams).

- [ ] **Wire up GPU inference logger** (`main.mojo:281`)
  - `logger.logInferenceResult(device_name, elapsed, correct, COUNT_TRAIN, batch_size, ftype)`
    is commented out after `batchedForward`. Uncomment and hook up.

- [ ] **`singleForward`: `gpu_guess` sentinel should be `Optional[Int]`** (`accel/ops.mojo:639`)
  - `var gpu_guess = 10` uses magic number 10 as "invalid". Typed `Optional` is cleaner.

- [ ] **Confirm matmul output: skip or apply `act_fn.simdForward()`?** (`accel/ops.mojo:160`)
  - `matMulFusedKernel` writes raw dot product + bias to `feats[img].output` without activation.
  - CPU `matmulForward` also skips activation with a FIXME. Decide and document the choice.

- [ ] **Kernel wrapper functions may be unnecessary now that `ctx` is passed** (`accel/ops.mojo:76`)
  - `conv1Forward`, `normalizeInputs`, etc. are thin wrappers around `ctx.enqueue_function`.
  - Consider force-inlining or calling kernels directly. Evaluate if the abstraction is worth keeping.

- [ ] **conv2 image loading could be more efficient** (`accel/ops.mojo:406`)
  - Comment: "could make this much more efficient." Candidate for shared-mem load optimization.

- [ ] **conv1 kernel: `INPUT > 1` not handled** (`accel/ops.mojo:501`)
  - Kernel hardcodes single-channel input. If ever extended beyond MNIST (grayscale), this breaks.

- [ ] **Implement `LeNet5GPUBuffers.__del__`** (`accel/model.mojo:115`)
  - Placeholder comment exists. Needed to properly release GPU memory when not using an arena.

- [ ] **`comptime for` explodes compile time in arena/buffer setup** (`accel/ops.mojo:746`)
  - Noted during `batchedForward` implementation. Investigate if a runtime loop is acceptable.

---

## CPU Pipeline

- [ ] **`predict` / `predictNew` could be methods of `LeNet5`** (`cpu/ops.mojo:691`)
  - Standalone free functions that take `lenet` as first arg are natural method candidates.

- [ ] **`trainBatchParallel` accumulation is single-threaded** (`cpu/ops.mojo:751`)
  - The loop that calls `buffer.accumulateFromOther(deltas[i], 1.0)` runs serially after `parallelize`.
  - Should use atomics or a critical section, or restructure to reduce into a tree.

- [ ] **`testingParallel`: handle `len(data) % batch_size != 0`** (`cpu/ops.mojo:942`)
  - Current loop silently drops the tail. Add a remainder pass or assert even divisibility.

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

- [ ] **Remove or fix dead `MNISTBatch.__init__(images: List[Image], ...)` overload** (`dataloader.mojo`)
  - This custom init is never called. It was broken (wrong memcpy count). Delete it.

- [ ] **`image.mojo`: implement or remove `padded: Bool` flag** (`image.mojo:64`, `image.mojo:97`)
  - Both `normalized()` and `_normalize()` have a `padded: Bool = True` param that does nothing.
  - Either wire it up (zero-pad vs crop paths) or drop the parameter.

- [ ] **Delete `Image._normalize` static method** (`image.mojo`)
  - Already marked `@deprecated`. Once all callers confirmed migrated to `self.normalized(tensor)`, delete.

- [ ] **Consider merging `getTrainBatch`/`getTestBatch` into one method** (`dataloader.mojo:137`)
  - Comment suggests a single method with a `test_or_train` arg (or a `DataSplit` enum).

- [ ] **`_readData` (deprecated): skip `InlineArray` intermediate** (`dataloader.mojo:193`)
  - Dead method anyway — delete or keep the idea as a comment for future `_readTrainData` optimization.

---

## Logging (`resultlogger.mojo`)

- [ ] **Implement JSON log format** (`resultlogger.mojo:12`, `resultlogger.mojo:29`)
  - `LogFormat.JSON` is defined but `toCSV`/`getHeaders` are the only trait methods. Add `toJSON`.

- [ ] **`LeNet5Logger.test_size` field — clarify or improve** (`resultlogger.mojo:96`)
  - Comment says "kinda just gonna be the batch_size since CPU only for now." Revisit when GPU
    logging is wired up.

- [ ] **`headers_written: Bool` — make this better** (`resultlogger.mojo:191`)
  - Likely should be handled at file-open time or via a first-write check, not a mutable bool field.

---

## Cleanup / Dead Code

- [ ] **Remove unused `reflect` import in `main.mojo`** (`main.mojo:11`)

- [ ] **FIXME: `CPUSession` — tie arena + model lifetimes together** (`main.mojo:224`)
  - `benchmark.keep(arena)` is used to keep the arena alive alongside `arena_model`.
  - A `CPUSession[LeNet5]` struct (like `DeviceSession` on GPU) would make joint lifetimes explicit.

- [ ] **`FeatureGPUBuffers` still exists in `accel/model.mojo` after split to `accel/feature.mojo`**
  - Both files define `FeatureGPUBuffers` and `FeatureGPU`. The one in `accel/model.mojo` appears
    to be the old version. Confirm which is active and delete the other.

---

## Upstream Bug Reports to File

- [ ] **`convoluteForward` slice syntax — API bug** (`cpu/ops.mojo:389`)
  - Slice call requires specific form that differs from what docs describe. File Mojo issue.

- [ ] **`convoluteForward` slice IndexList vs Int — docs bug** (`cpu/ops.mojo:405`)
  - Docs say `IndexList` is expected but passing `Int` is needed. File Mojo issue.
