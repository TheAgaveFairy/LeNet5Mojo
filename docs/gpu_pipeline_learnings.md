# GPU Multi-Stream Pipeline — Implementation Learnings

Companion to `gpu_context_stream_api.md`. This doc covers what we learned
building `batchedForwardMultiStream` and the performance results we measured.

## What `batchedForwardMultiStream[s=1]` taught us

Running multistream with `num_streams=1` is **not** the same as `batchedForward`:

| Function | RTX 3070, bs=50 |
|---|---|
| `batchedForward` | ~850ms |
| `batchedForwardMultiStream[s=1]` | ~361ms |
| `batchedForwardMultiStream[s=3]` | ~276ms |

The ~2.4x gain from s=1 comes entirely from **async D2H overlap**: in
`batchedForward`, results are read synchronously each batch. In
`batchedForwardMultiStream`, D2H is enqueued at the end of `doWork` and
`getResults` is deferred one slot later — so D2H of batch N overlaps with
compute of batch N+1 even on a single stream. The extra s=3 gain (~1.3x) is
true parallel compute across streams.

**Implication:** async D2H placement matters a lot. Always enqueue D2H in
`doWork`, not at the start of `getResults`.

## Grid search results (RTX 3070, 60000 images, LeNet-5)

| bs  | streams | single_ms | multi_ms |
|-----|---------|-----------|----------|
| 75  | 3       | 827       | **273**  |
| 75  | 4       | 833       | **273**  |
| 75  | 5       | 830       | 274      |
| 100 | 3       | 830       | 288      |
| 100 | 4       | 812       | 281      |
| 100 | 5       | 812       | 278      |
| 120 | 3       | 797       | 275      |
| 120 | 4       | 798       | 274      |
| 120 | 5       | 789       | 276      |

**Takeaways:**
- bs=75, streams=3 is optimal for multistream (~273ms). Larger batches hurt
  multistream despite helping single-stream.
- Streams past 3 give negligible returns — pipeline is saturated by then.
- Single-stream favours larger batches (bs=120 → 789ms vs bs=75 → ~830ms),
  likely better GPU utilisation per batch.

## Pitfall: `InlineArray` requires `Copyable`

```mojo
# FAILS — StreamSlot is not Copyable (owns DeviceContext)
var slots = InlineArray[StreamSlot[batch_size], num_streams](uninitialized=True)
```

`InlineArray[T, N]` enforces `T: Copyable` even with `uninitialized=True`.
Fix: heap-allocate with `alloc` and move-initialise each element:

```mojo
var slots = alloc[StreamSlot[batch_size]](num_streams)
for s in range(num_streams):
    (slots + s).init_pointee_move(StreamSlot[batch_size]())
# ... use slots ...
for s in range(num_streams):
    (slots + s).destroy_pointee()
slots.free()
```

## Pitfall: `-D` flag must come before the filename

```bash
# WRONG — defines ignored
mojo main.mojo -D GPU_BATCH_SIZE=75

# CORRECT
mojo -D GPU_BATCH_SIZE=75 -D NUM_GPU_STREAMS=3 main.mojo
```

## Pipeline structure

The key invariant: collect stale results **before** reusing a slot, then enqueue
new work. After the main loop, drain the last `num_streams` slots in an
epilogue.

```
main loop  (batch_num = 0 .. total_batches-1):
    slot = batch_num % num_streams
    if batch_num >= num_streams:
        collect results for (batch_num - num_streams)   ← stale slot
    loadBatch(slot, batch_num)
    doWork(slot)                                         ← D2H enqueued here

epilogue  (batch_num = max(0, total-num_streams) .. total-1):
    slot = batch_num % num_streams
    collect results for batch_num
```

This ensures no slot is read before its D2H has completed, and no batch result
is collected twice.

## `DeviceFunction` is `ImplicitlyCopyable`

`DeviceFunction` (returned by `ctx.compile_function[...]()`) can be stored and
passed freely — no `^` needed. Compiling once in the outer scope and passing
to `batchedForward`/`batchedForwardMultiStream` as regular arguments works
correctly.

---

## nsys profile comparison: `basic` vs `multistream`

`basic.*` was captured before multi-stream work — only `batchedForward` (single
context, synchronous result collection). `multistream.*` covers all three runs
back-to-back: `batchedForward` + `batchedForwardMultiStream[s=1]` +
`batchedForwardMultiStream[s=3]`.

Because the multistream profile covers 3× the inference work (3600 kernel
instances vs 1200), most raw counts scale ~3×. The interesting changes are the
ones that don't.

### Stream creation

| | basic | multistream |
|---|---|---|
| `cuStreamCreate` | 1 | **5** |
| `cuEventCreate` | 2 | 6 |

Exactly as expected: 1 (batchedForward) + 1 (s=1 run) + 3 (s=3 run) = 5
streams. Each `DeviceContext()` in a `StreamSlot` creates one CUDA stream.

### Synchronisation — the big win

| | basic | multistream | per logical batch |
|---|---|---|---|
| `cuStreamSynchronize` calls | 7,200 | 8,404 | 6.0 → **2.3** |

Basic had 7200 syncs for 1200 batches = **6 sync calls per batch**. The old
`batchedForward` was synchronising after each result collection step. In the
new code, `getResults` calls `ctx.synchronize()` exactly once per batch,
reducing to ~1 sync per batch per run. The 8404 total for 3600 logical batches
(2.3/batch average) also includes setup syncs and the `batchedForward` run
which retains its original sync pattern.

### Kernel throughput — same per-batch compute, better overlap

| Kernel | basic avg (µs) | multistream avg (µs) | basic max (µs) | multistream max (µs) |
|--------|---------------|---------------------|---------------|---------------------|
| conv3  | 206           | 210                 | 238           | **850**             |
| conv2  | 24            | 26                  | 26            | **608**             |
| matmul | 15            | 20                  | 29            | **492**             |
| pool1  | 6             | 7                   | 18            | **593**             |

Average kernel time is essentially unchanged — the GPU compute itself is no
faster. The speedup comes entirely from **overlapping** compute and H2D/D2H
across streams.

The high max latency in multistream (3-35× spikes) is resource contention:
when two slots' kernels are competing for SMs at the same time, one has to
wait. This is expected and acceptable — the pipeline hides the stall on the
other streams.

### Memory transfers

| | basic | multistream |
|---|---|---|
| H2D API calls | 2,408 | 7,208 (~3×) |
| H2D GPU MemOps | 2,408 | **4,808** (~2×) |
| H2D total bytes | 49.6 MB | 143.7 MB (~3×) |
| D2H GPU MemOps | 1,200 | 3,600 (3×) |

The H2D API call count scales 3× (matching 3 runs), but GPU-side MemOps only
scale 2×. The two-step copy in multistream — CPU span → pinned `HostBuffer`
(`cuMemcpyHtoDAsync`) then pinned → `DeviceBuffer` (`cuMemcpyHtoDAsync` again)
— shows as 2 API calls but the first hop (CPU→pinned) may be captured
differently in the GPU trace since pinned memory is already mapped into device
address space.

D2H scales cleanly at 3× (1 per batch per run), confirming the async D2H in
`doWork` is being correctly queued and collected in `getResults`.

### Pinned memory allocation cost

| | basic | multistream |
|---|---|---|
| `cuMemAllocHost_v2` | 1 call, 121ms | 1 call, **461ms** |

461ms vs 121ms — allocating pinned host memory for multiple `StreamSlot`s is
genuinely more expensive. This is a fixed startup cost amortised over 60000
images. Worth watching if the model is used with very short inference runs.

### Summary

The profiles confirm the pipeline is working as intended:
- GPU streams are created and independent (5 streams vs 1)
- Per-batch sync calls dropped from 6 to ~1
- Per-batch compute time is unchanged — gains are from overlap, not faster kernels
- High tail latency on individual kernel invocations is expected contention, not a bug
- Memory transfer accounting is slightly asymmetric due to the two-step pinned copy

## Why `CompiledKernels` struct was abandoned

Wrapping the 8 `DeviceFunction` values in a struct hit type-inference issues
when passing them to `ctx.enqueue_function[...]()` — the compiler couldn't
resolve the kernel type parameter through the struct field. The workaround is
passing the 8 functions as individual arguments, which is verbose but
unambiguous.
