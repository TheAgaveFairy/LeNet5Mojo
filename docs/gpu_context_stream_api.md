# GPU Context & Stream API — Mojo stdlib findings

## Core model: `DeviceContext` IS a stream

The docstring says it plainly: `DeviceContext` represents **"a single stream of execution on a particular accelerator."** Every `enqueue_*` call on a context goes to its own internal stream. Creating two `DeviceContext` objects creates two independent streams on the same device.

```mojo
var ctx_a = DeviceContext()   # stream A
var ctx_b = DeviceContext()   # stream B — truly independent
```

## Buffer ownership determines which stream copies run on

`DeviceBuffer` and `HostBuffer` each hold a reference to the `DeviceContext` that created them. All copy helpers delegate through `buffer.context().enqueue_copy(...)`:

```mojo
fn enqueue_copy_from(self, src: HostBuffer[...]) raises:
    self.context().enqueue_copy(self, src)   # always ctx's own stream
```

**Implication:** there is no way to put a buffer copy on a `DeviceStream` returned by `create_stream`. Copies always run on the owning context's stream. This is why `create_stream` + event dance is messy — the H2D copy will never be on the created stream.

## `create_stream` vs multiple `DeviceContext`

| | `ctx.create_stream()` | separate `DeviceContext()` |
|---|---|---|
| Independent execution | Yes | Yes |
| Buffer copies on stream | **No** — still on owning ctx | **Yes** — each buffer is owned by its ctx |
| Cross-stream sync | `stream.enqueue_wait_for(event)` + `stream.record_event(event)` | `ctx.enqueue_wait_for(other_ctx)` or `ctx.create_event()` |
| Kernel dispatch | `stream.enqueue_function(f, ...)` | `ctx.enqueue_function[f](...)` |
| CPU wait | `stream.synchronize()` | `ctx.synchronize()` |
| Shared device memory | Yes (CUDA memory is device-wide) | Yes |

**For `StreamSlot`, separate `DeviceContext` per slot is the right choice.** Buffer allocations, fills, copies, and kernel launches all land on the same stream automatically. No event plumbing required.

## Synchronization APIs

### Within a stream
```mojo
ctx.synchronize()             # CPU blocks until ctx's stream is done
stream.synchronize()          # same for a raw DeviceStream
```

### Cross-context (GPU-side, non-blocking CPU)
```mojo
ctx_b.enqueue_wait_for(ctx_a) # ctx_b's stream waits for ctx_a's stream to flush
```

### Cross-stream via events (when using `create_stream`)
```mojo
var event = ctx.create_event()           # create unrecorded event (timing disabled by default)
stream_a.record_event(event)             # record after stream_a's current tail
stream_b.enqueue_wait_for(event)         # stream_b waits before continuing — GPU-side, no CPU block
event.synchronize()                      # CPU block until event fires (use sparingly)
```

`create_event()` defaults to `disable_timing=True` (no overhead). Pass `blocking_sync=True` only if you need `event.synchronize()` to block efficiently.

## `DeviceContext.enqueue_copy` overloads (all on ctx's stream)

```mojo
ctx.enqueue_copy(dst_dev, src_host_ptr)   # H2D: ptr → DeviceBuffer
ctx.enqueue_copy(dst_host_ptr, src_dev)   # D2H: DeviceBuffer → ptr
ctx.enqueue_copy(dst_dev, src_dev)        # D2D
ctx.enqueue_copy(dst_dev, src_host_buf)   # H2D: HostBuffer → DeviceBuffer
ctx.enqueue_copy(dst_host_buf, src_dev)   # D2H: DeviceBuffer → HostBuffer
ctx.enqueue_copy(dst_ptr, src_ptr, size)  # D2D via raw pointers
```

All are truly async (non-blocking on CPU). `dst.context()` is used implicitly by `buffer.enqueue_copy_from/to(...)`.

## `DeviceContext.stream()` — low-level escape hatch

```mojo
var raw_stream: DeviceStream = ctx.stream()  # get the underlying DeviceStream
```

Marked `@doc_private`. Returns a `DeviceStream` handle to ctx's internal stream — useful for calling `record_event`/`enqueue_wait_for` on the same stream that `enqueue_copy` uses.

## Recommended `StreamSlot` design

```
StreamSlot owns:
  var ctx: DeviceContext          ← one context = one independent stream
  var device_arena: GPUBumpArenaAllocator(ctx, ...)
  var hosted_inputs: HostBuffer   allocated from ctx
  var device_inputs: DeviceBuffer allocated from ctx
  var hosted_outputs: HostBuffer  allocated from ctx
  var outputs_buffer: DeviceBuffer allocated from ctx
  ... FeatureGPUBuffers allocated from ctx's arena

loadBatch:
  hosted_inputs.enqueue_copy_from(batch_span_ptr)  → ctx stream
  device_inputs.enqueue_copy_from(hosted_inputs)   → ctx stream (H2D, after fill)

doWork:
  ctx.enqueue_function[norm](...) → ctx stream (after H2D, same stream → ordered)
  ... all kernels ...
  ctx.enqueue_function[gather](...) → ctx stream
  hosted_outputs.enqueue_copy_from(outputs_buffer)  → ctx stream (D2H, after gather)

getResults:
  ctx.synchronize()               ← CPU blocks until D2H done
  read from hosted_outputs
```

No events needed. Ordering is guaranteed because all ops share one stream.

## HostBuffer.enqueue_copy_from with Span

`HostBuffer.enqueue_copy_from` takes `HostBuffer` or `DeviceBuffer` — not `Span`. To copy a `Span[UInt8, _]` into a `HostBuffer`:

```mojo
ctx.enqueue_copy(hosted_inputs, span.unsafe_ptr())
# or: directly memcpy since HostBuffer is pinned host memory and unsafe_ptr() is accessible
```
