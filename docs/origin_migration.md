# Origin Migration — surviving the "no AnyOrigin in struct fields" nightly

A mid-2026 nightly tightened the origin system. Code that compiled for months
suddenly failed with:

```
error: struct fields cannot expose AnyOrigin in their type;
       pixels has type 'Image.PixelTensor'
```

This doc records what broke, the two ways out, which one we took, and the exact
patterns so the next person (or the eventual "approach B" migration) has a map.

## Status update (2026-06-25): approach A is *correct*, not a stopgap

The upstream rename proposal —
<https://github.com/modular/modular/blob/main/mojo/proposals/untracked-and-any-origin-renames.md>
— reclassifies the two origins this migration touched, and it lands in our favor:

| Old name | New name | Proposal's verdict |
|----------|----------|--------------------|
| `MutExternalOrigin` | `MutUntrackedOrigin` | "a **legitimate, supported tool**" for external / hand-managed memory |
| `MutAnyOrigin` | `MutUnsafeAnyOrigin` | "temporary escape hatch… **will never be stabilized**… slated for removal" |

Takeaways (so nobody "fixes" what's already right):

- **We moved fields + stored-`DeviceFunction` kernel params FROM the deprecated
  origin TO the supported one.** That's the sanctioned direction, not a hack.
  `untrack()` / `untrack_imm()` are keepers, not scaffolding.
- **Field side is already future-proof.** `MutUntrackedOrigin` *is* the
  post-rename name — no edit needed when the rename lands.
- **The `*Session` pattern (`DeviceSession`/`CPUSession`) makes approach B
  largely unnecessary.** A session owns arena + views together, so a view can't
  outlive its buffer — the lifetime boundary is structural. `UntrackedOrigin`
  ("compiler isn't tracking; I manage it by hand") is honest here: the Session
  *is* that management. B would re-derive, virally across the host→device
  boundary, a guarantee the Session already gives by construction. **Not worth
  it.** Polish A; don't pursue B.
- **Only forward debt:** the ~60 surviving `MutAnyOrigin` *function arguments*
  (activation `forward`/`backward`, `convoluteForward/Backward`, loss helpers)
  become `MutUnsafeAnyOrigin` and will warn someday. The clean fix is
  origin-generic args (`o: MutOrigin` for writers, `_` for readers — already done
  for `_randHelper`), but it "colors" ~60 leaf signatures and cascades through
  every activation impl's `backward`. **Deliberately deferred** — do it in one
  mechanical sweep if/when the rename actually starts warning, not before.

## What the compiler now forbids

`MutAnyOrigin` / `ImmutAnyOrigin` are *erased* origins — they name "some origin,
trust me." The nightly bans them in two structural positions:

1. **Struct fields.** `var pixels: LayoutTensor[..., MutAnyOrigin]` is rejected.
   A stored field with an erased origin lets a view silently outlive its
   backing buffer, so the compiler refuses to track it.
2. **Kernel signatures, transitively.** A `DeviceFunction` value (what
   `ctx.compile_function[k]()` returns) embeds the kernel's *argument list* in
   its type. If any kernel param is `UnsafePointer[FeatureGPU, MutAnyOrigin]`,
   then storing that `DeviceFunction` in a field re-exposes `AnyOrigin` —
   same error, one level removed. This bit `CompiledKernels`.

What is **still allowed**: `*AnyOrigin` on plain **function/kernel arguments**
that aren't captured into a stored `DeviceFunction`. Only *storage* is the
problem.

## The key conversion rule (learned by probing the compiler)

Argument passing is **not** symmetric:

| Direction (by value)                     | Works? |
|------------------------------------------|--------|
| `MutUntrackedOrigin` value → `MutAnyOrigin` param | ✅ yes (Any is the top) |
| `MutAnyOrigin` value → `MutUntrackedOrigin` param | ❌ no implicit conversion |

Consequence: after moving a **field** to `MutUntrackedOrigin`, by-value calls
into helpers that still take `MutAnyOrigin` args *just work* (e.g.
`LeNet5.forward`, `convoluteForward`). Only two situations force a change:

- **`mut`-ref params** need an exact origin match — make them origin-generic.
- **Kernel params whose `DeviceFunction` is stored** must themselves become a
  concrete origin, and then the launch-site value must match (no Any→Untracked).

Also: the bare `UntrackedOrigin` / `AnyOrigin` names are parametric over `mut`,
so they can't be dropped into a slot that needs a concrete `Origin[mut=…]`. Use
the resolved **`MutUntrackedOrigin`** / **`ImmutUntrackedOrigin`**.

## Two approaches

**A — `*UntrackedOrigin` (escape hatch).** Swap the erased origin for an
*untracked* one: still no provenance tracking (same (un)safety as before), but
it's a concrete origin the field/signature rules accept. Lifetime is managed by
hand — which is already true for every arena- and device-buffer-backed view in
this project.

**B — real origin parameters.** Parameterize each view-holding struct by the
origin of its backing storage (`struct Image[origin: MutOrigin]: …`). The
compiler then tracks provenance and turns use-after-free into a compile error.
Correct, but viral: the `[origin]` param propagates through every holder, every
by-value function, and — the expensive part — across the host→device boundary
into kernel signatures and the device-type encoding.

**We took A**, deliberately, to get green without threading origins through the
GPU boundary. B remains "overdue" but is a separate sit-down refactor. The
field-declaration edits A makes are *shared* with B; only the `untrack()`
wrappers at constructors are throwaway.

## The scaffolding: `src/origin_util.mojo`

Two one-line helpers do all the work. They build a `LayoutTensor` naturally
(origin inferred from the buffer) and rebind the result down to an untracked
origin for storage:

```mojo
def untrack[
    dt: DType, l: Layout, mut: Bool, //, o: Origin[mut=mut]
](t: LayoutTensor[dt, l, o]) -> LayoutTensor[dt, l, MutUntrackedOrigin]:
    return rebind[LayoutTensor[dt, l, MutUntrackedOrigin]](t)

def untrack_imm[...] -> LayoutTensor[dt, l, ImmutUntrackedOrigin]:
    ...  # read-only counterpart
```

All params are inferred, so call sites are clean regardless of the source
(DeviceBuffer, UnsafePointer, Span). When approach B happens, deleting these two
functions and every `untrack(` call is the mechanical part of the cleanup.

## Patterns applied

**Field declaration** — straight token swap:

```mojo
# before
var pixels: LayoutTensor[DType.uint8, Self.PixelLayout, MutAnyOrigin]
# after
var pixels: LayoutTensor[DType.uint8, Self.PixelLayout, MutUntrackedOrigin]
```

**Constructor of a `LayoutTensor` field** — wrap in `untrack(...)`, drop the
explicit origin so it's inferred:

```mojo
# before
self.input = LayoutTensor[ftype, Self.input_layout, MutAnyOrigin](buf)
# after
self.input = untrack(LayoutTensor[ftype, Self.input_layout](buf))
```

**`UnsafePointer` field** — there's no `untrack` for raw pointers; rebind
inline:

```mojo
# before
self.buffer = alloc[UInt8](capacity_bytes)        # field is MutAnyOrigin
# after  (field is MutUntrackedOrigin)
self.buffer = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
    alloc[UInt8](capacity_bytes)
)
```

**`mut`-ref helper** — make the origin a parameter instead of hardcoding it:

```mojo
# before
def _randHelper[layout: Layout](
    mut tensor: LayoutTensor[ftype, layout, MutAnyOrigin], scale: sftype): ...
# after
def _randHelper[layout: Layout, o: MutOrigin](
    mut tensor: LayoutTensor[ftype, layout, o], scale: sftype): ...
```

**Kernel params behind a stored `DeviceFunction`** — change the kernel
signature to a concrete origin, then make the launch-site value match:

```mojo
# kernel param
feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]      # was MutAnyOrigin
raw_pixels: LayoutTensor[..., ImmutUntrackedOrigin]        # was ImmutAnyOrigin

# launch site — value must already be untracked (Any→Untracked won't convert)
var raw_pixels_tensor = untrack_imm(
    LayoutTensor[DType.uint8, batch_pixels_layout](self.device_inputs)
)
```

The `StreamSlot` fields feeding these launches (`features_ptr`, `outputs`,
`guesses`) were moved to `MutUntrackedOrigin`, so they line up with the new
kernel params for free.

## Files touched

| File | What changed |
|------|--------------|
| `src/origin_util.mojo` | **new** — `untrack` / `untrack_imm` helpers |
| `src/image.mojo` | `PixelTensor` field + 3 ctors |
| `src/cpu/model.mojo` | `LeNet5` + `Feature` fields/ctors; `_randHelper` origin-generic |
| `src/cpu/arena.mojo` | pointer fields rebound; test-struct tensor fields |
| `src/accel/model.mojo` | `LeNet5GPU` weight/bias fields + ctors |
| `src/accel/feature.mojo` | `FeatureGPU` layer fields + ctors |
| `src/accel/ops.mojo` | `StreamSlot` fields; 7 kernel param origins; launch-site tensor |
| `src/dataloader.mojo` | deleted broken stub ctor (the free origin `o` never unified with `Self.origin`); `@fieldwise_init` provides the real two-span ctor |

## Footnote: kernel-internal locals were left alone

`MutAnyOrigin` on a `var` **inside** a kernel body (shared-memory
`stack_allocation`, scratch tensors) does *not* hit either rule — it's neither a
field nor part of the signature — so those were intentionally not touched.

## Deprecation warnings — cleared

The initial migration left 8 warnings; a follow-up pass (commit `1664b4c`)
took them to zero **without** touching arena design, using the compiler's
preferred "keep a concrete origin" path:

- `ImplicitlyDestructible` → `ImplicitlyDeletable` (`accel/arena.mojo`)
- `cpu/arena.mojo`: `alloc()` now returns `MutUntrackedOrigin` (matching the
  buffer field it carves from) instead of widening to `MutAnyOrigin`; the system
  allocator rebinds its free-`alloc` pointer on return. Every caller already
  pipes the result through `untrack()`, so nothing downstream changed.
- `cpu/model.mojo` `loadInput`: build the `DataTensor` via `untrack(...)` rather
  than constructing a `MutAnyOrigin` view from an already-untracked `.ptr`.
- `accel/ops.mojo` `_batchRun`: see automatic parameterization below.

### Automatic parameterization (`_`) and the mutability catch

`_` in a parameter slot unbinds it (Mojo auto-creates an inferred parameter).
For an origin, `_` unbinds the origin **but leaves its mutability free**
(`Origin[mut=_]`). That has a sharp consequence:

| Body of the function | `_` for the origin |
|----------------------|--------------------|
| only reads / passes the view through | ✅ works, and is terser |
| **writes** through the view | ❌ `error: expression must be mutable` |

`_batchRun` only calls `self` (non-`mut`) methods on its slots, so
`UnsafePointer[StreamSlot[batch_size], _]` is fine and removes an explicit
origin parameter. `_randHelper` writes `tensor.ptr[i] = …`, so it must keep
`[layout: Layout, o: MutOrigin]` — the explicit `MutOrigin` bound carries the
`mut=True` the body depends on; `_` would force the body to also handle
`mut=False` and fail. The `untrack` helpers likewise keep named `dt, l, o`
because those parameters are referenced in the return type.

Rule of thumb: reach for `_` on pass-through/read-only views; keep an explicit
`MutOrigin` (or named param) wherever the body mutates or needs to name the
parameter elsewhere.

## Reference

A minimal, self-contained comparison of **A vs B vs C** (all compile and run;
every claim compiler-verified, including B's use-after-move catch) lives in
`ignoreme/origin_mwe.mojo`. Read that first to feel the trade-offs:
A = store view @ untracked (no tracking); B = store view @ real origin param
(viral, but UAF is a compile error); C = store the *storage*, rebuild views
per-method @ `origin_of(self.storage)` (tracked to self, no viral param, no
`untrack()` — the "tracking back without going viral" door).
