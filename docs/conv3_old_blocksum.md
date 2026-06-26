# The old conv3: `block.sum` reduction (superseded by Tier A)

Blog-prep note. Captures the structure of `conv3FusedKernelOld` (the original
block-reduction conv4_5 kernel) before it's deleted, because the contrast with
the current per-thread kernel is a good story: the "obvious" GPU mapping
(thread-per-input, reduce) loses to thread-per-output on these small dimensions.

## What conv4_5 computes

A fully-connected layer in conv clothing (`LENGTH_FEATURE5 = 1`, so the 5×5
kernel sees one window): `out[oc] = act( bias[oc] + Σ_{400} layer4[i] · W[oc,i] )`,
for `oc` in `0..LAYER5=120`, contracting `K = LAYER4·5·5 = 400` inputs.

## Old kernel: thread-per-input-product + `block.sum` (the intuitive mapping)

```
Grid  = (batch_size, div_chans_conv3 = 8)
Block = conv3_reduction_threads = 512  (1D; next_pow2(400))
```

- Each **block** handles `num_ocs = 120 // 8 = 15` output channels for **one image**.
- Each **thread** owns one of the 400 `(in_chan, row, col)` input elements
  (`flat_idx → in_chan/row/col`); threads `400..511` are padding, contribute 0.
- For each of the 15 output channels (a `comptime for` — *making it runtime
  dropped accuracy*, an unexplained quirk worth a footnote):
  - every active thread forms one product `feat_val · W[in_chan, oc, row, col]`,
  - `block.sum[block_size=512, broadcast=False](prod)` tree-reduces all 512
    threads' products to a single sum,
  - thread 0 adds bias, applies the activation, writes `layer5[oc]`.

So per block: **15 full 512-wide block reductions**, one per output channel.

### Why it was the natural first attempt
It mirrors the math directly: map each input×weight product to a thread, sum
them. Reductions feel like the canonical GPU primitive.

## Why it lost (→ current `conv3FusedKernel`, "Tier A")

The current kernel inverts the mapping:

```
Grid  = (batch_size)
Block = LAYER5 = 120   (one thread per OUTPUT channel)
```

Each thread does the **entire 400-MAC dot product in registers** (per-thread
accumulation), reading `layer4` once from shared memory — **no reduction, no
per-output barriers**.

Measured / structural reasons the block.sum version is worse here:

- **~7× worse ps/MAC** than per-thread register accumulation (nsys; see the
  GPU-bottleneck notes). The 15 tree reductions/block are pure overhead the
  per-thread version doesn't pay.
- **Wasted lanes:** 112 of every 512 threads (400..511) are padding that exists
  only to round the reduction width to a power of two.
- **`block.sum` launch gotcha:** the 1-D `block_size` overload *deadlocks* under
  a multi-dimensional launch — the per-output kernel sidesteps `block.sum`
  entirely. (See the block.sum-overload note.)
- These dimensions are tiny (M=120, K=400). A block-reduction shines when the
  contraction is large and reused; here the per-output dot product fits in
  registers and the reduction machinery is all cost.

## The arc for the writeup
intuitive map-reduce (thread = input product, `block.sum` per output) →
thread-per-output register accumulation (Tier A) → *next:* tiled GEMM with the
batch as the N dimension (Tier B), where the contraction finally gets big enough
(N = batch) that real tiling pays. conv4_5 and the final matmul are the **same**
dense layer, so one GEMM serves both — see `docs/` GEMM/SoA planning.

## Original code (preserved)

```mojo
def conv3FusedKernelOld[batch_size: Int](
    lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]
) -> None:
    comptime out_chans = LAYER5
    comptime div_chans = div_chans_conv3        # 8
    comptime num_ocs   = out_chans // div_chans # 15
    comptime ksq       = LENGTH_KERNEL * LENGTH_KERNEL

    var flat_idx  = Int(thread_idx.x)           # 0..511
    var img_idx   = block_idx.x
    var chans_set = block_idx.y
    var offset    = chans_set * num_ocs

    var active = flat_idx < conv3_feat_total    # 400
    var in_chan = 0; var row = 0; var col = 0
    var feat_val: sftype = 0
    if active:
        in_chan = flat_idx // ksq
        var rem = flat_idx % ksq
        row = rem // LENGTH_KERNEL
        col = rem % LENGTH_KERNEL
        feat_val = rebind[sftype](feats[img_idx].layer4[in_chan, row, col])

    comptime for oc in range(num_ocs):          # runtime loop drops accuracy (!)
        var prod: sftype = 0
        if active:
            prod = feat_val * rebind[sftype](
                lenet.weight4_5[in_chan, oc + offset, row, col])
        var total = block.sum[block_size=conv3_reduction_threads,
                              broadcast=False](prod)
        if flat_idx == 0:
            var biased = total + rebind[sftype](lenet.bias4_5[oc + offset])
            feats[img_idx].layer5[oc + offset, 0, 0] = act_fn.simdForward(biased)
```

(Note: on the current nightly this body no longer type-checks — `simdForward`
can't infer `width` from the scalar `biased` — which is how the build-everything
sweep surfaced it as dead code.)
