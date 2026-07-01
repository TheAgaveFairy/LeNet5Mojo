from layout import Layout, LayoutTensor
from std.bit import next_power_of_two  # prev_power_of_two

from std.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    HostBuffer,
    DeviceFunction,
)
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives import block
from std.gpu.memory import AddressSpace

from constants import (
    ftype,
    sftype,
    act_fn,
    FeatureLayouts,
    WeightLayouts,
    BiasLayouts,
)
from accel.model import LeNet5GPU
from accel.feature import FeatureGPU, FeatureGPUBuffers
from origin_util import untrack, untrack_imm

def gemm3[
        batch_size: Int
](
    lenet: LeNet5GPU,
    feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin],
    outputs: LayoutTensor[
        ftype, Layout.row_major(batch_size, OUTPUT), MutUntrackedOrigin
    ],
    guesses: LayoutTensor[
        DType.uint8, Layout.row_major(batch_size), MutUntrackedOrigin
    ],
) -> None:
    var img_idx = block_idx.x
    var r = thread_idx.y
    var c = thread_idx.x
    

def matMulFusedKernel[
    batch_size: Int
](
    lenet: LeNet5GPU,
    feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin],
    outputs: LayoutTensor[
        ftype, Layout.row_major(batch_size, OUTPUT), MutUntrackedOrigin
    ],
    guesses: LayoutTensor[
        DType.uint8, Layout.row_major(batch_size), MutUntrackedOrigin
    ],
) -> None:
    """
    Enough threads per block to do one output channel at a time as a reduction,
    so make it a power of two.
    Grid Dim = batch_size
    Block Dim = next_power_of_two(in_chans).
    Writes logits into the batched outputs tensor (gather fused away) and the
    argmax into guesses — thread 0 sees every logit sequentially, so the running
    max is free and only 1 byte/img needs the trip back to host.
    """
    var img_idx = block_idx.x
    var thread = thread_idx.x
    comptime reduction_size = next_power_of_two(LAYER5)  # 120 -> 128

    # TODO: dram to local call possible
    var feat = feats[img_idx].layer5[thread, 0, 0] if thread < LAYER5 else 0
    var best = sftype.MIN
    var best_idx: UInt8 = 0

    comptime for oc in range(OUTPUT):
        var weight = lenet.weight5_6[thread, oc] if thread < LAYER5 else 0
        var prod = feat * weight
        var answer = block.sum[block_size=reduction_size, broadcast=False](prod)
        if thread == 0:
            var logit = rebind[sftype](
                answer + rebind[sftype](lenet.bias5_6[oc])
            )
            outputs[img_idx, oc] = logit
            # raw logits by design: no act_fn after the final FC layer
            if logit > best:
                best = logit
                best_idx = UInt8(oc)

    if thread == 0:
        guesses[img_idx] = best_idx

