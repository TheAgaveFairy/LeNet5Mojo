"""GPU forward-pass kernels and the batched multi-stream inference pipeline."""

from layout import Layout, LayoutTensor, lt_to_tt
from layout import (
    row_major as tt_row_major,
)  # new-style Layout for TileTensor APIs
from layout.tile_io import copy_dram_to_sram_async
from std.math import abs, sqrt, max, min, ceildiv
from std.bit import next_power_of_two  # prev_power_of_two
from std.memory import memcpy, memset_zero
from std.sys import size_of, stderr
import std.sys.defines as defines

from std.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    HostBuffer,
    DeviceFunction,
)
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    barrier,
    global_idx,
    WARP_SIZE,
)
from std.gpu.primitives import block
from std.gpu.memory import AddressSpace, async_copy_wait_all

from constants import (
    ftype,
    sftype,
    act_fn,
    LENGTH_KERNEL,
    LENGTH_FEATURE0,
    LENGTH_FEATURE1,
    LENGTH_FEATURE2,
    LENGTH_FEATURE3,
    LENGTH_FEATURE4,
    LENGTH_FEATURE5,
    GPU_TILE_SIZE,
    INPUT,
    LAYER1,
    LAYER2,
    LAYER3,
    LAYER4,
    LAYER5,
    OUTPUT,
    IMAGE_SIZE,
    PADDED_SIZE,
    PADDING,
    NUM_GPU_STREAMS,
    GPU_STREAM_BATCH_SIZE,
    DISPLAY,
    FeatureLayouts,
    BatchedFeatureLayouts,
    WeightLayouts,
    BiasLayouts,
)
from accel.model import LeNet5GPU
from accel.feature import FeatureGPUBuffers
from accel.arena import GPUBumpArenaAllocator
from origin_util import untrack, untrack_imm
from dataloader import MNISTDataView

# IME these don't change performance a ton
comptime div_chans_conv2 = defines.get_defined_int[
    "DIV_CHANS_CONV2", 4
]()  # lower risks using too many resources, any factor of 16
comptime div_chans_conv3 = defines.get_defined_int[
    "DIV_CHANS_CONV3", 8
]()  # needs to be a factor of 120

comptime conv3_feat_total = LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL
comptime conv3_reduction_threads = next_power_of_two(conv3_feat_total)


def normalizeInputsKernel[
    batch_size: Int
](
    raw_pixels: LayoutTensor[
        DType.uint8,
        Layout.row_major(batch_size, IMAGE_SIZE, IMAGE_SIZE),
        ImmutUntrackedOrigin,
    ],
    input: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].input, MutUntrackedOrigin
    ],
):
    """Call with grid_dim=batch_size, block_dim=next_power_of_two(IMAGE_SIZE*IMAGE_SIZE) (1D).
    Pads and normalizes raw uint8 pixels into the feature input buffer.
    """
    # TODO: for all ops, cpu and accel, let's use input_layout_tensor.shape[0]() style calls instead of constants as a style guide
    comptime img_h = raw_pixels.shape[1]()  # IMAGE_SIZE (rows)
    comptime img_w = raw_pixels.shape[2]()  # IMAGE_SIZE (cols)
    comptime img_sz = img_h * img_w
    comptime reduction_size = next_power_of_two(img_sz)

    var img = block_idx.x
    var flat = thread_idx.x
    var active = flat < img_sz
    var row = (flat // img_w) if active else 0
    var col = flat % img_w
    var pix = sftype(
        rebind[UInt8](raw_pixels[img, row, col])
    ) if active else sftype(0)

    var sum_total = block.sum[block_size=reduction_size, broadcast=False](pix)
    var sq_total = block.sum[block_size=reduction_size, broadcast=False](
        pix * pix
    )

    # 2-element shared buffer: [mean, std]
    var stats = LayoutTensor[
        ftype,
        Layout.row_major(2),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat == 0:
        var mean_val = sum_total / sftype(img_sz)
        var temp = sq_total / sftype(img_sz) - mean_val * mean_val
        stats[0] = mean_val
        var temp_fp32 = Float32(temp)
        stats[1] = sftype(
            sqrt(max(temp_fp32, Float32(0))) + Float32(1e-7)
        )  # NVIDIA GPU doesn't support fp64 sqrt (yet)
    barrier()

    if active:
        # buffers are zeroed at arena / allocator init, so padding border is already 0
        input[img, 0, row + PADDING, col + PADDING] = (pix - stats[0]) / stats[
            1
        ]


def gemmFusedKernel[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    bias_layout: Layout,
    epilogue_act: Bool = False,
    bias_per_col: Bool = False,
    TILE_SIZE: Int = GPU_TILE_SIZE,
](
    # concrete origins required: enqueue_function takes the kernel as a comptime
    # param, so `_` (inferred-from-args) origins never get bound and no concrete
    # DeviceFunction type exists — untracked is the supported spelling (see
    # untrack_imm docstring / docs/origin_migration.md)
    a: LayoutTensor[ftype, a_layout, ImmutUntrackedOrigin],
    b: LayoutTensor[ftype, b_layout, ImmutUntrackedOrigin],
    c: LayoutTensor[ftype, c_layout, MutUntrackedOrigin],
    bias: LayoutTensor[ftype, bias_layout, ImmutUntrackedOrigin],
):
    """Tiled shared-memory GEMM: c(M,N) = a(M,K) @ b(K,N) + bias, one thread
    per c element within a TILE x TILE block, optional act_fn epilogue.

    Grid Dim = (ceildiv(N, TILE_SIZE), ceildiv(M, TILE_SIZE))  — (x, y)!
    Block Dim = (TILE_SIZE, TILE_SIZE)
    bias is (M,) per-row, or (N,) per-col with bias_per_col (conv3/FC want
    per-out-channel). a/c tolerate strided layouts (AoS feature-arena views);
    b should be row-major for the masked zero-fill cancellation below.
    Developed + benched in tests/gemm.mojo.
    """
    comptime M = a.shape[0]()
    comptime K = a.shape[1]()
    comptime N = b.shape[1]()
    comptime assert b.shape[0]() == K, "bad shapes (a or b)"
    comptime if bias_per_col:
        comptime assert bias_layout.size() == N, "bias must be (N,)"
    else:
        comptime assert bias_layout.size() == M, "bias must be (M,)"

    comptime BK = ceildiv(K, TILE_SIZE)

    comptime tile_layout = Layout.row_major(TILE_SIZE, TILE_SIZE)
    comptime SharedTileType = LayoutTensor[
        ftype, tile_layout, MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var ta = SharedTileType.stack_allocation()
    var tb = SharedTileType.stack_allocation()

    var tile_row = block_idx.y  # range(BM)
    var tile_col = block_idx.x  # range(BN)
    var local_row = thread_idx.y  # range(TILE_SIZE)
    var local_col = thread_idx.x  # range(TILE_SIZE)
    var global_row = tile_row * TILE_SIZE + local_row
    var global_col = tile_col * TILE_SIZE + local_col

    # the async copiers speak TileTensor; convert views once, tile per bk step
    # TODO: take TileTensor params and convert host-side instead — saves the
    # per-thread conversion work here
    var a_tt = lt_to_tt(a)
    var b_tt = lt_to_tt(b)
    var ta_tt = lt_to_tt(ta)
    var tb_tt = lt_to_tt(tb)

    # one element per thread, thread x on cols = coalesced global reads
    comptime copy_threads = tt_row_major[TILE_SIZE, TILE_SIZE]()

    var accum: sftype = 0  # bias joins in the guarded epilogue
    for bk in range(BK):
        # masked bound is rows-only and linear (rows * row_stride): tb rows past
        # K zero-fill, which also cancels ta's K-edge column garbage (x * 0);
        # ta rows past M zero-fill; tb's N-edge column garbage is only read by
        # threads the output guard discards. Tile sub-views don't runtime-clip
        # dim0, so pass the clip via src_num_valid_rows. Last-row copies may
        # overread a few elements past the buffer end (linear bound) — absorbed
        # by device alloc padding.
        copy_dram_to_sram_async[thread_layout=copy_threads, masked=True](
            ta_tt,
            a_tt.tile[TILE_SIZE, TILE_SIZE](tile_row, bk),
            min(TILE_SIZE, M - tile_row * TILE_SIZE),
        )
        copy_dram_to_sram_async[thread_layout=copy_threads, masked=True](
            tb_tt,
            b_tt.tile[TILE_SIZE, TILE_SIZE](bk, tile_col),
            min(TILE_SIZE, K - bk * TILE_SIZE),
        )
        async_copy_wait_all()
        barrier()

        comptime for k in range(TILE_SIZE):
            var a_val = rebind[sftype](ta[local_row, k])
            var b_val = rebind[sftype](tb[k, local_col])
            accum = a_val.fma(b_val, accum)
        barrier()

    if global_row < M and global_col < N:
        comptime if bias_per_col:
            accum += rebind[sftype](bias[global_col])
        else:
            accum += rebind[sftype](bias[global_row])
        comptime if epilogue_act:
            accum = act_fn.simdForward(accum)
        c[global_row, global_col] = accum


# argmax is trivially parallel (one thread per image, no block cooperation),
# so block size is just an occupancy knob: 4 warps is the conventional
# minimum-overhead shape for tiny kernels
comptime ARGMAX_TPB = 4 * WARP_SIZE


def argMaxKernel[
    batch_size: Int
](
    outputs: LayoutTensor[
        ftype, Layout.row_major(batch_size, OUTPUT), ImmutUntrackedOrigin
    ],
    guesses: LayoutTensor[
        DType.uint8, Layout.row_major(batch_size), MutUntrackedOrigin
    ],
) -> None:
    """One thread per image, sequential scan of its OUTPUT logits — 10 reads.
    Was fused into the block.sum matmul; the GEMM path computes logits as a
    batch matrix, so argmax stands alone (still only 1 byte/img D2H).
    Grid Dim = ceildiv(batch_size, ARGMAX_TPB), Block Dim = ARGMAX_TPB.
    """
    var img_idx = global_idx.x
    if img_idx < batch_size:
        var best = sftype.MIN
        var best_idx: UInt8 = 0
        comptime for oc in range(OUTPUT):
            var logit = rebind[sftype](outputs[img_idx, oc])
            if logit > best:
                best = logit
                best_idx = UInt8(oc)
        guesses[img_idx] = best_idx


@deprecated("Replaced by gemmFusedKernel + argMaxKernel; kept for A/B.")
def matMulBlockSumKernel[
    batch_size: Int
](
    weight5_6: LayoutTensor[ftype, WeightLayouts.w56, ImmutUntrackedOrigin],
    bias5_6: LayoutTensor[ftype, BiasLayouts.b56, ImmutUntrackedOrigin],
    layer5: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer5, ImmutUntrackedOrigin
    ],
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
    var feat = layer5[img_idx, thread] if thread < LAYER5 else 0
    var best = sftype.MIN
    var best_idx: UInt8 = 0

    comptime for oc in range(OUTPUT):
        var weight = weight5_6[thread, oc] if thread < LAYER5 else 0
        var prod = feat * weight
        var answer = block.sum[block_size=reduction_size, broadcast=False](prod)
        if thread == 0:
            var logit = rebind[sftype](answer + rebind[sftype](bias5_6[oc]))
            outputs[img_idx, oc] = logit
            # raw logits by design: no act_fn after the final FC layer
            # TODO: parameterize act_fn epilogue
            if logit > best:
                best = logit
                best_idx = UInt8(oc)

    if thread == 0:
        guesses[img_idx] = best_idx


def maxPool2Kernel[
    batch_size: Int
](
    layer3: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer3, ImmutUntrackedOrigin
    ],
    layer4: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer4, MutUntrackedOrigin
    ],
) -> None:
    """
    Runs as block_dim = (LF4, LF4, LAYER4) = 5 * 5 * 16 = 400, grid_dim = (batch_size).
    One thread per output. 2x2 non-overlapping pool has no data reuse, so inputs
    are read straight from global — shared staging was pure overhead.
    col on thread_idx.x: lanes column-consecutive (stride-2 reads, stride-1
    writes) instead of chan-consecutive (stride-100 scatter).
    """
    var img_idx = block_idx.x
    var row = thread_idx.y  # range(LENGTH_FEATURE4)
    var col = thread_idx.x  # range(LENGTH_FEATURE4)
    var chan = thread_idx.z  # range(LAYER4)

    var tr = row * 2
    var tc = col * 2
    var temp: sftype = rebind[sftype](
        max(
            layer3[img_idx, chan, tr, tc],
            layer3[img_idx, chan, tr + 1, tc],
        )
    )
    temp = max(temp, rebind[sftype](layer3[img_idx, chan, tr + 1, tc + 1]))
    temp = max(temp, rebind[sftype](layer3[img_idx, chan, tr, tc + 1]))

    layer4[img_idx, chan, row, col] = temp


@deprecated("Replaced by Conv3GemmKernel (gemmFusedKernel); kept for A/B.")
def conv3FusedKernel[
    batch_size: Int
](
    weight4_5: LayoutTensor[ftype, WeightLayouts.w45, ImmutUntrackedOrigin],
    bias4_5: LayoutTensor[ftype, BiasLayouts.b45, ImmutUntrackedOrigin],
    layer4: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer4, ImmutUntrackedOrigin
    ],
    layer5: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer5, MutUntrackedOrigin
    ],
) -> None:
    """Call with grid_dim = (batch_size), block_dim = LAYER5. Each thread handles one output channel.
    """
    var img_idx = block_idx.x
    var oc = thread_idx.x

    comptime num_feats = LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL

    var local_feats = LayoutTensor[
        ftype,
        FeatureLayouts.layer4,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for i in range(oc, num_feats, LAYER5):
        local_feats.ptr[i] = layer4.ptr[img_idx * num_feats + i]

    barrier()

    var acc: sftype = 0.0
    comptime for ic in range(LAYER4):
        comptime for kw in range(LENGTH_KERNEL):
            comptime for kh in range(LENGTH_KERNEL):
                acc += rebind[sftype](local_feats[ic, kw, kh]) * rebind[sftype](
                    weight4_5[ic, oc, kw, kh]
                )

    layer5[img_idx, oc] = act_fn.simdForward(acc + bias4_5[oc])


def conv2FusedKernel[
    batch_size: Int
](
    weight2_3: LayoutTensor[ftype, WeightLayouts.w23, ImmutUntrackedOrigin],
    bias2_3: LayoutTensor[ftype, BiasLayouts.b23, ImmutUntrackedOrigin],
    layer2: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer2, ImmutUntrackedOrigin
    ],
    layer3: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer3, MutUntrackedOrigin
    ],
) -> None:
    """Conv2 + activation, one thread per output pixel, output channels split into
    `div_chans_conv2` sections. Stages the kernels and input tile into shared memory.
    Grid Dim = (batch_size, channel_divisions).
    Block Dim = (LENGTH_FEATURE3, LENGTH_FEATURE3, LAYER3 // div_chans).
    col on thread_idx.x so warp lanes are column-consecutive: coalesced image
    staging + output writes, broadcast kernel-weight reads (chan on x scattered
    all three, stride-100 lanes).
    """
    comptime CHANS_TO_HANDLE = LAYER3 // div_chans_conv2
    comptime assert LAYER3 % div_chans_conv2 == 0, "conv2 chan div ! %=0"
    comptime TPB = CHANS_TO_HANDLE * LENGTH_FEATURE3 * LENGTH_FEATURE3

    var img_idx = block_idx.x
    var chans_section = block_idx.y
    var col = thread_idx.x
    var row = thread_idx.y
    var local_chan = thread_idx.z
    var offset = chans_section * CHANS_TO_HANDLE
    var global_chan = local_chan + offset
    var flat_idx = (
        thread_idx.z * block_dim.y + thread_idx.y
    ) * block_dim.x + thread_idx.x

    var local_kernels = LayoutTensor[
        ftype,
        Layout.row_major(LAYER2, CHANS_TO_HANDLE, LENGTH_KERNEL, LENGTH_KERNEL),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # TODO: could make this much more efficient
    comptime for ic in range(LAYER2):
        if row < LENGTH_KERNEL and col < LENGTH_KERNEL:
            local_kernels[ic, local_chan, row, col] = weight2_3[
                ic, global_chan, row, col
            ]

    var local_image = LayoutTensor[
        ftype,
        FeatureLayouts.layer2,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var idx = flat_idx
    while idx < local_image.size():
        # batched layer2 is row-major, so one image's slab is contiguous:
        # flat-copy from its offset instead of 3D index math per element
        local_image.ptr[idx] = layer2.ptr[img_idx * local_image.size() + idx]
        idx += TPB

    barrier()

    var result: sftype = 0
    comptime for ic in range(LAYER2):
        comptime for i in range(LENGTH_KERNEL):
            comptime for j in range(LENGTH_KERNEL):
                var in_row = row + i
                var in_col = col + j
                result += rebind[sftype](
                    local_image[ic, in_row, in_col]
                ) * rebind[sftype](local_kernels[ic, local_chan, i, j])

    layer3[img_idx, global_chan, row, col] = act_fn.simdForward(
        result + bias2_3[global_chan]
    )


def conv1PoolFusedKernel[
    batch_size: Int
](
    weight0_1: LayoutTensor[ftype, WeightLayouts.w01, ImmutUntrackedOrigin],
    bias0_1: LayoutTensor[ftype, BiasLayouts.b01, ImmutUntrackedOrigin],
    input: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].input, ImmutUntrackedOrigin
    ],
    layer2: LayoutTensor[
        ftype, BatchedFeatureLayouts[batch_size].layer2, MutUntrackedOrigin
    ],
) -> None:
    """Conv1 + activation + 2x2 maxpool fused: input → layer2, layer1 never
    touches global memory (its buffer is now dead in the GPU path).

    Grid Dim = (batch_size)
    Block Dim = (LENGTH_FEATURE2, LENGTH_FEATURE2) = 14 x 14
    One block per image, one thread per POOLED output pixel. Stages the same
    shared data as the old conv1 (150 weight floats, padded 32x32 input, biases;
    196 threads stride-load 1024 pixels). Each thread then computes the four
    conv outputs of its 2x2 pool window in registers (6 oc x 4 px x 25 MACs
    against shared) and writes the max. Pooling is applied post-activation to
    stay bit-exact with the CPU path (act_fn isn't guaranteed monotonic — GELU).
    """
    # Single-channel only: the input staging + MAC loop below assume one input
    # channel (MNIST grayscale). Fail at compile time rather than silently
    # producing wrong results if INPUT is ever bumped for a multi-channel set.
    comptime assert INPUT == 1, (
        "conv1PoolFusedKernel hardcodes INPUT==1 (single channel);"
        " multi-channel input not implemented"
    )
    comptime TPB = LENGTH_FEATURE2 * LENGTH_FEATURE2  # 196
    var img_idx = block_idx.x
    var row = thread_idx.y  # pooled output row, range(LENGTH_FEATURE2)
    var col = thread_idx.x  # pooled output col
    var flat_idx = row * block_dim.x + col

    var local_kernels = LayoutTensor[
        ftype,
        WeightLayouts.w01,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat_idx < local_kernels.size():
        local_kernels.ptr[flat_idx] = weight0_1.ptr[flat_idx]

    # INPUT > 1 not handled — guarded by the comptime assert above.
    var local_image = LayoutTensor[
        ftype,
        FeatureLayouts.input,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # TODO: copy_dram_to_sram_async() call

    var tid = flat_idx
    while tid < LENGTH_FEATURE0 * LENGTH_FEATURE0:
        var r = tid // LENGTH_FEATURE0
        var c = tid % LENGTH_FEATURE0
        local_image[0, r, c] = input[img_idx, 0, r, c]
        tid += TPB

    var local_biases = LayoutTensor[
        ftype,
        BiasLayouts.b01,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat_idx < LAYER1:
        local_biases[flat_idx] = bias0_1[flat_idx]

    barrier()

    comptime for oc in range(LAYER1):
        var best = sftype.MIN
        comptime for pr in range(2):
            comptime for pc in range(2):
                var result: sftype = 0
                comptime for ic in range(INPUT):
                    comptime for i in range(LENGTH_KERNEL):
                        comptime for j in range(LENGTH_KERNEL):
                            var in_row = row * 2 + pr + i
                            var in_col = col * 2 + pc + j
                            result += rebind[sftype](
                                local_image[ic, in_row, in_col]
                            ) * rebind[sftype](local_kernels[ic, oc, i, j])
                best = max(
                    best,
                    act_fn.simdForward(
                        result + rebind[sftype](local_biases[oc])
                    ),
                )
        layer2[img_idx, oc, row, col] = best


def printerGPU[
    layout: Layout
](storage: DeviceBuffer[ftype], label: String = "") raises -> None:
    """Debugging helper."""
    print("GPU", label, ":")
    try:
        with storage.map_to_host() as data:
            var tensor = LayoutTensor[ftype, layout, MutAnyOrigin](data)
            print(tensor)
        print()
    except e:
        print(e)


def compareBuffers[
    layout: Layout
](
    ctx: DeviceContext,
    device_buffer: DeviceBuffer[ftype],
    host_buffer: UnsafePointer[sftype, _],
    label: String = "",
):
    """Debugging helper — compares GPU buffer to CPU pointer element-wise."""

    comptime size = layout.size()
    var epsilon: sftype = -1.0
    for i in range(size):
        if abs(host_buffer[i]) > epsilon:
            epsilon = abs(host_buffer[i])
    epsilon /= 100  # allow 1% error
    comptime max_display = 1000
    var count = 0
    var pad = " " if label.byte_length() > 0 else ""
    print("Comparing GPU to CPU" + pad, label, ":")
    try:
        with device_buffer.map_to_host() as dev:
            for i in range(size):
                if (
                    dev[i] < host_buffer[i] - epsilon
                    or dev[i] > host_buffer[i] + epsilon
                ):
                    count += 1
                    if count < max_display:
                        print(
                            "\t!=,",
                            i,
                            "dev:",
                            round(dev[i], 3),
                            "host:",
                            round(host_buffer[i], 3),
                            ((dev[i] - host_buffer[i]) * 100) / host_buffer[i],
                            "% difference",
                        )
    except e:
        print(e)
    print(
        "\t...",
        count,
        "/",
        size,
        "errors between CPU and GPU. Max",
        max_display,
        "shown.",
    )


@deprecated("Fallback; argMax should be done on GPU.")
def batchedArgMax[
    batch_size: Int
](
    outputs: LayoutTensor[ftype, Layout.row_major(batch_size, OUTPUT), _],
    out guesses: InlineArray[UInt8, batch_size],
):
    # TODO: take in an "actual length" argument that defaults to batch_size but allows for short batches
    guesses = type_of(guesses)(uninitialized=True)  # out arg
    for b in range(batch_size):
        var max_idx: UInt8 = 17  # nonsense sentinel
        var max_val = sftype.MIN
        for i in range(OUTPUT):
            var v = rebind[sftype](outputs[b, i])
            if v > max_val:
                max_val = v
                max_idx = UInt8(i)
        guesses[b] = max_idx


# --- GEMM view plumbing -------------------------------------------------------
# Batched (SoA) feature buffers are dense row-major, so the GEMM 'a' operands
# are pure reshapes: layer4 (bs, 16, 5, 5) reads as (bs, 400), and layer5 is
# already (bs, 120). Views only — no data movement. conv3 and FC are then
# plain GEMMs with batch as M. (Pre-SoA these were FEAT_STRIDE-strided views
# over per-image arena slabs — the dense rows are the free coalescing win.)
comptime CONV3_K = LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL  # 400

comptime GemmLayer4Layout[batch_size: Int] = Layout.row_major(
    batch_size, CONV3_K
)
comptime OutputsLayout[batch_size: Int] = Layout.row_major(batch_size, OUTPUT)

comptime Conv3GemmKernel[batch_size: Int] = gemmFusedKernel[
    GemmLayer4Layout[batch_size],
    WeightLayouts.w45g,
    BatchedFeatureLayouts[batch_size].layer5,
    BiasLayouts.b45,
    epilogue_act=True,
    bias_per_col=True,
]
comptime FCGemmKernel[batch_size: Int] = gemmFusedKernel[
    BatchedFeatureLayouts[batch_size].layer5,
    WeightLayouts.w56,
    OutputsLayout[batch_size],
    BiasLayouts.b56,
    epilogue_act=False,  # raw logits by design: no act_fn after the final FC
    bias_per_col=True,
]


struct CompiledKernels[batch_size: Int](Movable):
    """The full forward-pass kernel set for one batch size, compiled once.

    Field types via `type_of(...)` — the checked `compile_function` return type
    embeds the kernel's arg list, so launches through these fields keep
    compile-time validation. Bare `DeviceFunction` fields don't parse ("is not
    concrete") and this nightly has no unchecked variant; see
    `ignoreme/mvp_compiled_kernels.mojo` for the experiment trail.
    """

    var norm: type_of(
        DeviceContext().compile_function[
            normalizeInputsKernel[Self.batch_size]
        ]()
    )
    var conv1: type_of(
        DeviceContext().compile_function[
            conv1PoolFusedKernel[Self.batch_size]
        ]()
    )
    var conv2: type_of(
        DeviceContext().compile_function[conv2FusedKernel[Self.batch_size]]()
    )
    var pool2: type_of(
        DeviceContext().compile_function[maxPool2Kernel[Self.batch_size]]()
    )
    var conv3: type_of(
        DeviceContext().compile_function[Conv3GemmKernel[Self.batch_size]]()
    )
    var matmul: type_of(
        DeviceContext().compile_function[FCGemmKernel[Self.batch_size]]()
    )
    var argmax: type_of(
        DeviceContext().compile_function[argMaxKernel[Self.batch_size]]()
    )

    def __init__(out self, ctx: DeviceContext) raises:
        self.norm = ctx.compile_function[
            normalizeInputsKernel[Self.batch_size]
        ]()
        self.conv1 = ctx.compile_function[
            conv1PoolFusedKernel[Self.batch_size]
        ]()
        self.conv2 = ctx.compile_function[conv2FusedKernel[Self.batch_size]]()
        self.pool2 = ctx.compile_function[maxPool2Kernel[Self.batch_size]]()
        self.conv3 = ctx.compile_function[Conv3GemmKernel[Self.batch_size]]()
        self.matmul = ctx.compile_function[FCGemmKernel[Self.batch_size]]()
        self.argmax = ctx.compile_function[argMaxKernel[Self.batch_size]]()


struct StreamSlot[batch_size: Int](Movable):
    """One stream's worth of resources for the pipelined multi-stream run: its own
    `DeviceContext` (stream), device arena, per-image feature buffers, and the
    pinned/device input, output, and guess buffers. Allocated once and reused
    across batches so streams overlap.
    """

    var ctx: DeviceContext
    var device_arena: GPUBumpArenaAllocator
    var features: FeatureGPUBuffers[Self.batch_size]
    var hosted_inputs: HostBuffer[DType.uint8]
    var device_inputs: DeviceBuffer[DType.uint8]
    var outputs_buffer: DeviceBuffer[
        ftype
    ]  # device logits (debug/inspection — not D2H'd in the hot path)
    var outputs: LayoutTensor[
        ftype, Layout.row_major(Self.batch_size, OUTPUT), MutUntrackedOrigin
    ]
    var guesses_buffer: DeviceBuffer[
        DType.uint8
    ]  # argmax per image, staged for d2h
    var hosted_guesses: HostBuffer[DType.uint8]
    var guesses: LayoutTensor[
        DType.uint8, Layout.row_major(Self.batch_size), MutUntrackedOrigin
    ]

    def __init__(out self) raises:
        """Allocate this slot's stream, arena, batched feature buffers, and I/O
        buffers. Syncs once at the end.
        """
        comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
        self.ctx = DeviceContext()
        self.device_arena = GPUBumpArenaAllocator(
            self.ctx, FeatureGPUBuffers[Self.batch_size].sizeInBytes()
        )
        self.features = FeatureGPUBuffers[Self.batch_size](self.device_arena)
        self.device_inputs = self.ctx.enqueue_create_buffer[DType.uint8](
            img_sz * Self.batch_size
        )
        self.hosted_inputs = self.ctx.enqueue_create_host_buffer[DType.uint8](
            img_sz * Self.batch_size
        )
        self.outputs_buffer = self.ctx.enqueue_create_buffer[ftype](
            Self.batch_size * OUTPUT
        )
        self.outputs = untrack(
            LayoutTensor[ftype, Layout.row_major(Self.batch_size, OUTPUT)](
                self.outputs_buffer
            )
        )
        self.guesses_buffer = self.ctx.enqueue_create_buffer[DType.uint8](
            Self.batch_size
        )
        self.hosted_guesses = self.ctx.enqueue_create_host_buffer[DType.uint8](
            Self.batch_size
        )
        self.guesses = untrack(
            LayoutTensor[DType.uint8, Layout.row_major(Self.batch_size)](
                self.guesses_buffer
            )
        )
        self.ctx.synchronize()

    def loadBatch(self, batch: Span[UInt8, _]) raises:
        """
        Takes in a Span that should represent (batch_size * 784) bytes
        (784 = 28*28 raw uint8 pixels per image).

        If the span size isn't a multiple of the image size, that's a serious problem. Raise!

        If the span size is valid but doesn't match what the StreamSlot expects (num_images < batch_size)
        we'll just pad zeros and probably nothing needs to be done otherwise (non-fatal).

        While we could limit the scope of our strategy to load, store, and transfer raw
        image pixels around and avoid some complexity and checks,
        this function is set to undertake the following:

        We take in a constructed, contiguous span of UInt8 (raw pixels) from somewhere
        and memcpy to a pinned HostBuffer. The HostBuffer allows for async uploading operation
        to the GPU, so this intermediate is effectively required to make use of multiple streams.
        For such a small, host-side copy, we shouldn't expect that cost to hold us back.
        """
        comptime img_sz = size_of[UInt8]() * IMAGE_SIZE * IMAGE_SIZE

        if len(batch) % img_sz != 0:
            raise Error(
                "Error! StreamSlot input batch (Span) invalid - not a multiple"
                " of image size!"
            )

        # copy only what the span holds — enqueue_copy_from(ptr) reads the buffer's
        # FULL length from the source pointer, an OOB read for a short batch
        var dst = self.hosted_inputs.unsafe_ptr()
        memcpy(dest=dst, src=batch.unsafe_ptr(), count=len(batch))
        comptime full_bytes = img_sz * Self.batch_size
        if len(batch) < full_bytes:  # short tail batch: pad the rest
            # zero images normalize to NaN, but padded slots are never tallied
            comptime if DISPLAY:
                print(
                    "Rest of GPU StreamSlot batch padded with zeros.",
                    file=stderr,
                )
            memset_zero(dst + len(batch), full_bytes - len(batch))

        self.device_inputs.enqueue_copy_from(self.hosted_inputs)

    def doWork(
        self,
        kernels: CompiledKernels[Self.batch_size],
        model: LeNet5GPU,
    ) raises:
        """Enqueue the full forward pipeline on this slot's stream
        (normalize → conv1 → pool1 → conv2 → pool2 → conv3 → matmul), then stage the
        argmax guesses D2H. Async — call `getResults` to sync and read them.
        """
        # flat [N, H, W] upload transport (C=1 folded out). Could be [N, 1, H, W]
        # for full [C,H,W] parity with Image/features, but it's a raw staging
        # buffer and normalizeInputsKernel already writes the channel into
        # feats[img].input[0, ...], so left as-is. (See Image [1,28,28] change.)
        comptime batch_pixels_layout = Layout.row_major(
            Self.batch_size, IMAGE_SIZE, IMAGE_SIZE
        )
        var raw_pixels_tensor = untrack_imm(
            LayoutTensor[DType.uint8, batch_pixels_layout](self.device_inputs)
        )

        # Batched feature views — pure pointer wrapping, no data movement.
        # Built mut once per layer; readers take untrack_imm(view) at the call.
        comptime feat_layouts = BatchedFeatureLayouts[Self.batch_size]
        var input_feats = untrack(
            LayoutTensor[ftype, feat_layouts.input](self.features.input)
        )
        var layer2 = untrack(
            LayoutTensor[ftype, feat_layouts.layer2](self.features.layer2)
        )
        var layer3 = untrack(
            LayoutTensor[ftype, feat_layouts.layer3](self.features.layer3)
        )
        var layer4 = untrack(
            LayoutTensor[ftype, feat_layouts.layer4](self.features.layer4)
        )
        var layer5 = untrack(
            LayoutTensor[ftype, feat_layouts.layer5](self.features.layer5)
        )
        # same layer4 bytes reshaped (bs, 16, 5, 5) -> (bs, 400) for GEMM 'a'
        var layer4_gemm = untrack_imm(
            LayoutTensor[ftype, GemmLayer4Layout[Self.batch_size]](
                self.features.layer4
            )
        )

        self.ctx.enqueue_function(
            kernels.norm,
            raw_pixels_tensor,
            input_feats,
            grid_dim=(Self.batch_size),
            block_dim=(next_power_of_two(IMAGE_SIZE * IMAGE_SIZE)),
        )
        self.ctx.enqueue_function(
            kernels.conv1,
            untrack_imm(model.weight0_1),
            untrack_imm(model.bias0_1),
            untrack_imm(input_feats),
            layer2,
            grid_dim=(Self.batch_size),
            block_dim=(LENGTH_FEATURE2, LENGTH_FEATURE2),
        )
        self.ctx.enqueue_function(
            kernels.conv2,
            untrack_imm(model.weight2_3),
            untrack_imm(model.bias2_3),
            untrack_imm(layer2),
            layer3,
            grid_dim=(Self.batch_size, div_chans_conv2),
            block_dim=(
                LENGTH_FEATURE3,
                LENGTH_FEATURE3,
                LAYER3 // div_chans_conv2,
            ),
        )
        self.ctx.enqueue_function(
            kernels.pool2,
            untrack_imm(layer3),
            layer4,
            grid_dim=(Self.batch_size),
            block_dim=(LENGTH_FEATURE4, LENGTH_FEATURE4, LAYER4),
        )
        comptime BM_TILES = ceildiv(Self.batch_size, GPU_TILE_SIZE)
        self.ctx.enqueue_function(
            kernels.conv3,
            layer4_gemm,
            untrack_imm(model.weight4_5_gemm),
            layer5,
            untrack_imm(model.bias4_5),
            grid_dim=(ceildiv(LAYER5, GPU_TILE_SIZE), BM_TILES),
            block_dim=(GPU_TILE_SIZE, GPU_TILE_SIZE),
        )
        self.ctx.enqueue_function(
            kernels.matmul,
            untrack_imm(layer5),
            untrack_imm(model.weight5_6),
            self.outputs,
            untrack_imm(model.bias5_6),
            grid_dim=(ceildiv(OUTPUT, GPU_TILE_SIZE), BM_TILES),
            block_dim=(GPU_TILE_SIZE, GPU_TILE_SIZE),
        )
        self.ctx.enqueue_function(
            kernels.argmax,
            untrack_imm(self.outputs),
            self.guesses,
            grid_dim=(ceildiv(Self.batch_size, ARGMAX_TPB)),
            block_dim=(ARGMAX_TPB),
        )
        # 1 byte/img — logits stay on device
        self.hosted_guesses.enqueue_copy_from(self.guesses_buffer)

    def getResults(self, labels: Span[UInt8, _]) raises -> Int:
        """Returns number correct for a batch. Syncs the slot's stream first.
        A short `labels` span (padded tail batch) counts only its real slots —
        the zero-padded slots produce garbage guesses and are skipped.
        """
        self.ctx.synchronize()
        # argmax already done on device — just compare guess bytes to labels.
        # `n` counts only the real slots, so a padded tail batch skips its
        # zero-padded slots. (A SIMD version was explored but isn't worth the
        # complexity for a ~1 KB/batch compare run once after the D2H sync —
        # see ignoreme/simd_eq_mwe.mojo.)
        var n = min(len(labels), Self.batch_size)
        var correct = 0
        for j in range(n):
            correct += Int(self.hosted_guesses[j] == labels[j])
        return correct


# TODO: convert stream_slots to a Span
def _batchRun[
    batch_size: Int
](
    stream_slots: UnsafePointer[StreamSlot[batch_size], _],
    data: MNISTDataView,
    model: LeNet5GPU,
    kernels: CompiledKernels[batch_size],
    num_streams: Int = NUM_GPU_STREAMS,
) raises -> Int:
    """Run batches over pre-allocated stream slots. Does not alloc or free slots.
    """
    var count = len(data)
    var total_correct = 0
    comptime batch_bytes = batch_size * IMAGE_SIZE * IMAGE_SIZE
    # ceildiv, not floor: dispatch the short tail batch (loadBatch zero-pads it,
    # getResults tallies only its real slots) so any batch size covers all images
    var total_batches = ceildiv(count, batch_size)

    for batch_num in range(total_batches):
        var slot_idx = batch_num % num_streams
        var batch_start = (
            batch_num * batch_bytes
        )  # batch_size * IMAGE_SIZE * IMAGE_SIZE
        var batch_span = data.raw_pixels[
            batch_start : batch_start + batch_bytes
        ]

        if batch_num >= num_streams:
            var stale = batch_num - num_streams
            var stale_start = stale * batch_size
            total_correct += stream_slots[slot_idx].getResults(  # D2H
                data.raw_labels[stale_start : stale_start + batch_size]
            )

        stream_slots[slot_idx].loadBatch(batch_span)  # H2D
        stream_slots[slot_idx].doWork(kernels, model)  # kernels

    var epilogue_start = max(0, total_batches - num_streams)
    for batch_num in range(epilogue_start, total_batches):
        var slot_idx = batch_num % num_streams
        var label_start = batch_num * batch_size
        total_correct += stream_slots[slot_idx].getResults(
            data.raw_labels[label_start : label_start + batch_size]
        )

    return total_correct


def batchedForwardMultiStream[
    batch_size: Int = GPU_STREAM_BATCH_SIZE
](
    ctx: DeviceContext,
    data: MNISTDataView,
    model: LeNet5GPU,
    kernels: CompiledKernels[batch_size],
    num_streams: Int = NUM_GPU_STREAMS,
) raises -> Int:
    """Effective batch size is batch_size * num_streams. Allocates and frees slots each call.
    """
    var stream_slots = alloc[StreamSlot[batch_size]](num_streams)
    for s in range(num_streams):
        (stream_slots + s).init_pointee_move(StreamSlot[batch_size]())
    try:
        var result = _batchRun(stream_slots, data, model, kernels, num_streams)
        for s in range(num_streams):
            (stream_slots + s).destroy_pointee()
        stream_slots.free()
        return result
    except e:
        for s in range(num_streams):
            (stream_slots + s).destroy_pointee()
        stream_slots.free()
        print("batchedForwardMultiStream ERROR", e)
        raise e^
