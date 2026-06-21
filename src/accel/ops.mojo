from layout import Layout, LayoutTensor
from std.math import abs, sqrt, max
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
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives import block
from std.gpu.memory import AddressSpace

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
)
from accel.model import LeNet5GPU
from accel.feature import FeatureGPU, FeatureGPUBuffers
from accel.arena import GPUBumpArenaAllocator
from origin_util import untrack, untrack_imm
from dataloader import MNISTDataView

# IME these don't change performance a ton
comptime div_chans_conv2 = defines.get_defined_int["DIV_CHANS_CONV2", 4]()  # lower risks using too many resources, any factor of 16
comptime div_chans_conv3 = defines.get_defined_int["DIV_CHANS_CONV3", 8]()  # needs to be a factor of 120

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
    feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin],
):
    """Call with grid_dim=batch_size, block_dim=next_power_of_two(IMAGE_SIZE*IMAGE_SIZE) (1D).
    Pads and normalizes raw uint8 pixels into the feature input buffer.
    """
    comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
    comptime reduction_size = next_power_of_two(img_sz)

    var img = block_idx.x
    var flat = thread_idx.x
    var active = flat < img_sz
    # inactive threads clamp to (0,0) — valid index, value masked to 0 below
    var row = (flat // IMAGE_SIZE) if active else 0
    var col = flat % IMAGE_SIZE
    var pix = sftype(rebind[UInt8](raw_pixels[img, row, col])) if active else sftype(0)

    var sum_total = block.sum[block_size=reduction_size, broadcast=False](pix)
    var sq_total = block.sum[block_size=reduction_size, broadcast=False](pix * pix)

    # 2-element shared buffer: [mean, std]. Only thread 0 writes, all active threads read.
    var stats = LayoutTensor[
        ftype, Layout.row_major(2), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat == 0:
        var mean_val = sum_total / sftype(img_sz)
        var temp = sq_total / sftype(img_sz) - mean_val * mean_val
        stats[0] = mean_val
        var temp_fp32 = Float32(temp)
        stats[1] = sftype(sqrt(max(temp_fp32, Float32(0))) + Float32(1e-7)) # NVIDIA GPU doesn't support fp64 sqrt (yet)
    barrier()

    if active:
        # buffers are zeroed at arena / allocator init, so padding border is already 0
        feats[img].input[0, row + PADDING, col + PADDING] = (pix - stats[0]) / stats[1]


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


def maxPool2Kernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None:
    """
    Runs as block_dim = (LAYER4, LF4, LF4) = 16 * 5 * 5 = 400, grid_dim = (batch_size).
    One thread per output. 2x2 non-overlapping pool has no data reuse, so inputs
    are read straight from global — shared staging was pure overhead.
    """
    var img_idx = block_idx.x
    var row = thread_idx.z  # range(LENGTH_FEATURE4)
    var col = thread_idx.y  # range(LENGTH_FEATURE4)
    var chan = thread_idx.x  # range(LAYER4)

    var tr = row * 2
    var tc = col * 2
    var temp: sftype = rebind[sftype](
        max(
            feats[img_idx].layer3[chan, tr, tc],
            feats[img_idx].layer3[chan, tr + 1, tc],
        )
    )
    temp = max(temp, rebind[sftype](feats[img_idx].layer3[chan, tr + 1, tc + 1]))
    temp = max(temp, rebind[sftype](feats[img_idx].layer3[chan, tr, tc + 1]))

    feats[img_idx].layer4[chan, row, col] = temp


def maxPool1Kernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None:
    """
    Runs as block_dim = (LF2, LF2), grid_dim = (batch_size, num_channels).
    One thread per output (was one per *input* with 75% idling after a shared
    staging load — no reuse in 2x2 non-overlapping pooling, so global reads win).
    """
    var img_idx = block_idx.x  # range(batch_size)
    var chan = block_idx.y  # range(LAYER2)
    var row = thread_idx.y  # range(LENGTH_FEATURE2)
    var col = thread_idx.x  # range(LENGTH_FEATURE2)

    var tr = row * 2
    var tc = col * 2
    var temp: sftype = rebind[sftype](
        max(
            feats[img_idx].layer1[chan, tr, tc],
            feats[img_idx].layer1[chan, tr + 1, tc],
        )
    )
    temp = max(temp, rebind[sftype](feats[img_idx].layer1[chan, tr + 1, tc + 1]))
    temp = max(temp, rebind[sftype](feats[img_idx].layer1[chan, tr, tc + 1]))
    feats[img_idx].layer2[chan, row, col] = temp


def conv3FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None: # TODO: convert these kernels to take Spans
    """Call with grid_dim = (batch_size), block_dim = LAYER5. Each thread handles one output channel."""
    var img_idx = block_idx.x
    var oc = thread_idx.x

    comptime num_feats = LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL

    var local_feats = LayoutTensor[
        ftype,
        FeatureGPU.layer4_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for i in range(oc, num_feats, LAYER5):
        local_feats.ptr[i] = feats[img_idx].layer4.ptr[i]

    barrier()

    var acc: sftype = 0.0
    comptime for ic in range(LAYER4):
        comptime for kw in range(LENGTH_KERNEL):
            comptime for kh in range(LENGTH_KERNEL):
                acc += rebind[sftype](local_feats[ic, kw, kh]) * rebind[sftype](lenet.weight4_5[ic, oc, kw, kh])

    feats[img_idx].layer5[oc, 0, 0] = act_fn.simdForward(
        acc + lenet.bias4_5[oc]
    )


def conv3FusedKernelOld[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None:
    """
    Grid Dim = (batch_size, chan_div = 8)
    Block Dim = (conv3_reduction_threads = 512), 1D.
    Each block handles num_ocs = 120 // chan_div = 15 output channels for one image.
    Each thread owns one of the LAYER4*5*5 = 400 (in_chan, row, col) products;
    threads 400..511 are padding and contribute 0 to the block.sum reduction.
    """
    comptime out_chans = LAYER5
    comptime div_chans = div_chans_conv3
    comptime num_ocs = out_chans // div_chans
    comptime ksq = LENGTH_KERNEL * LENGTH_KERNEL

    var flat_idx = Int(thread_idx.x)  # 0..conv3_reduction_threads-1
    var img_idx = block_idx.x
    var chans_set = block_idx.y
    var offset = chans_set * num_ocs

    var active = flat_idx < conv3_feat_total
    var in_chan = 0
    var row = 0
    var col = 0
    var feat_val: sftype = 0
    if active:
        in_chan = flat_idx // ksq
        var rem = flat_idx % ksq
        row = rem // LENGTH_KERNEL
        col = rem % LENGTH_KERNEL
        # could also just use flat_idx and access things with layer4.ptr[flat_idx] etc - ordering doesn't matter for this reduction
        feat_val = rebind[sftype](feats[img_idx].layer4[in_chan, row, col])

    # if this isn't a 'comptime for', accuracy goes down
    comptime for oc in range(num_ocs):
        var prod: sftype = 0
        if active:
            prod = feat_val * rebind[sftype](
                lenet.weight4_5[in_chan, oc + offset, row, col]
            )
        var total = block.sum[
            block_size=conv3_reduction_threads, broadcast=False
        ](prod)
        if flat_idx == 0:
            var biased = total + rebind[sftype](lenet.bias4_5[oc + offset])
            feats[img_idx].layer5[oc + offset, 0, 0] = act_fn.simdForward(
                biased
            )


def conv2FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None:
    """
    Grid Dim = (batch_size, channel_divisions).
    Block Dim = (LAYER3 // div_chans, LENGTH_FEATURE3, LENGTH_FEATURE3).
    """
    comptime CHANS_TO_HANDLE = LAYER3 // div_chans_conv2
    comptime assert LAYER3 % div_chans_conv2 == 0, "conv2 chan div ! %=0"
    comptime TPB = CHANS_TO_HANDLE * LENGTH_FEATURE3 * LENGTH_FEATURE3

    var img_idx = block_idx.x
    var chans_section = block_idx.y
    var local_chan = thread_idx.x
    var col = thread_idx.y
    var row = thread_idx.z
    var offset = chans_section * CHANS_TO_HANDLE
    var global_chan = local_chan + offset
    var flat_idx = (
        thread_idx.x * block_dim.y * block_dim.z
        + thread_idx.y * block_dim.z
        + thread_idx.z
    )

    var local_kernels = LayoutTensor[
        ftype,
        Layout.row_major(LAYER2, CHANS_TO_HANDLE, LENGTH_KERNEL, LENGTH_KERNEL),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # TODO: could make this much more efficient
    comptime for ic in range(LAYER2):
        if row < LENGTH_KERNEL and col < LENGTH_KERNEL:
            local_kernels[ic, local_chan, row, col] = lenet.weight2_3[
                ic, global_chan, row, col
            ]

    var local_image = LayoutTensor[
        ftype,
        FeatureGPU.layer2_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var idx = flat_idx
    while idx < local_image.size():
        #var tch = idx // (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        #var rem = idx % (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        #var tr = rem // LENGTH_FEATURE2
        #var tc = rem % LENGTH_FEATURE2
        #local_image[tch, tr, tc] = feats[img_idx].layer2[tch, tr, tc]
        local_image.ptr[idx] = feats[img_idx].layer2.ptr[idx]
        idx += TPB

    _ = """
    var local_biases = LayoutTensor[
        ftype,
        Layout.row_major(CHANS_TO_HANDLE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if row == 0 and col == 0:
        local_biases[local_chan] = lenet.bias2_3[global_chan]

    barrier()
    """
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

    feats[img_idx].layer3[global_chan, row, col] = act_fn.simdForward(result + lenet.bias2_3[global_chan])


def conv1FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: UnsafePointer[FeatureGPU, MutUntrackedOrigin]) -> None:
    """
    Grid Dim = (batch_size)
    Block Dim = (LENGTH_FEATURE1, LENGTH_FEATURE1) = 28 x 28
    One block per image, one thread per output pixel. Cooperatively stages
    everything the block reuses into shared memory: all INPUT*LAYER1 5x5
    weight kernels (150 floats), the padded 32x32 input (strided loads, 784
    threads cover 1024 pixels), and the LAYER1 biases. After one barrier,
    each thread computes all LAYER1=6 output channels for its (row, col) —
    6 x 25 fully unrolled MACs against shared — and writes activated results
    to layer1. Shared staging pays off here (unlike the pools) because every
    input pixel is reused by up to 25 neighboring threads x 6 channels.
    """
    # Single-channel only: the input staging + MAC loop below assume one input
    # channel (MNIST grayscale). Fail at compile time rather than silently
    # producing wrong results if INPUT is ever bumped for a multi-channel set.
    comptime assert INPUT == 1, "conv1FusedKernel hardcodes INPUT==1 (single channel); multi-channel input not implemented"
    var img_idx = block_idx.x
    var row = thread_idx.y
    var col = thread_idx.x
    var flat_idx = row * block_dim.x + col

    var local_kernels = LayoutTensor[
        ftype,
        LeNet5GPU.w0_1_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if flat_idx < local_kernels.size():
        local_kernels.ptr[flat_idx] = lenet.weight0_1.ptr[flat_idx]

    # INPUT > 1 not handled — guarded by the comptime assert above.
    var local_image = LayoutTensor[
        ftype,
        FeatureGPU.input_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # TODO: copy_dram_to_sram_async() call

    var tid = flat_idx
    while tid < LENGTH_FEATURE0 * LENGTH_FEATURE0:
        var r = tid // LENGTH_FEATURE0
        var c = tid % LENGTH_FEATURE0
        local_image[0, r, c] = feats[img_idx].input[0, r, c]
        tid += LENGTH_FEATURE1 * LENGTH_FEATURE1

    var local_biases = LayoutTensor[
        ftype,
        LeNet5GPU.b0_1_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if row == 0 and col < LAYER1:
        local_biases[col] = lenet.bias0_1[col]

    barrier()

    comptime for oc in range(LAYER1):
        var result: sftype = 0
        comptime for ic in range(INPUT):
            comptime for i in range(LENGTH_KERNEL):
                comptime for j in range(LENGTH_KERNEL):
                    var in_row = row + i
                    var in_col = col + j
                    result += rebind[sftype](
                        local_image[ic, in_row, in_col]
                    ) * rebind[sftype](local_kernels[ic, oc, i, j])
        feats[img_idx].layer1[oc, row, col] = act_fn.simdForward(result + local_biases[oc])


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
        DeviceContext().compile_function[conv1FusedKernel[Self.batch_size]]()
    )
    var pool1: type_of(
        DeviceContext().compile_function[maxPool1Kernel[Self.batch_size]]()
    )
    var conv2: type_of(
        DeviceContext().compile_function[conv2FusedKernel[Self.batch_size]]()
    )
    var pool2: type_of(
        DeviceContext().compile_function[maxPool2Kernel[Self.batch_size]]()
    )
    var conv3: type_of(
        DeviceContext().compile_function[conv3FusedKernel[Self.batch_size]]()
    )
    var matmul: type_of(
        DeviceContext().compile_function[matMulFusedKernel[Self.batch_size]]()
    )

    def __init__(out self, ctx: DeviceContext) raises:
        self.norm = ctx.compile_function[
            normalizeInputsKernel[Self.batch_size]
        ]()
        self.conv1 = ctx.compile_function[conv1FusedKernel[Self.batch_size]]()
        self.pool1 = ctx.compile_function[maxPool1Kernel[Self.batch_size]]()
        self.conv2 = ctx.compile_function[conv2FusedKernel[Self.batch_size]]()
        self.pool2 = ctx.compile_function[maxPool2Kernel[Self.batch_size]]()
        self.conv3 = ctx.compile_function[conv3FusedKernel[Self.batch_size]]()
        self.matmul = ctx.compile_function[
            matMulFusedKernel[Self.batch_size]
        ]()


struct StreamSlot[batch_size: Int](Movable):
    var ctx: DeviceContext
    var device_arena: GPUBumpArenaAllocator
    var device_buffers: UnsafePointer[FeatureGPUBuffers, MutUntrackedOrigin]

    # older version that used InlineArrays to store FeatureGPUs lead to the compiler synthesizeing moves that unrolled N times which *exploded* compile times
    var device_features: DeviceBuffer[DType.uint8]
    var features_ptr: UnsafePointer[FeatureGPU, MutUntrackedOrigin]
    var hosted_inputs: HostBuffer[DType.uint8]
    var device_inputs: DeviceBuffer[DType.uint8]
    var outputs_buffer: DeviceBuffer[ftype]  # device logits (debug/inspection — not D2H'd in the hot path)
    var outputs: LayoutTensor[
        ftype, Layout.row_major(Self.batch_size, OUTPUT), MutUntrackedOrigin
    ]
    var guesses_buffer: DeviceBuffer[DType.uint8]  # argmax per image, staged for d2h
    var hosted_guesses: HostBuffer[DType.uint8]
    var guesses: LayoutTensor[
        DType.uint8, Layout.row_major(Self.batch_size), MutUntrackedOrigin
    ]

    def __init__(out self) raises:
        comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
        self.ctx = DeviceContext()
        self.device_arena = GPUBumpArenaAllocator(
            self.ctx, Self.batch_size * FeatureGPUBuffers.sizeInBytes()
        )
        self.device_buffers = rebind[
            UnsafePointer[FeatureGPUBuffers, MutUntrackedOrigin]
        ](alloc[FeatureGPUBuffers](Self.batch_size))
        # Local, NOT a field: only needed to seed device_features below. Keeping it
        # off the struct avoids unrolling N FeatureGPU moves on every StreamSlot move.
        # TODO: could probably bypass this intermediate
        var features = InlineArray[FeatureGPU, Self.batch_size](
            uninitialized=True
        )
        for j in range(Self.batch_size):
            (self.device_buffers + j).init_pointee_move(
                FeatureGPUBuffers(self.device_arena)
            )
            (features.unsafe_ptr() + j).init_pointee_move(
                FeatureGPU((self.device_buffers + j)[])
            )
        comptime feat_bytes = Self.batch_size * size_of[FeatureGPU]()
        self.device_features = self.ctx.enqueue_create_buffer[DType.uint8](
            feat_bytes
        )
        self.device_features.enqueue_copy_from(
            features.unsafe_ptr().bitcast[UInt8]()
        )
        self.features_ptr = rebind[
            UnsafePointer[FeatureGPU, MutUntrackedOrigin]
        ](self.device_features.unsafe_ptr().bitcast[FeatureGPU]())
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
            LayoutTensor[
                ftype, Layout.row_major(Self.batch_size, OUTPUT)
            ](self.outputs_buffer)
        )
        self.guesses_buffer = self.ctx.enqueue_create_buffer[DType.uint8](
            Self.batch_size
        )
        self.hosted_guesses = self.ctx.enqueue_create_host_buffer[DType.uint8](
            Self.batch_size
        )
        self.guesses = untrack(
            LayoutTensor[
                DType.uint8, Layout.row_major(Self.batch_size)
            ](self.guesses_buffer)
        )
        self.ctx.synchronize()

    def __del__(deinit self):
        for i in range(Self.batch_size):
            (self.device_buffers + i).destroy_pointee()
        self.device_buffers.free()

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
        if len(batch) < full_bytes:  # expected to have a perfect batch
            print(
                "Rest of GPU StreamSlot batch padded with zeros.", file=stderr
            )  # TODO: proper logging
            memset_zero(dst + len(batch), full_bytes - len(batch))

        self.device_inputs.enqueue_copy_from(self.hosted_inputs)

    def doWork(
        self,
        kernels: CompiledKernels[Self.batch_size],
        model: LeNet5GPU,
    ) raises:
        comptime batch_pixels_layout = Layout.row_major(
            Self.batch_size, IMAGE_SIZE, IMAGE_SIZE
        )
        var raw_pixels_tensor = untrack_imm(
            LayoutTensor[DType.uint8, batch_pixels_layout](self.device_inputs)
        )

        self.ctx.enqueue_function(
            kernels.norm,
            raw_pixels_tensor,
            self.features_ptr,
            grid_dim=(Self.batch_size),
            block_dim=(next_power_of_two(IMAGE_SIZE * IMAGE_SIZE)),
        )
        self.ctx.enqueue_function(
            kernels.conv1,
            model,
            self.features_ptr,
            grid_dim=(Self.batch_size),
            block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
        )
        self.ctx.enqueue_function(
            kernels.pool1,
            model,
            self.features_ptr,
            grid_dim=(Self.batch_size, LAYER1),
            block_dim=(LENGTH_FEATURE2, LENGTH_FEATURE2),
        )
        self.ctx.enqueue_function(
            kernels.conv2,
            model,
            self.features_ptr,
            grid_dim=(Self.batch_size, div_chans_conv2),
            block_dim=(
                LAYER3 // div_chans_conv2,
                LENGTH_FEATURE3,
                LENGTH_FEATURE3,
            ),
        )
        self.ctx.enqueue_function(
            kernels.pool2,
            model,
            self.features_ptr,
            grid_dim=(Self.batch_size),
            block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
        )
        self.ctx.enqueue_function(
            kernels.conv3,
            model,
            self.features_ptr,
            grid_dim=(Self.batch_size),
            block_dim=(LAYER5),
        )
        self.ctx.enqueue_function(
            kernels.matmul,
            model,
            self.features_ptr,
            self.outputs,
            self.guesses,
            grid_dim=(Self.batch_size),
            block_dim=(next_power_of_two(LAYER5)),
        )
        # 1 byte/img — logits stay on device
        self.hosted_guesses.enqueue_copy_from(self.guesses_buffer)

    def getResults(self, labels: Span[UInt8, _]) raises -> Int:
        """Returns number correct for a batch. Syncs the slot's stream first. Does not check for a full batch - handle at call."""
        if len(labels) < Self.batch_size:
            raise Error(
                t"getResults: labels span ({len(labels)}) shorter than"
                t" batch_size ({Self.batch_size})"
            )
        self.ctx.synchronize()
        var correct = 0
        # argmax already done on device — just compare guess bytes to labels
        for j in range(Self.batch_size):
            if self.hosted_guesses[j] == labels[j]:
                correct += 1
        return correct

# TODO: convert stream_slots to a Span, pass CompiledKernels
def _batchRun[
    batch_size: Int
](
    stream_slots: UnsafePointer[StreamSlot[batch_size], _],
    data: MNISTDataView,
    model: LeNet5GPU,
    kernels: CompiledKernels[batch_size],
    num_streams: Int = NUM_GPU_STREAMS,
) raises -> Int:
    """Run batches over pre-allocated stream slots. Does not alloc or free slots."""
    var count = len(data)
    var total_correct = 0
    comptime batch_bytes = batch_size * IMAGE_SIZE * IMAGE_SIZE
    var total_batches = count // batch_size

    for batch_num in range(total_batches):
        var slot_idx = batch_num % num_streams
        var batch_start = batch_num * batch_bytes #batch_size * IMAGE_SIZE * IMAGE_SIZE
        var batch_span = data.raw_pixels[
            batch_start : batch_start + batch_bytes
        ]

        if batch_num >= num_streams:
            var stale = batch_num - num_streams
            var stale_start = stale * batch_size
            total_correct += stream_slots[slot_idx].getResults( # D2H
                data.raw_labels[stale_start : stale_start + batch_size]
            )

        stream_slots[slot_idx].loadBatch(batch_span) # H2D
        stream_slots[slot_idx].doWork( # kernels
            kernels, model
        )

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
    """Effective batch size is batch_size * num_streams. Allocates and frees slots each call."""
    var stream_slots = alloc[StreamSlot[batch_size]](num_streams)
    for s in range(num_streams):
        (stream_slots + s).init_pointee_move(StreamSlot[batch_size]())
    try:
        var result = _batchRun[batch_size](
            stream_slots, data, model, kernels, num_streams
        )
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
