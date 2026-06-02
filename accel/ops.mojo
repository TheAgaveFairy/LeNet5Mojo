from layout import Layout, LayoutTensor
from std.math import ceil, log2, abs, sqrt, max
from std.bit import next_power_of_two  # prev_power_of_two
from std.sys import size_of, stderr

from std.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    HostBuffer,
    DeviceFunction,
)
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace

from std.benchmark.compiler import keep

from cpu.model import LeNet5, Feature
from cpu.ops import loadInput, forward, argMax
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
from image import Image
from accel.model import LeNet5GPU
from accel.feature import FeatureGPU, FeatureGPUBuffers
from accel.arena import GPUBumpArenaAllocator
from dataloader import MNISTBatch

comptime div_chans_conv2 = 8  # any lower uses too many resources
comptime div_chans_conv3 = 8  # 8  # needs to be a factor of 120


def normalizeInputsKernel[
    batch_size: Int
](
    raw_pixels: LayoutTensor[
        DType.uint8,
        Layout.row_major(batch_size, IMAGE_SIZE, IMAGE_SIZE),
        ImmutAnyOrigin,
    ],
    feats: InlineArray[FeatureGPU, batch_size],
):
    """Call with blocks = batch_size, threads_per_block = (IMAGE_SIZE, IMAGE_SIZE). Remember, IMAGE_SIZE is unpadded. We need to both pad *and* normalize into our feature buffer inputs.
    """
    var img = block_idx.x
    var row = thread_idx.y  # 0..IMAGE_SIZE
    var col = thread_idx.x  # 0..IMAGE_SIZE
    var flat = row * IMAGE_SIZE + col

    # reduction
    comptime reduction_size = next_power_of_two(
        IMAGE_SIZE * IMAGE_SIZE
    )  # 1024 from 768
    var add_buffer = LayoutTensor[
        DType.uint64,
        Layout.row_major(reduction_size),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()  # dtype needs to fit hypothetical max of (UInt8.MAX * IMAGE_SIZE * IMAGE_SIZE) which requires at least 18bits
    var std_buffer = LayoutTensor[
        DType.uint64,
        Layout.row_major(reduction_size),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()  # dtype needs to fit hypothetical max of (UInt8.MAX * IMAGE_SIZE * IMAGE_SIZE * 2) which requires at least 36bits
    var pix = UInt64(rebind[UInt8](raw_pixels[img, row, col]))
    add_buffer[flat] = pix
    std_buffer[flat] = pix * pix

    comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
    comptime remainder = reduction_size - img_sz
    if flat < remainder:
        add_buffer[flat + img_sz] = 0
        std_buffer[flat + img_sz] = 0

    barrier()

    var i = 1
    while i < reduction_size:
        if flat % (2 * i) == 0:  # and (flat + i) < reduction_size:
            add_buffer[flat] += add_buffer[flat + i]
            std_buffer[flat] += std_buffer[flat + i]
        barrier()
        i *= 2
    var mean = sftype(rebind[UInt64](add_buffer[0])) / sftype(
        IMAGE_SIZE * IMAGE_SIZE
    )
    var temp = sftype(rebind[UInt64](std_buffer[0])) / sftype(
        IMAGE_SIZE * IMAGE_SIZE
    ) - (mean * mean)
    var std = sqrt(max(Float32(temp), Float32(0))) + sftype(
        1e-7
    )  # clamp to avoid NaN from negative temp (float32 cancellation)
    barrier()

    # normalize and load
    var norm: sftype = (sftype(pix) - mean) / sftype(std)
    feats[img].input[
        0, row + PADDING, col + PADDING
    ] = norm  # buffers are zeroed at arena / allocator init


def normalizeInputs[
    batch_size: Int
](
    ctx: DeviceContext,
    raw_pixels: LayoutTensor[
        DType.uint8,
        Layout.row_major(batch_size, IMAGE_SIZE, IMAGE_SIZE),
        ImmutAnyOrigin,
    ],
    feats: InlineArray[FeatureGPU, batch_size],
    norm_kernel: DeviceFunction,
) raises -> None:
    ctx.enqueue_function(
        norm_kernel,
        raw_pixels,
        feats,
        grid_dim=(batch_size),
        block_dim=(IMAGE_SIZE, IMAGE_SIZE),
    )
    # TODO: since we're passing the ctx around now, these kernel wrapper functions are much simplified. we should either force inlining or maybe just call the kernels directly


def gatherOutputsKernel[
    batch_size: Int
](
    feats: InlineArray[FeatureGPU, batch_size],
    outputs: LayoutTensor[
        ftype, Layout.row_major(batch_size, OUTPUT), MutAnyOrigin
    ],
):
    var img = block_idx.x
    var i = thread_idx.x
    outputs[img, i] = feats[img].output[i]


def gatherOutputs[
    batch_size: Int
](
    ctx: DeviceContext,
    feats: InlineArray[FeatureGPU, batch_size],
    outputs: LayoutTensor[
        ftype, Layout.row_major(batch_size, OUTPUT), MutAnyOrigin
    ],
    gather_kernel: DeviceFunction,
) raises -> None:
    ctx.enqueue_function(
        gather_kernel,
        feats,
        outputs,
        grid_dim=(batch_size),
        block_dim=(OUTPUT),
    )


def matMulFusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Enough threads per block to do one output channel at a time as a reduction,
    so make it a power of two.
    Grid Dim = batch_size
    Block Dim = 1 << ceil(log2(in_chans)).
    """
    var img_idx = block_idx.x
    var thread = thread_idx.x
    comptime reduction_size = 1 << Int(
        ceil(log2(Float64(LAYER5)))
    )  # 128 when LAYER5 is 120

    var local_weights = LayoutTensor[
        ftype,
        LeNet5GPU.w5_6_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var local_feats = LayoutTensor[
        ftype,
        Layout.row_major(LAYER5),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for oc in range(OUTPUT):
        if thread < LAYER5:
            local_weights[thread, oc] = lenet.weight5_6[thread, oc]
    if thread < LAYER5:
        local_feats[thread] = feats[img_idx].layer5[thread, 0, 0]

    barrier()

    var reduction_buffer = LayoutTensor[
        ftype,
        Layout.row_major(reduction_size),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for oc in range(OUTPUT):
        if thread < LAYER5:
            reduction_buffer[thread] = rebind[sftype](
                local_weights[thread, oc]
            ) * rebind[sftype](local_feats[thread])
        else:
            reduction_buffer[thread] = 0.0
        barrier()

        var i = 1
        while i < reduction_size:
            if thread % (2 * i) == 0:
                reduction_buffer[thread] += reduction_buffer[thread + i]
            barrier()
            i *= 2

        if thread == 0:
            var temp = rebind[sftype](reduction_buffer[0] + lenet.bias5_6[oc])
            feats[img_idx].output[oc] = rebind[sftype](temp)
            # TODO: confirm we don't want to do act_fn.simdForward() call


def matMulForward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    matmul_kernel: DeviceFunction,
) raises -> None:
    comptime reduction_size = 1 << Int(ceil(log2(Float64(LAYER5))))  # 128
    ctx.enqueue_function(
        matmul_kernel,
        lenet,
        feats,
        grid_dim=(batch_size),
        block_dim=(reduction_size),
    )


def maxPool2Kernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Runs as block_dim = (LAYER4, LF4, LF4) = 16 * 5 * 5 = 400, grid_dim = (batch_size).
    """
    var img_idx = block_idx.x
    var row = thread_idx.z  # range(LENGTH_FEATURE4)
    var col = thread_idx.y  # range(LENGTH_FEATURE4)
    var chan = thread_idx.x  # range(LAYER4)

    var local_image = LayoutTensor[
        ftype,
        FeatureGPU.layer3_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tr = row * 2
    var tc = col * 2
    local_image[chan, tr, tc] = feats[img_idx].layer3[chan, tr, tc]
    local_image[chan, tr + 1, tc] = feats[img_idx].layer3[chan, tr + 1, tc]
    local_image[chan, tr, tc + 1] = feats[img_idx].layer3[chan, tr, tc + 1]
    local_image[chan, tr + 1, tc + 1] = feats[img_idx].layer3[
        chan, tr + 1, tc + 1
    ]
    barrier()

    var temp: sftype = rebind[sftype](
        max(local_image[chan, tr, tc], local_image[chan, tr + 1, tc])
    )
    temp = max(temp, rebind[sftype](local_image[chan, tr + 1, tc + 1]))
    temp = max(temp, rebind[sftype](local_image[chan, tr, tc + 1]))

    feats[img_idx].layer4[chan, row, col] = temp


def maxPool2Forward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    pool2_kernel: DeviceFunction,
) raises -> None:
    ctx.enqueue_function(
        pool2_kernel,
        lenet,
        feats,
        grid_dim=(batch_size),
        block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
    )


def maxPool1Kernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Runs as block_dim = (LF1, LF1), grid_dim = (batch_size, num_channels).
    """
    var img_idx = block_idx.x  # range(batch_size)
    var chan = block_idx.y  # range(LAYER1)
    var row = thread_idx.y  # range(LENGTH_FEATURE1)
    var col = thread_idx.x  # range(LENGTH_FEATURE1)

    var local_image = LayoutTensor[
        ftype,
        Layout.row_major(LENGTH_FEATURE1, LENGTH_FEATURE1),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    local_image[row, col] = feats[img_idx].layer1[chan, row, col]
    barrier()

    if row % 2 == 0 and col % 2 == 0:
        var temp: sftype = rebind[sftype](
            max(local_image[row, col], local_image[row + 1, col])
        )
        temp = max(temp, rebind[sftype](local_image[row + 1, col + 1]))
        temp = max(temp, rebind[sftype](local_image[row, col + 1]))
        feats[img_idx].layer2[chan, row // 2, col // 2] = temp


def maxPool1Forward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    pool1_kernel: DeviceFunction,
) raises -> None:
    ctx.enqueue_function(
        pool1_kernel,
        lenet,
        feats,
        grid_dim=(batch_size, LAYER1),
        block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
    )


def conv3FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size, chan_div = 8)
    Block Dim = (in_channels = 16, kernel_size = 5, ks = 5)
    Each block handles num_ocs = 120 // chan_div = 15 output channels for one image.
    """
    comptime in_chans = LAYER4
    comptime out_chans = LAYER5
    comptime div_chans = div_chans_conv3
    comptime num_ocs = out_chans // div_chans
    comptime feat_total = Float64(LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL)
    comptime reduction_size = 1 << Int(ceil(log2(feat_total)))

    var in_chan = thread_idx.x
    var col = thread_idx.y
    var row = thread_idx.z
    var flat_idx = (
        in_chan * LENGTH_KERNEL * LENGTH_KERNEL + row * LENGTH_KERNEL + col
    )

    var img_idx = block_idx.x
    var chans_set = block_idx.y
    var offset = chans_set * num_ocs

    var local_weights = LayoutTensor[
        ftype,
        Layout.row_major(in_chans, num_ocs, LENGTH_KERNEL, LENGTH_KERNEL),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var local_feats = LayoutTensor[
        ftype,
        FeatureGPU.layer4_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var reduction_buffer = LayoutTensor[
        ftype,
        Layout.row_major(reduction_size),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    comptime for oc in range(num_ocs):
        local_weights[in_chan, oc, row, col] = lenet.weight4_5[
            in_chan, oc + offset, row, col
        ]
    local_feats[in_chan, row, col] = feats[img_idx].layer4[in_chan, row, col]
    barrier()

    for oc in range(num_ocs):
        var temp = rebind[sftype](
            local_weights[in_chan, oc, row, col]
            * local_feats[in_chan, row, col]
        )
        reduction_buffer[flat_idx] = temp
        barrier()
        var i = 1
        while i < reduction_size:
            if flat_idx % (2 * i) == 0 and (flat_idx + i) < Int(feat_total):
                reduction_buffer[flat_idx] += reduction_buffer[flat_idx + i]
            barrier()
            i *= 2

        if flat_idx == 0:
            temp = rebind[sftype](
                reduction_buffer[0] + lenet.bias4_5[oc + offset]
            )
            feats[img_idx].layer5[oc + offset, 0, 0] = act_fn.simdForward(temp)


def conv3Forward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv3_kernel: DeviceFunction,
) raises -> None:
    """Each block handles num_ocs output channels for one image."""
    comptime assert (
        LAYER5 % div_chans_conv3 == 0
    ), "conv3 channel divisions must divide evenly"
    ctx.enqueue_function(
        conv3_kernel,
        lenet,
        feats,
        grid_dim=(batch_size, div_chans_conv3),
        block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
    )


def conv2FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size, channel_divisions).
    Block Dim = (LAYER3 // div_chans, LENGTH_FEATURE3, LENGTH_FEATURE3).
    """
    comptime CHANS_TO_HANDLE = LAYER3 // div_chans_conv2
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
        var tch = idx // (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        var rem = idx % (LENGTH_FEATURE2 * LENGTH_FEATURE2)
        var tr = rem // LENGTH_FEATURE2
        var tc = rem % LENGTH_FEATURE2
        local_image[tch, tr, tc] = feats[img_idx].layer2[tch, tr, tc]
        idx += TPB

    var local_biases = LayoutTensor[
        ftype,
        Layout.row_major(CHANS_TO_HANDLE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if row == 0 and col == 0:
        local_biases[local_chan] = lenet.bias2_3[global_chan]

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

    feats[img_idx].layer3[global_chan, row, col] = act_fn.simdForward(
        rebind[sftype](result + local_biases[local_chan])
    )


def conv2Forward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv2_kernel: DeviceFunction,
) raises -> None:
    comptime assert (
        LAYER3 % div_chans_conv2 == 0
    ), "conv2 channel divisions must divide evenly"
    ctx.enqueue_function(
        conv2_kernel,
        lenet,
        feats,
        grid_dim=(batch_size, div_chans_conv2),
        block_dim=(
            LAYER3 // div_chans_conv2,
            LENGTH_FEATURE3,
            LENGTH_FEATURE3,
        ),
    )


def conv1FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size)
    Block Dim = (LENGTH_FEATURE1, LENGTH_FEATURE1) = 28 x 28
    I'll explain this kernel later.
    """
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

    # TODO: INPUT > 1 not handled
    var local_image = LayoutTensor[
        ftype,
        FeatureGPU.input_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
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
        feats[img_idx].layer1[oc, row, col] = act_fn.simdForward(
            rebind[sftype](result + local_biases[oc])
        )


def conv1Forward[
    batch_size: Int
](
    ctx: DeviceContext,
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv1_kernel: DeviceFunction,
) raises -> None:
    ctx.enqueue_function(
        conv1_kernel,
        lenet,
        feats,
        grid_dim=(batch_size),
        block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
    )


def printerGPU[
    layout: Layout
](storage: DeviceBuffer[ftype], label: String = "") raises -> None:
    """Debugging helper."""
    print("GPU", label, ":")
    try:
        with DeviceContext() as ctx:
            with storage.map_to_host() as data:
                var tensor = LayoutTensor[ftype, layout, MutAnyOrigin](data)
                print(tensor)
            print()
            ctx.synchronize()
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


def singleForward(
    img: Image,
    model: LeNet5GPU,
    lenet_cpu: LeNet5,
    conv1: DeviceFunction,
    pool1: DeviceFunction,
    conv2: DeviceFunction,
    pool2: DeviceFunction,
    conv3: DeviceFunction,
    matmul: DeviceFunction,
    gather: DeviceFunction,
) raises -> Int:
    comptime batch_size = 1

    with DeviceContext() as ctx:
        var feat_cpu = Feature()
        loadInput(feat_cpu, img)
        forward(lenet_cpu, feat_cpu)
        var cpu_guess = argMax(feat_cpu.output)

        var feat_bufs = FeatureGPUBuffers(ctx)
        var feats = InlineArray[FeatureGPU, batch_size](
            fill=FeatureGPU(feat_bufs)
        )
        feat_bufs.loadInput(img)
        conv1Forward[batch_size](ctx, model, feats, conv1)
        maxPool1Forward[batch_size](ctx, model, feats, pool1)
        conv2Forward[batch_size](ctx, model, feats, conv2)
        maxPool2Forward[batch_size](ctx, model, feats, pool2)
        conv3Forward[batch_size](ctx, model, feats, conv3)
        matMulForward[batch_size](ctx, model, feats, matmul)
        ctx.synchronize()

        var host_output_layer = type_of(feat_cpu.output).stack_allocation()
        with feat_bufs.output.map_to_host() as ans:
            for i in range(host_output_layer.size()):
                host_output_layer.ptr[i] = ans[i]
        return argMax(host_output_layer)


def batchedArgMax[
    batch_size: Int
](
    outputs: LayoutTensor[ftype, Layout.row_major(batch_size, OUTPUT), _],
    out guesses: InlineArray[UInt8, batch_size],
):
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


# TODO: refactor batchedForward to take exactly one pre-sliced batch (batch_size images) rather than
# the full dataset. Move the loop to the caller. Benefits: (1) caller can own the GPU pixel staging
# buffer (allocated once, reused across calls, avoiding re-alloc per call), (2) enables ping-pong
# streaming where the caller submits the next H2D copy on a second stream before syncing the current
# compute stream, (3) makes batchedForward usable for training loops that need per-batch logic.
# Suggested caller pattern: for i in range(0, total, batch_size): batchedForward(ctx, data.slice(i, batch_size), ...)
@deprecated("Use the multi-stream version, even if only with 1 stream.")
def batchedForward[
    batch_size: Int
](
    ctx: DeviceContext,
    data: MNISTBatch,
    model: LeNet5GPU,
    norm: DeviceFunction,
    conv1: DeviceFunction,
    pool1: DeviceFunction,
    conv2: DeviceFunction,
    pool2: DeviceFunction,
    conv3: DeviceFunction,
    matmul: DeviceFunction,
    gather: DeviceFunction,
) raises -> Int:
    var count = len(data)
    print(t"batchedForward batch_size: {batch_size}, with {count} images.")
    _ = """
    comptime assert (
        count % batch_size == 0
    ), "count must be divisible by batch_size"
    """
    var correct = 0

    try:
        var arena = GPUBumpArenaAllocator(
            ctx, batch_size * FeatureGPUBuffers.sizeInBytes()
        )
        var all_bufs = alloc[FeatureGPUBuffers](batch_size)
        var features = InlineArray[FeatureGPU, batch_size](uninitialized=True)
        for j in range(batch_size):
            (all_bufs + j).init_pointee_move(FeatureGPUBuffers(arena))
            (features.unsafe_ptr() + j).init_pointee_move(
                FeatureGPU((all_bufs + j)[])
            )
        var outputs_buffer = ctx.enqueue_create_buffer[ftype](
            batch_size * OUTPUT
        )
        # ctx.synchronize()
        var outputs = LayoutTensor[
            ftype, Layout.row_major(batch_size, OUTPUT), MutAnyOrigin
        ](outputs_buffer)

        comptime batch_bytes = batch_size * IMAGE_SIZE * IMAGE_SIZE
        var gpu_raw_pixels_arena = GPUBumpArenaAllocator(
            ctx, batch_bytes
        )  # already just "bytes"

        for i in range(0, count, batch_size):
            var batch_start = i * IMAGE_SIZE * IMAGE_SIZE
            gpu_raw_pixels_arena.buffer.enqueue_copy_from(
                data.raw_pixels[batch_start : batch_start + batch_bytes]
            )

            ctx.synchronize()
            comptime batch_pixels_layout = Layout.row_major(
                batch_size, IMAGE_SIZE, IMAGE_SIZE
            )
            var raw_pixels_tensor = LayoutTensor[
                DType.uint8, batch_pixels_layout, ImmutAnyOrigin
            ](gpu_raw_pixels_arena.buffer)

            _ = """
            if i == 0:
                print("ref")
                print(data.raw_pixels[batch_start : batch_start + IMAGE_SIZE * IMAGE_SIZE])# batch_bytes])
                
                with gpu_raw_pixels_arena.buffer.map_to_host() as b:
                    var bt = LayoutTensor[DType.uint8, Layout.row_major(IMAGE_SIZE, IMAGE_SIZE), ImmutAnyOrigin](b)
                    print("device")
                    print(bt)
                    #for j in range(IMAGE_SIZE * IMAGE_SIZE):
                    #    print(b[j])
            """

            normalizeInputs(ctx, raw_pixels_tensor, features, norm)
            ctx.synchronize()  # not sure why this is needed
            conv1Forward(ctx, model, features, conv1)
            maxPool1Forward(ctx, model, features, pool1)
            conv2Forward(ctx, model, features, conv2)
            maxPool2Forward(ctx, model, features, pool2)
            conv3Forward(ctx, model, features, conv3)
            matMulForward(ctx, model, features, matmul)
            gatherOutputs(ctx, features, outputs, gather)

            # ctx.synchronize()

            with outputs_buffer.map_to_host() as outs:
                var hosted_outputs = LayoutTensor[
                    ftype, Layout.row_major(batch_size, OUTPUT), MutAnyOrigin
                ](outs.unsafe_ptr())
                var results = batchedArgMax(hosted_outputs)
                for j in range(batch_size):
                    if results[j] == UInt8(data.raw_labels[i + j]):
                        correct += 1
        all_bufs.free()
        keep(gpu_raw_pixels_arena)
        keep(outputs_buffer)
    except e:
        print("batchedForward ERROR", e)
        raise e^

    return correct


struct StreamSlot[batch_size: Int](Movable):
    var ctx: DeviceContext
    var device_arena: GPUBumpArenaAllocator
    var device_buffers: UnsafePointer[FeatureGPUBuffers, MutAnyOrigin]
    var features: InlineArray[FeatureGPU, Self.batch_size]
    var hosted_inputs: HostBuffer[DType.uint8]
    var hosted_outputs: HostBuffer[ftype]
    var device_inputs: DeviceBuffer[DType.uint8]
    var outputs_buffer: DeviceBuffer[ftype]  # staging buffer for d2h
    var outputs: LayoutTensor[
        ftype, Layout.row_major(Self.batch_size, OUTPUT), MutAnyOrigin
    ]

    def __init__(out self) raises:
        comptime img_sz = IMAGE_SIZE * IMAGE_SIZE
        self.ctx = DeviceContext()
        self.device_arena = GPUBumpArenaAllocator(
            self.ctx, Self.batch_size * FeatureGPUBuffers.sizeInBytes()
        )
        self.device_buffers = alloc[FeatureGPUBuffers](Self.batch_size)
        self.features = InlineArray[FeatureGPU, Self.batch_size](
            uninitialized=True
        )
        for j in range(Self.batch_size):
            (self.device_buffers + j).init_pointee_move(
                FeatureGPUBuffers(self.device_arena)
            )
            (self.features.unsafe_ptr() + j).init_pointee_move(
                FeatureGPU((self.device_buffers + j)[])
            )
        self.device_inputs = self.ctx.enqueue_create_buffer[DType.uint8](
            img_sz * Self.batch_size
        )
        self.hosted_inputs = self.ctx.enqueue_create_host_buffer[DType.uint8](
            img_sz * Self.batch_size
        )
        self.hosted_outputs = self.ctx.enqueue_create_host_buffer[ftype](
            Self.batch_size * OUTPUT
        )
        self.outputs_buffer = self.ctx.enqueue_create_buffer[ftype](
            Self.batch_size * OUTPUT
        )
        self.outputs = LayoutTensor[
            ftype, Layout.row_major(Self.batch_size, OUTPUT), MutAnyOrigin
        ](self.outputs_buffer)
        self.ctx.synchronize()

    def __del__(deinit self):
        for i in range(Self.batch_size):
            (self.device_buffers + i).destroy_pointee()
        self.device_buffers.free()

    def loadBatch(self, batch: Span[UInt8, _]) raises:
        """
        Takes in a Span that should represent (batch_size * 768) bytes.

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

        if (
            len(batch) / img_sz != Self.batch_size
        ):  # expected to have a perfect batch
            print(
                "Rest of GPU StreamSlot batch padded with zeros.", file=stderr
            )  # TODO: proper logging
            self.hosted_inputs.enqueue_fill(0)

        self.hosted_inputs.enqueue_copy_from(batch.unsafe_ptr())
        self.device_inputs.enqueue_copy_from(self.hosted_inputs)

    def doWork(
        self,
        norm: DeviceFunction,
        conv1: DeviceFunction,
        pool1: DeviceFunction,
        conv2: DeviceFunction,
        pool2: DeviceFunction,
        conv3: DeviceFunction,
        matmul: DeviceFunction,
        gather: DeviceFunction,
        model: LeNet5GPU,
    ) raises:
        comptime batch_pixels_layout = Layout.row_major(
            Self.batch_size, IMAGE_SIZE, IMAGE_SIZE
        )
        var raw_pixels_tensor = LayoutTensor[
            DType.uint8, batch_pixels_layout, ImmutAnyOrigin
        ](self.device_inputs)

        self.ctx.enqueue_function(
            norm,
            raw_pixels_tensor,
            self.features,
            grid_dim=(Self.batch_size),
            block_dim=(IMAGE_SIZE, IMAGE_SIZE),
        )
        self.ctx.enqueue_function(
            conv1,
            model,
            self.features,
            grid_dim=(Self.batch_size),
            block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
        )
        self.ctx.enqueue_function(
            pool1,
            model,
            self.features,
            grid_dim=(Self.batch_size, LAYER1),
            block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
        )
        self.ctx.enqueue_function(
            conv2,
            model,
            self.features,
            grid_dim=(Self.batch_size, div_chans_conv2),
            block_dim=(
                LAYER3 // div_chans_conv2,
                LENGTH_FEATURE3,
                LENGTH_FEATURE3,
            ),
        )
        self.ctx.enqueue_function(
            pool2,
            model,
            self.features,
            grid_dim=(Self.batch_size),
            block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
        )
        self.ctx.enqueue_function(
            conv3,
            model,
            self.features,
            grid_dim=(Self.batch_size, div_chans_conv3),
            block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
        )
        self.ctx.enqueue_function(
            matmul,
            model,
            self.features,
            grid_dim=(Self.batch_size),
            block_dim=(next_power_of_two(LAYER5)),
        )
        self.ctx.enqueue_function(
            gather,
            self.features,
            self.outputs,
            grid_dim=(Self.batch_size),
            block_dim=(OUTPUT),
        )
        self.hosted_outputs.enqueue_copy_from(self.outputs_buffer)

    def getResults(self, labels: Span[UInt8, _]) raises -> Int:
        """Returns number correct for a batch. Syncs the slot's stream first."""
        self.ctx.synchronize()
        var correct = 0
        var hosted_outputs = LayoutTensor[
            ftype, Layout.row_major(Self.batch_size, OUTPUT), MutAnyOrigin
        ](self.hosted_outputs.unsafe_ptr())
        var results = batchedArgMax(hosted_outputs)
        for j in range(Self.batch_size):
            if results[j] == UInt8(labels[j]):
                correct += 1
        return correct


def batchedForwardMultiStream[
    batch_size: Int = GPU_STREAM_BATCH_SIZE, num_streams: Int = NUM_GPU_STREAMS
](  # TODO: clarify what precisely "batch_size" means in the context of multistreams
    ctx: DeviceContext,
    data: MNISTBatch,
    model: LeNet5GPU,
    norm: DeviceFunction,
    conv1: DeviceFunction,
    pool1: DeviceFunction,
    conv2: DeviceFunction,
    pool2: DeviceFunction,
    conv3: DeviceFunction,
    matmul: DeviceFunction,
    gather: DeviceFunction,
) raises -> Int:
    var count = len(data)
    print(
        t"batchedForwardMultiStream: batch_size={batch_size},"
        t" num_streams={num_streams}, count={count}"
    )

    var total_correct = 0
    comptime batch_bytes = batch_size * IMAGE_SIZE * IMAGE_SIZE
    var total_batches = count // batch_size

    try:
        var stream_slots = alloc[StreamSlot[batch_size]](num_streams)
        for s in range(num_streams):
            (stream_slots + s).init_pointee_move(StreamSlot[batch_size]())

        for batch_num in range(total_batches):
            var slot_idx = batch_num % num_streams
            var batch_start = batch_num * batch_size * IMAGE_SIZE * IMAGE_SIZE
            var batch_span = data.raw_pixels[
                batch_start : batch_start + batch_bytes
            ]

            if batch_num >= num_streams:
                var stale = batch_num - num_streams
                var stale_start = stale * batch_size
                total_correct += (stream_slots + slot_idx)[].getResults(
                    data.raw_labels[stale_start : stale_start + batch_size]
                )

            (stream_slots + slot_idx)[].loadBatch(batch_span)
            (stream_slots + slot_idx)[].doWork(
                norm, conv1, pool1, conv2, pool2, conv3, matmul, gather, model
            )

        var epilogue_start = max(0, total_batches - num_streams)
        for batch_num in range(epilogue_start, total_batches):
            var slot_idx = batch_num % num_streams
            var label_start = batch_num * batch_size
            total_correct += (stream_slots + slot_idx)[].getResults(
                data.raw_labels[label_start : label_start + batch_size]
            )

        for s in range(num_streams):
            (stream_slots + s).destroy_pointee()
        stream_slots.free()

    except e:
        print("batchedForwardMultiStream ERROR", e)
        raise e^

    return total_correct
