from layout import Layout, LayoutTensor
from std.math import ceil, log2

from std.gpu.host import DeviceContext, DeviceFunction
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace

from cpu.model import LeNet5
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
    PADDED_SIZE,
)
from image import Image
from gpu.model import LeNet5GPU, FeatureGPU, FeatureGPUBuffers

comptime div_chans_conv2 = 8  # any lower uses too many resources
comptime div_chans_conv3 = 8  # needs to be a factor of 120


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

        var i = 1
        while i < reduction_size:
            if thread % (2 * i) == 0:
                reduction_buffer[thread] += reduction_buffer[thread + i]
            barrier()
            i *= 2

        if thread == 0:
            var temp = rebind[sftype](reduction_buffer[0] + lenet.bias5_6[oc])
            feats[img_idx].output[oc] = act_fn.gpu_forward(temp)


def matMulForward[
    batch_size: Int
](
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    matmul_kernel: DeviceFunction,
) raises -> None:
    comptime reduction_size = 1 << Int(ceil(log2(Float64(LAYER5))))  # 128
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(
                matmul_kernel,
                lenet,
                feats,
                grid_dim=(batch_size),
                block_dim=(reduction_size),
            )
            ctx.synchronize()
    except e:
        print(e)


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
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    pool2_kernel: DeviceFunction,
) raises -> None:
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(
                pool2_kernel,
                lenet,
                feats,
                grid_dim=(batch_size),
                block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
            )
            ctx.synchronize()
    except e:
        print(e)


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
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    pool1_kernel: DeviceFunction,
) raises -> None:
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(
                pool1_kernel,
                lenet,
                feats,
                grid_dim=(batch_size, LAYER1),
                block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
            )
            ctx.synchronize()
    except e:
        print(e)


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
            feats[img_idx].layer5[oc + offset, 0, 0] = act_fn.gpu_forward(temp)


def conv3Forward[
    batch_size: Int
](
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv3_kernel: DeviceFunction,
) raises -> None:
    """Each block handles num_ocs output channels for one image."""
    comptime assert (
        LAYER5 % div_chans_conv3 == 0
    ), "conv3 channel divisions must divide evenly"
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(
                conv3_kernel,
                lenet,
                feats,
                grid_dim=(batch_size, div_chans_conv3),
                block_dim=(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4),
            )
            ctx.synchronize()
    except e:
        print(e)


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
    if flat_idx < local_biases.size():
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

    feats[img_idx].layer3[global_chan, row, col] = act_fn.gpu_forward(
        rebind[sftype](result + local_biases[local_chan])
    )


def conv2Forward[
    batch_size: Int
](
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv2_kernel: DeviceFunction,
) raises -> None:
    comptime assert (
        LAYER3 % div_chans_conv2 == 0
    ), "conv2 channel divisions must divide evenly"
    try:
        with DeviceContext() as ctx:
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
            ctx.synchronize()
    except e:
        print(e)


def conv1FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Grid Dim = (batch_size)
    Block Dim = (LENGTH_FEATURE1, LENGTH_FEATURE1) = 28 x 28
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
        feats[img_idx].layer1[oc, row, col] = act_fn.gpu_forward(
            rebind[sftype](result + local_biases[oc])
        )


def conv1Forward[
    batch_size: Int
](
    lenet: LeNet5GPU,
    feats: InlineArray[FeatureGPU, batch_size],
    conv1_kernel: DeviceFunction,
) raises -> None:
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(
                conv1_kernel,
                lenet,
                feats,
                grid_dim=(batch_size),
                block_dim=(LENGTH_FEATURE1, LENGTH_FEATURE1),
            )
            ctx.synchronize()
    except e:
        print(e)


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
    device_buffer: DeviceBuffer[ftype],
    host_buffer: UnsafePointer[sftype, _],
    label: String = "",
):
    """Debugging helper — compares GPU buffer to CPU pointer element-wise."""
    from std.math import abs

    var epsilon: sftype = -1.0
    for i in range(layout.size()):
        if abs(host_buffer[i]) > epsilon:
            epsilon = abs(host_buffer[i])
    epsilon /= 100  # allow 1% error
    comptime max_display = 1000
    var count = 0
    print("Comparing GPU to CPU", label, ":")
    try:
        with DeviceContext() as ctx:
            with device_buffer.map_to_host() as dev:
                for i in range(layout.size()):
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
                                round(dev[i], 2),
                                "host:",
                                round(host_buffer[i], 2),
                                ((dev[i] - host_buffer[i]) * 100)
                                / host_buffer[i],
                                "% difference",
                            )
    except e:
        print(e)
    print(
        "\t...",
        count,
        "/",
        layout.size(),
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
) raises -> UInt8:
    var gpu_guess = 10  # invalid sentinel — TODO: make Optional[Int]
    var img_copy = img

    comptime batch_size = 1

    try:
        with DeviceContext() as ctx:
            var feat_cpu = lenet.Feature()
            lenet.loadInput(feat_cpu, img_copy)
            lenet.forward(lenet_cpu, feat_cpu)
            var cpu_guess = lenet.argMax(feat_cpu.output)

            var feats = InlineArray[FeatureGPU, batch_size](
                fill=FeatureGPU(ctx)
            )
            var feat_bufs = FeatureGPUBuffers(ctx, feats[0])
            feat_bufs.loadInput(img)
            conv1Forward[batch_size](model, feats, conv1)
            maxPool1Forward[batch_size](model, feats, pool1)
            conv2Forward[batch_size](model, feats, conv2)
            maxPool2Forward[batch_size](model, feats, pool2)
            conv3Forward[batch_size](model, feats, conv3)
            matMulForward[batch_size](model, feats, matmul)

            var host_output_layer = type_of(feat_cpu.output).stack_allocation()
            with feat_bufs.output_storage.map_to_host() as ans:
                for i in range(host_output_layer.size()):
                    host_output_layer.ptr[i] = ans[i]
            gpu_guess = lenet.argMax(host_output_layer)
    except e:
        print(e)

    return gpu_guess


def getResults[
    batch_size: Int
](
    ctx: DeviceContext, features: InlineArray[FeatureGPU, batch_size]
) raises -> InlineArray[UInt8, batch_size]:
    var output = InlineArray[UInt8, batch_size](
        fill=69
    )  # 69 = sentinel "bad value"
    try:
        for j in range(batch_size):
            var bufs = FeatureGPUBuffers(ctx, features[j])
            with bufs.output_storage.map_to_host() as result:
                var idx: UInt = 13  # sentinel
                var val: sftype = -1.0
                for k in range(OUTPUT):
                    if result[k] > val:
                        idx = k
                        val = result[k]
                output[j] = idx
    except e:
        print(e)
    return output^


def batchedForward[
    count: Int, batch_size: Int
](
    data: UnsafePointer[Image],
    model: LeNet5GPU,
    conv1: DeviceFunction,
    pool1: DeviceFunction,
    conv2: DeviceFunction,
    pool2: DeviceFunction,
    conv3: DeviceFunction,
    matmul: DeviceFunction,
) raises -> UInt:
    comptime assert (
        count % batch_size == 0
    ), "count must be divisible by batch_size"
    print("Batched forward, batch size:", batch_size)
    var correct = 0

    try:
        with DeviceContext() as ctx:
            var features = InlineArray[FeatureGPU, batch_size](
                fill=FeatureGPU(ctx)
            )
            # TODO: @parameter explodes compile time — left as runtime loop
            for i in range(0, count, batch_size):
                for j in range(batch_size):
                    var bufs = FeatureGPUBuffers(ctx, features[j])
                    bufs.loadInput(data[i + j])

                conv1Forward(model, features, conv1)
                maxPool1Forward(model, features, pool1)
                conv2Forward(model, features, conv2)
                maxPool2Forward(model, features, pool2)
                conv3Forward(model, features, conv3)
                matMulForward(model, features, matmul)

                var results = getResults(ctx, features)
                comptime for j in range(batch_size):
                    if results[j] == UInt(data[i + j].label):
                        correct += 1
    except e:
        print("batchedForward ERROR", e)
        raise e

    return correct
