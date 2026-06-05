from layout import Layout, LayoutTensor
from std.math import abs, sqrt, max
from std.bit import next_power_of_two  # prev_power_of_two
from std.sys import size_of, stderr

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
from dataloader import MNISTBatch

comptime div_chans_conv2 = 8  # any lower uses too many resources
comptime div_chans_conv3 = 8  # 8  # needs to be a factor of 120

# conv3 reduces LAYER4 * 5 * 5 = 400 products per output channel via block.sum.
# block.sum needs a 1D block whose size is a multiple of the warp size, so we
# pad up to the next power of two (512) and let threads >= 400 contribute 0.
comptime conv3_feat_total = LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL
comptime conv3_reduction_threads = next_power_of_two(conv3_feat_total)


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


def matMulFusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
    """
    Enough threads per block to do one output channel at a time as a reduction,
    so make it a power of two.
    Grid Dim = batch_size
    Block Dim = next_power_of_two(in_chans)
    """
    var img_idx = block_idx.x
    var thread = thread_idx.x
    comptime reduction_size = next_power_of_two(LAYER5)  # 120 -> 128

    # dram to local call possible
    var feat = feats[img_idx].layer5[thread, 0, 0] if thread < LAYER5 else 0

    comptime for oc in range(OUTPUT):
        var weight = lenet.weight5_6[thread, oc] if thread < LAYER5 else 0
        var prod = feat * weight
        var answer = block.sum[block_size=reduction_size, broadcast=False](prod)
        if thread == 0:
            feats[img_idx].output[oc] = answer + lenet.bias5_6[oc]
            # TODO: confirm we don't want to do act_fn.simdForward() call


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


def conv3FusedKernel[
    batch_size: Int
](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, batch_size]) -> None:
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
            block_dim=(conv3_reduction_threads),
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
