from layout import LayoutTensor, Layout
from std.math import exp, sqrt, log
from std.algorithm.functional import vectorize
from std.utils.index import IndexList
from std.memory import memcpy
import std.benchmark as benchmark

from std.time import perf_counter_ns
from helpers import showProgress
from resultlogger import LeNet5Logger

from cpu.model import LeNet5, Feature
from constants import (
    ftype,
    sftype,
    nelts,
    act_fn,
    LENGTH_KERNEL,
    PADDED_SIZE,
    ALPHA,
)
from image import Image
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as CPUArena


def argMax[layout: Layout](output: LayoutTensor[ftype, layout, _]) -> Int:
    var largest_value: sftype = FloatLiteral[].negative_infinity
    var pos: Int = 0
    comptime for i in range(layout.size()):
        var value = rebind[sftype](output[i])
        if value > largest_value:
            largest_value = value
            pos = i
    return pos


def crossEntropyLossSIMD[
    layout: Layout
](preds: LayoutTensor[ftype, layout, _], label: Int) -> Float64:
    """
    Input is treated as if it is always a 1d 'vector'.
    SIMD vectorized.
    """
    var global_max: sftype = preds.ptr[0]

    @parameter
    def find_max[width: Int](i: Int) unified {read, mut global_max}:
        var nums = preds.ptr.load[width=width](i)
        var local_max = nums.reduce_max()
        if local_max > global_max:
            global_max = local_max

    vectorize[nelts](comptime (layout.size()), find_max)

    var exp_sum: sftype = 0.0

    @parameter
    def calc_exp[width: Int](i: Int) unified {read, mut exp_sum}:
        var ps = preds.ptr.load[width](i)
        var maxes = SIMD[ftype, width](global_max)
        var diff = ps - maxes
        exp_sum += exp(diff).reduce_add()

    vectorize[nelts](comptime (layout.size()), calc_exp)

    var log_prob: sftype = rebind[sftype](
        (preds.ptr[label] - global_max) - log(exp_sum)
    )
    return -1.0 * Float64(log_prob)


def crossEntropyLoss[
    count: Int
](
    preds: LayoutTensor[ftype, Layout.row_major(count), MutAnyOrigin],
    label: Int,
) -> Float32:
    var max_val: sftype = rebind[sftype](preds[0])

    comptime for i in range(1, count):
        if preds[i] > max_val:
            max_val = rebind[sftype](preds[i])

    var exp_sum: sftype = 0.0
    comptime for i in range(count):
        var temp = rebind[sftype](preds[i] - max_val)
        exp_sum += exp(temp)

    var log_prob: sftype = rebind[sftype](
        (preds[label] - max_val) - log(exp_sum)
    )
    return -1.0 * Float32(log_prob)


def softMax[
    count: Int
](
    input: LayoutTensor[ftype, Layout.row_major(count), _],
    loss: LayoutTensor[ftype, Layout.row_major(count), MutAnyOrigin],
    label: Int,
):
    var inner: sftype = 0.0
    for i in range(count):
        var res: sftype = 0.0
        for j in range(count):
            res += exp(rebind[sftype](input[j]) - rebind[sftype](input[i]))
        loss[i] = 1.0 / res
        inner -= rebind[sftype](loss[i] * loss[i])

    inner += rebind[sftype](loss[label])
    for i in range(count):
        var temp: sftype = sftype(1.0) if i == label else sftype(0.0)
        loss[i] *= temp - rebind[sftype](loss[i]) - inner


def loadTarget(features: Feature, errors: Feature, label: Int) -> None:
    softMax(features.output, errors.output, label)


def convoluteBackward[
    in_chan: Int,
    out_chan: Int,
    feat_size: Int,
    kernel_size: Int,
](
    input: LayoutTensor[
        ftype, Layout.row_major(in_chan, feat_size, feat_size), MutAnyOrigin
    ],
    inerror: LayoutTensor[
        ftype, Layout.row_major(in_chan, feat_size, feat_size), MutAnyOrigin
    ],
    outerror: LayoutTensor[
        ftype,
        Layout.row_major(
            out_chan, feat_size - kernel_size + 1, feat_size - kernel_size + 1
        ),
        MutAnyOrigin,
    ],
    weight: LayoutTensor[
        ftype,
        Layout.row_major(in_chan, out_chan, kernel_size, kernel_size),
        MutAnyOrigin,
    ],
    wdeltas: LayoutTensor[
        ftype,
        Layout.row_major(in_chan, out_chan, kernel_size, kernel_size),
        MutAnyOrigin,
    ],
    bdeltas: LayoutTensor[ftype, Layout.row_major(out_chan), MutAnyOrigin],
):
    comptime out_feat_size = feat_size - kernel_size + 1

    comptime for x in range(in_chan):
        comptime for y in range(out_chan):
            var inerror_slice = rebind[
                LayoutTensor[
                    ftype, Layout.row_major(feat_size, feat_size), MutAnyOrigin
                ]
            ](
                inerror.slice[
                    Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1, 2)
                ](IndexList[1](x))
            )

            var weight_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(kernel_size, kernel_size),
                    MutAnyOrigin,
                ]
            ](
                weight.slice[
                    Slice(0, kernel_size),
                    Slice(0, kernel_size),
                    IndexList[2](2, 3),
                ](IndexList[2](x, y))
            )

            var outerror_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(out_feat_size, out_feat_size),
                    MutAnyOrigin,
                ]
            ](
                outerror.slice[
                    Slice(0, out_feat_size),
                    Slice(0, out_feat_size),
                    IndexList[2](1, 2),
                ](IndexList[1](y))
            )
            convoluteFull(weight_slice, outerror_slice, inerror_slice)

    act_fn.backward(input, inerror, inerror)
    _ = """
    for c in range(in_chan): # each element gets "actiongrad"
        for m in range(feat_size):
            for n in range(feat_size):
                inerror[c, m, n] *= 1 if input[c, m, n] > 0 else 0
    """
    comptime for c in range(out_chan):
        comptime for i in range(out_feat_size):
            comptime for j in range(out_feat_size):
                bdeltas[c] += outerror[c, i, j]

    comptime for x in range(in_chan):
        comptime for y in range(out_chan):
            # input[x], wd[x][y], outerror[y]
            var input_slice = rebind[
                LayoutTensor[
                    ftype, Layout.row_major(feat_size, feat_size), MutAnyOrigin
                ]
            ](
                input.slice[
                    Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1, 2)
                ](IndexList[1](x))
            )

            var wdeltas_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(kernel_size, kernel_size),
                    MutAnyOrigin,
                ]
            ](
                wdeltas.slice[
                    Slice(0, kernel_size),
                    Slice(0, kernel_size),
                    IndexList[2](2, 3),
                ](IndexList[2](x, y))
            )

            var outerror_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(out_feat_size, out_feat_size),
                    MutAnyOrigin,
                ]
            ](
                outerror.slice[
                    Slice(0, out_feat_size),
                    Slice(0, out_feat_size),
                    IndexList[2](1, 2),
                ](IndexList[1](y))
            )

            convoluteValid(outerror_slice, input_slice, wdeltas_slice)


def convoluteValid[
    # feat_size: Int,
    # kernel_size: Int,
    k_layout: Layout,
    i_layout: Layout,
    r_layout: Layout,
](
    kernel: LayoutTensor[ftype, k_layout, _],  # (kernel_size, kernel_size)
    image: LayoutTensor[ftype, i_layout, _],  # (feat_size, feat_size)
    result: LayoutTensor[
        ftype,
        r_layout,
        MutAnyOrigin,
    ],  # (out_size, out_size) == (feat - kern + 1, feat - kern + 1)
) -> None:
    comptime assert (
        kernel.shape[0]() == kernel.shape[1]()
    ), "Kernel shape incorrect."
    comptime assert (
        image.shape[0]() == image.shape[1]()
    ), "Image shape incorrect."
    comptime assert (
        result.shape[0]() == result.shape[1]()
    ), "Result shape incorrect."
    comptime kernel_size = kernel.shape[0]()
    comptime feat_size = image.shape[1]()
    comptime assert result.shape[0]() == (
        feat_size - kernel_size + 1
    ), "Incorrect shapes."

    comptime for i in range(result.shape[0]()):  # each output pixel row
        comptime for j in range(result.shape[1]()):  # each output pixel column
            comptime for a in range(
                kernel.shape[0]()
            ):  # for each weight row of a kernel
                comptime for b in range(
                    kernel.shape[1]()
                ):  # for each weight col of a kernel
                    result[i, j] += image[i + a, j + b] * kernel[a, b]


def convoluteFull[
    k_layout: Layout, i_layout: Layout, r_layout: Layout
](
    kernel: LayoutTensor[ftype, k_layout, _],  # (kernel_size, kernel_size)
    image: LayoutTensor[
        ftype,
        i_layout,  # (feat - kern + 1, feat - kern + 1)
        _,
    ],
    result: LayoutTensor[ftype, r_layout, MutAnyOrigin],  # (feat, feat)
) -> None:
    comptime assert (
        kernel.shape[0]() == kernel.shape[1]()
    ), "Kernel shape incorrect."
    comptime assert (
        image.shape[0]() == image.shape[1]()
    ), "Image shape incorrect."
    comptime assert (
        result.shape[0]() == result.shape[1]()
    ), "Result shape incorrect."
    comptime feat_size = result.shape[0]()
    comptime kernel_size = kernel.shape[0]()
    comptime assert image.shape[0]() == (
        feat_size - kernel_size + 1
    ), "Incorrect shapes."

    comptime for i in range(image.shape[0]()):  # each input pixel row
        for j in range(image.shape[1]()):  # each input pixel column
            for a in range(
                kernel.shape[0]()
            ):  # for each weight row of a kernel
                for b in range(
                    kernel.shape[1]()
                ):  # for each weight col of a kernel
                    result[i + a, j + b] += image[i, j] * kernel[a, b]


def convoluteForward[
    in_chan: Int,
    out_chan: Int,
    feat_size: Int,
    kernel_size: Int,
](
    kernels: LayoutTensor[
        ftype, Layout.row_major(in_chan, out_chan, kernel_size, kernel_size), _
    ],
    bias: LayoutTensor[ftype, Layout.row_major(out_chan), _],
    image: LayoutTensor[
        ftype, Layout.row_major(in_chan, feat_size, feat_size), _
    ],
    result: LayoutTensor[
        ftype,
        Layout.row_major(
            out_chan, feat_size - kernel_size + 1, feat_size - kernel_size + 1
        ),
        MutAnyOrigin,
    ],
) -> None:
    comptime out_feat_size = feat_size - kernel_size + 1

    comptime for x in range(kernels.shape[0]()):  # number of input channels
        for y in range(kernels.shape[1]()):  # number of output channels
            # slicing syntax (gives a 2d for now) = [ Slice(rows wanted), Slice(cols wanted) IndexList[2](dimensions you want) ] (IndexList[2](dim0, dim1) # etc, or can just be a Scalar offset for each dim to use)
            var kern_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(kernel_size, kernel_size),
                    MutAnyOrigin,
                ]
            ](
                kernels.slice[
                    Slice(0, kernel_size),
                    Slice(0, kernel_size),
                    IndexList[2](2, 3),
                ](IndexList[2](x, y))
            )

            var image_slice = rebind[
                LayoutTensor[
                    ftype, Layout.row_major(feat_size, feat_size), MutAnyOrigin
                ]
            ](
                image.slice[
                    Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1, 2)
                ](
                    IndexList[1](x)
                )  # FIXME: submit API / syntax bug report
            )

            var result_slice = rebind[
                LayoutTensor[
                    ftype,
                    Layout.row_major(out_feat_size, out_feat_size),
                    MutAnyOrigin,
                ]
            ](
                result.slice[
                    Slice(0, out_feat_size),
                    Slice(0, out_feat_size),
                    IndexList[2](1, 2),
                ](
                    IndexList[1](y)
                )  # FIXME: submit bug report (docs expect IndexList, not Int)
            )

            convoluteValid(kern_slice, image_slice, result_slice)

    # activation function (named "action")
    comptime for c in range(result.shape[0]()):
        for i in range(result.shape[1]()):
            for j in range(result.shape[2]()):
                result[c, i, j] += bias[c]
                result[c, i, j] = (
                    result[c, i, j] if result[c, i, j] > 0.0 else 0.0
                )


# "out feat size" is from the perspective of the forward pass... I might want to clear up names
def maxPoolBackward[
    num_channels: Int, in_feat_size: Int, out_feat_size: Int
](
    input: LayoutTensor[
        ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), _
    ],
    inerror: LayoutTensor[
        ftype,
        Layout.row_major(num_channels, in_feat_size, in_feat_size),
        MutAnyOrigin,
    ],
    outerror: LayoutTensor[
        ftype, Layout.row_major(num_channels, out_feat_size, out_feat_size), _
    ],
):
    comptime len0 = inerror.shape[1]() // outerror.shape[1]()
    comptime len1 = inerror.shape[2]() // outerror.shape[2]()

    comptime for i in range(num_channels):
        for o0 in range(out_feat_size):
            for o1 in range(out_feat_size):
                var x0 = Int(0)
                var x1 = Int(0)
                var ismax: Int

                # branchless approach again
                for l0 in range(len0):
                    for l1 in range(len1):
                        ismax = (
                            1 if input[i, o0 * len0 + l0, o1 * len1 + l1]
                            > input[i, o0 * len0 + x0, o1 * len1 + x1] else 0
                        )
                        x0 += ismax * (l0 - x0)
                        x1 += ismax * (l1 - x1)

                inerror[i, o0 * len0 + x0, o1 * len1 + x1] = outerror[i, o0, o1]


def maxPoolForward[
    num_channels: Int, in_feat_size: Int, out_feat_size: Int
](
    input: LayoutTensor[
        ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), _
    ],
    output: LayoutTensor[
        ftype,
        Layout.row_major(num_channels, out_feat_size, out_feat_size),
        MutAnyOrigin,
    ],
):
    var lenx = input.shape[1]() // output.shape[1]()
    var leny = input.shape[2]() // output.shape[2]()

    comptime for c in range(output.shape[0]()):  # each channel
        for i in range(output.shape[1]()):  # feature size
            for j in range(
                output.shape[2]()
            ):  # feature size (should match shape[1]())
                var x0: Int = 0
                var y0: Int = 0

                for x in range(lenx):
                    for y in range(leny):
                        var temp_idx_x = Int(i * lenx + x)
                        var temp_idx_y = Int(j * leny + y)
                        var temp_idx_xx = Int(i * lenx + x0)
                        var temp_idx_yy = Int(j * leny + y0)

                        var ismax = (
                            1 if input[c, temp_idx_x, temp_idx_y]
                            > input[c, temp_idx_xx, temp_idx_yy] else 0
                        )
                        x0 += Int(ismax * (x - x0))
                        y0 += Int(ismax * (y - y0))

                var temp_idx_xx = Int(i * lenx + x0)
                var temp_idx_yy = Int(j * leny + y0)

                output[c, i, j] = input[c, temp_idx_xx, temp_idx_yy]


def matmulBackward[
    num_chan: Int,
    feat_size: Int,
    output_size: Int,
](
    input: LayoutTensor[
        ftype, Layout.row_major(num_chan, feat_size, feat_size), _
    ],
    inerror: LayoutTensor[
        ftype, Layout.row_major(num_chan, feat_size, feat_size), MutAnyOrigin
    ],
    outerror: LayoutTensor[ftype, Layout.row_major(output_size), _],
    weight: LayoutTensor[
        ftype,
        Layout.row_major(num_chan * feat_size * feat_size, output_size),
        _,
    ],
    wdeltas: LayoutTensor[
        ftype,
        Layout.row_major(num_chan * feat_size * feat_size, output_size),
        MutAnyOrigin,
    ],
    bdeltas: LayoutTensor[ftype, Layout.row_major(output_size), MutAnyOrigin],
) -> None:
    comptime total_feats = feat_size * feat_size

    comptime for x in range(weight.shape[0]()):
        for y in range(output_size):
            var ie_i = x // (total_feats)
            var rem = x % total_feats
            var ie_j = rem // feat_size
            var ie_k = rem % feat_size
            inerror[ie_i, ie_j, ie_k] += outerror[y] * weight[x, y]

    act_fn.backward(input, inerror, inerror)
    _ = """
    for i in range(num_chan):
        for j in range(feat_size):
            for k in range(feat_size):
                inerror[i, j, k] *= 1 if input[i, j, k] > 0 else 0
    """

    comptime for i in range(output_size):
        bdeltas[i] += outerror[i]

    comptime for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            var ie_i = x // (total_feats)  # num_chan
            var rem = x % total_feats
            var ie_j = rem // feat_size  # feat_size
            var ie_k = rem % feat_size  # feat_size
            wdeltas[x, y] += input[ie_i, ie_j, ie_k] * outerror[y]


def matmulForward[
    num_chan: Int,
    feat_size: Int,
    output_size: Int,
](
    input: LayoutTensor[
        ftype, Layout.row_major(num_chan, feat_size, feat_size), _
    ],
    output: LayoutTensor[ftype, Layout.row_major(output_size), MutAnyOrigin],
    weight: LayoutTensor[
        ftype,
        Layout.row_major(num_chan * feat_size * feat_size, output_size),
        _,
    ],
    bias: LayoutTensor[ftype, Layout.row_major(output_size), _],
) -> None:
    # input is m x l, weight is l x n, output is m x n
    # input is (layer5, feat5, feat5), weight is (layer5 * feat5 * feat5, output), output is (output)
    # feature_length5 is equal to the value 1
    comptime for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            for f in range(feat_size):
                output[y] += input[x, f, f] * weight[x, y]

    comptime for i in range(output.shape[0]()):
        output[i] += bias[i]
        output[i] = output[i] if output[i] > 0 else 0


# BECOME STRUCT METHODS


def loadInput(features: Feature, image: Image):
    memcpy(
        src=image.pixels.ptr,
        dest=features.input.ptr,
        count=PADDED_SIZE * PADDED_SIZE,
    )
    # for i in range(PADDED_SIZE):
    #    for j in range(PADDED_SIZE):
    #        features.input[0, i, j] = image.pixels[i, j]


def forward(lenet: LeNet5, features: Feature):
    convoluteForward(
        lenet.weight0_1, lenet.bias0_1, features.input, features.layer1
    )
    # input, l1, lf0, lk

    maxPoolForward(features.layer1, features.layer2)
    # l1 lf1 lf2

    convoluteForward(
        lenet.weight2_3, lenet.bias2_3, features.layer2, features.layer3
    )
    # l2 l3 lf2 lk

    maxPoolForward(features.layer3, features.layer4)
    # l3 lf3 lf4

    convoluteForward(
        lenet.weight4_5, lenet.bias4_5, features.layer4, features.layer5
    )
    # l4 l5 lf4 lk

    matmulForward(
        features.layer5, features.output, lenet.weight5_6, lenet.bias5_6
    )
    # LAYER5, LEA_f5, output


def backward(
    lenet: LeNet5, deltas: LeNet5, errors: Feature, features: Feature
) -> None:
    matmulBackward(
        features.layer5,
        errors.layer5,
        errors.output,
        lenet.weight5_6,
        deltas.weight5_6,
        deltas.bias5_6,
    )
    # l5, lf5, output

    convoluteBackward[
        kernel_size=LENGTH_KERNEL
    ](  # not sure why this needs specifying now #FIXME:
        features.layer4,
        errors.layer4,
        errors.layer5,
        lenet.weight4_5,
        deltas.weight4_5,
        deltas.bias4_5,
    )
    # l4 l5 lf4 lk

    maxPoolBackward(features.layer3, errors.layer3, errors.layer4)
    # l3 lf3 lf4

    convoluteBackward[
        kernel_size=LENGTH_KERNEL
    ](  # not sure why this needs specifying now #FIXME:
        features.layer2,
        errors.layer2,
        errors.layer3,
        lenet.weight2_3,
        deltas.weight2_3,
        deltas.bias2_3,
    )
    # l2 l3 lf2 lk

    maxPoolBackward(features.layer1, errors.layer1, errors.layer2)
    # l1 lf1 lf2

    convoluteBackward[
        kernel_size=LENGTH_KERNEL
    ](  # not sure why this needs specifying now #FIXME:
        features.input,
        errors.input,
        errors.layer1,
        lenet.weight0_1,
        deltas.weight0_1,
        deltas.bias0_1,
    )
    # input l1 lf0 lk


def predict(lenet: LeNet5, image: Image) -> Int:
    # TODO: Probably could be a method of LeNet5.
    var feat_arena = CPUArena(Feature._calcArenaSize())
    var feat = Feature(feat_arena)
    loadInput(feat, image)
    forward(lenet, feat)
    return argMax(feat.output)

def predictNew(lenet: LeNet5, feat: Feature, image: Image) -> Int:
    loadInput(feat, image)
    forward(lenet, feat)
    return argMax(feat.output)

def trainBatch(
    mut model: LeNet5, inputs: Span[mut=False, Image, _]
) -> Tuple[Int, Float32]:
    var batch_size = len(inputs)
    var correct = 0
    var total_loss: Float32 = 0.0

    var buffer_arena = CPUArena(LeNet5._calcArenaSize())
    var buffer = LeNet5(buffer_arena)

    var feat_arena = CPUArena(Feature._calcArenaSize())
    var error_arena = CPUArena(Feature._calcArenaSize())
    var delta_arena = CPUArena(LeNet5._calcArenaSize())

    var feat = Feature(feat_arena)
    var errors = Feature(error_arena)
    var deltas = LeNet5(delta_arena)

    for i in range(batch_size):
        #var feat = Feature()
        #var errors = Feature()
        #var deltas = LeNet5()
        loadInput(feat, inputs[i])
        forward(model, feat)
        var pred = argMax(feat.output)
        var the_label = Int(inputs[i].label)
        if pred == the_label:
            correct += 1

        var loss = crossEntropyLoss(feat.output, the_label)
        total_loss += loss
        loadTarget(feat, errors, the_label)
        backward(model, deltas, errors, feat)
        buffer.accumulateFromOther(deltas, 1.0)

        feat_arena.clear()
        error_arena.clear()
        delta_arena.clear()

    var k: sftype = sftype(ALPHA) / sftype(batch_size)
    model.accumulateFromOther(buffer, k)

    var avg_loss = total_loss / Float32(batch_size)

    benchmark.keep(buffer_arena)
    benchmark.keep(delta_arena)
    benchmark.keep(feat_arena)
    benchmark.keep(error_arena)

    return Tuple[Int, Float32](correct, avg_loss)


def training(
    mut model: LeNet5,
    data: List[Image],
    batch_size: Int,
    mut logger: Some[LeNet5Logger],
):
    # print("Training")
    var total_size = len(data)
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        var start_time = perf_counter_ns()
        var results_tuple = trainBatch(model, data[i : i + batch_size])
        var correct = results_tuple[0]
        var avg_loss = results_tuple[1]
        var end_time = perf_counter_ns()
        var elapsed = end_time - start_time
        try:
            logger.logTrainingEpoch(
                "CPU", i, elapsed, correct, total_size, avg_loss, ALPHA, ftype
            )
        except e:
            print("logging error during CPU training:", e)
        # LOSS, LR


def training(
    mut model: LeNet5,
    data: List[Image],
    batch_size: Int,
    # total_size: Int,
):
    # print("Training")
    var total_size = len(data)
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        _ = trainBatch(model, data[i : i + batch_size])


def testing(model: LeNet5, data: List[Image]) -> Int:
    var correct = 0
    var feat_arena = CPUArena(Feature._calcArenaSize())
    var feat = Feature(feat_arena)
    for i in range(len(data)):
        #var pred = predict(model, data[i])
        var pred = predictNew(model, feat, data[i])
        feat_arena.clear()
        var actual = Int(data[i].label)
        correct += 1 if pred == actual else 0
    benchmark.keep(feat_arena)
    return correct
