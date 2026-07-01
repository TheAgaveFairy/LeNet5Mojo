from layout import LayoutTensor, Layout
from std.math import exp, sqrt, log
from std.algorithm.functional import vectorize
from std.algorithm import parallelize
from std.utils.index import IndexList
from std.memory import memcpy
import std.benchmark as benchmark
from std.sys import stderr
from std.sys.info import size_of

from std.time import perf_counter_ns
from resultlogger import MultiFileLogger

from cpu.model import LeNet5, Feature
from constants import (
    ftype,
    sftype,
    nelts,
    act_fn,
    LENGTH_KERNEL,
    PADDED_SIZE,
    ALPHA,
    DISPLAY,
    CPU_ALLOCATOR,
)
from image import Image
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as CPUArena


def showProgress(progress: Int, total: Int) -> None:
    comptime bar_width = 50
    var ratio = Float32(progress) / Float32(total)
    var filled = Int(Float32(bar_width) * ratio)
    print("\r[", end="")
    for _ in range(filled):
        print("=", end="")
    for _ in range(filled, bar_width):
        print(" ", end="")
    print("]", round(ratio * 100, 3), "%", end="")


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
](preds: LayoutTensor[ftype, layout, _], label: Int) -> Float32:
    """
    Input is treated as if it is always a 1d 'vector'.
    SIMD vectorized.
    """
    var global_max: sftype = preds.ptr[0]

    def find_max[width: Int](i: Int) {read, mut global_max}:
        var nums = preds.ptr.load[width=width](i)
        var local_max = nums.reduce_max()
        if local_max > global_max:
            global_max = local_max

    vectorize[nelts](comptime (layout.size()), find_max)

    var exp_sum: sftype = 0.0

    def calc_exp[width: Int](i: Int) {read, mut exp_sum}:
        var ps = preds.ptr.load[width](i)
        var maxes = SIMD[ftype, width](global_max)
        var diff = ps - maxes
        exp_sum += exp(diff).reduce_add()

    vectorize[nelts](comptime (layout.size()), calc_exp)

    var log_prob: sftype = rebind[sftype](
        (preds.ptr[label] - global_max) - log(exp_sum)
    )
    return -1.0 * Float32(log_prob)


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


# NOTE: the 3 call sites must pass `kernel_size=LENGTH_KERNEL` explicitly — it will NOT infer.
# Mojo binds parameters left-to-right with NO deferred unification. `input` binds in_chan/feat_size;
# the very next arg `outerror` has layout (out_chan, feat_size-kernel_size+1, ...), which the compiler
# checks immediately while out_chan/kernel_size are still unbound. `feat_size-kernel_size+1` is
# non-invertible arithmetic so kernel_size can't be solved there, and it does NOT skip ahead to
# `weight` (where kernel_size appears directly as dims 2,3). Reordering weight/wdeltas BEFORE outerror
# makes it infer (verified: ignoreme/probe_conv_infer.mojo, fB vs fA) — but that argument order reads
# worse, so we keep the explicit param. Believed to be an upstream inference limitation, not user
# error ("types parameters include unfolded expression at parser time"). See TODO.md.
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

    # TODO: rebind helper for slicing or see if we can remove rebinds entirely
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

    comptime for c in range(result.shape[0]()):
        for i in range(result.shape[1]()):
            for j in range(result.shape[2]()):
                result[c, i, j] += bias[c]
    # TODO: fuse this into the above loop using simdForward()
    act_fn.forward(result)


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
    # Clean pooling: floor-div len drops trailing rows if not divisible. Write index is provably
    # in-bounds ((out-1)*len + len-1 < out*len <= in), so these guard ignored rows / garbage calls.
    # Precondition: caller must pre-zero `inerror` — backward only scatters into the argmax cells.
    comptime assert out_feat_size > 0, "maxPoolBackward: out_feat_size must be > 0"
    comptime assert (
        in_feat_size % out_feat_size == 0
    ), "maxPoolBackward: in_feat_size must be divisible by out_feat_size"
    comptime len0 = inerror.shape[1]() // outerror.shape[1]()
    comptime len1 = inerror.shape[2]() // outerror.shape[2]()

    comptime for i in range(num_channels):
        for o0 in range(out_feat_size):
            for o1 in range(out_feat_size):
                var x0 = Int(0)
                var x1 = Int(0)
                var ismax: Int

                # branchless approach again
                # TODO: see if this is actually faster
                for l0 in range(len0):
                    for l1 in range(len1):
                        ismax = (
                            1 if input[i, o0 * len0 + l0, o1 * len1 + l1]
                            > input[i, o0 * len0 + x0, o1 * len1 + x1] else 0
                        )
                        x0 += ismax * (l0 - x0)
                        x1 += ismax * (l1 - x1)

                inerror[i, o0 * len0 + x0, o1 * len1 + x1] = outerror[i, o0, o1]

# TODO: also compare this "branchless" to a "normal" maxpool
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
    comptime lenx = input.shape[1]() // output.shape[1]()
    comptime leny = input.shape[2]() // output.shape[2]()

    comptime for c in range(output.shape[0]()):  # each channel
        comptime for i in range(output.shape[1]()):  # feature size
            comptime for j in range(
                output.shape[2]()
            ):  # feature size (should match shape[1]())
                var x0: Int = 0
                var y0: Int = 0

                comptime for x in range(lenx):
                    comptime for y in range(leny):
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

    comptime for i in range(output_size):
        bdeltas[i] += outerror[i]

    comptime for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            var ie_i = x // (total_feats)  # num_chan
            var rem = x % total_feats
            var ie_j = rem // feat_size  # feat_size
            var ie_k = rem % feat_size  # feat_size
            wdeltas[x, y] += input[ie_i, ie_j, ie_k] * outerror[y]

# TODO: this is not production grade, i have one somewhere to copy over...
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
    # input is (layer5, feat5, feat5), weight is (layer5 * feat5 * feat5, output), output is (output)
    # feature_length5 is equal to the value 1
    comptime for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            for f in range(feat_size):
                output[y] += input[x, f, f] * weight[x, y]

    comptime for i in range(output.shape[0]()):
        output[i] += bias[i]
        # output[i] = output[i] if output[i] > 0 else 0
    # act_fn.forward(output)
    # TODO: look into if this is good or bad
    # TODO: parameterize to enable/ disable, fuse into loop above?
    # FIXME: just a louder reminder


def trainBatchParallel(
    mut model: LeNet5, inputs: Span[mut=False, Image, _]
) -> Tuple[Int, Float32]:
    var batch_size = len(inputs)

    var buffer_arena = CPUArena(LeNet5.sizeInBytes())
    var buffer = LeNet5(buffer_arena)

    var arena_size = batch_size * (
        Feature.sizeInBytes() * 2 + LeNet5.sizeInBytes()
    )
    var intermediate_arena = CPUArena(arena_size)  # will abort if too big

    var features = alloc[Feature](batch_size)
    var errors = alloc[Feature](batch_size)
    var deltas = alloc[LeNet5](batch_size)

    var losses = alloc[Float32](batch_size)  # reduce add -> total_loss
    var corrects = alloc[Int](batch_size)  # reduce add, effectively bools

    for i in range(batch_size):
        # doing features[i] = Feature() will try and __del__ what "was already there" - bad
        (features + i).init_pointee_move(Feature(intermediate_arena))
        (errors + i).init_pointee_move(Feature(intermediate_arena))
        (deltas + i).init_pointee_move(LeNet5(intermediate_arena))
        # losses[i] = 0
        corrects[i] = 0

    def work(tid: Int) {read, mut intermediate_arena, mut corrects, mut losses}:
        features[tid].loadInput(inputs[tid])
        model.forward(features[tid])
        var pred = argMax(features[tid].output)
        var the_label = Int(inputs[tid].label)
        if pred == the_label:
            corrects[tid] = 1

        var loss = crossEntropyLossSIMD(features[tid].output, the_label)
        losses[tid] = loss
        loadTarget(features[tid], errors[tid], the_label)
        model.backward(deltas[tid], errors[tid], features[tid])

    parallelize(work, batch_size)

    var correct = 0
    var total_loss = Float32(0.0)
    # TODO: single threaded / atomic / critical
    for i in range(batch_size):
        buffer.accumulateFromOther(deltas[i], 1.0)
        correct += corrects[i]
        total_loss += losses[i]

    var k: sftype = sftype(ALPHA) / sftype(batch_size)
    model.accumulateFromOther(buffer, k)

    var avg_loss = total_loss / Float32(batch_size)

    # TODO: can we get rid of these keeps()
    benchmark.keep(buffer_arena)
    benchmark.keep(intermediate_arena)
    features.free()
    errors.free()
    deltas.free()
    losses.free()
    corrects.free()

    return Tuple[Int, Float32](correct, avg_loss)


def trainBatch(
    mut model: LeNet5, inputs: Span[mut=False, Image, _]
) -> Tuple[Int, Float32]:
    var batch_size = len(inputs)
    var correct = 0
    var total_loss: Float32 = 0.0

    var buffer_arena = CPUArena(LeNet5.sizeInBytes())
    var buffer = LeNet5(buffer_arena)

    var feat_arena = CPUArena(Feature.sizeInBytes())
    var error_arena = CPUArena(Feature.sizeInBytes())
    var delta_arena = CPUArena(LeNet5.sizeInBytes())

    var feat = Feature(feat_arena)
    var errors = Feature(error_arena)
    var deltas = LeNet5(delta_arena)

    for i in range(batch_size):
        # var feat = Feature()
        # var errors = Feature()
        # var deltas = LeNet5()
        feat.loadInput(inputs[i])
        model.forward(feat)
        var pred = argMax(feat.output)
        var the_label = Int(inputs[i].label)
        if pred == the_label:
            correct += 1

        var loss = crossEntropyLossSIMD(feat.output, the_label)
        total_loss += loss
        loadTarget(feat, errors, the_label)
        model.backward(deltas, errors, feat)
        buffer.accumulateFromOther(deltas, 1.0)

        feat_arena.wipe()
        error_arena.wipe()
        delta_arena.wipe()

    var k: sftype = sftype(ALPHA) / sftype(batch_size)
    model.accumulateFromOther(buffer, k)

    var avg_loss = total_loss / Float32(batch_size)

    # TODO: can we get rid of these keep() calls
    benchmark.keep(buffer_arena)
    benchmark.keep(delta_arena)
    benchmark.keep(feat_arena)
    benchmark.keep(error_arena)

    return Tuple[Int, Float32](correct, avg_loss)


def trainingParallel(
    mut model: LeNet5,
    data: List[Image],
    batch_size: Int,
    logger: Optional[MultiFileLogger] = None,
):
    if DISPLAY:
        print("Training: Multi-Threaded")
    var total_size = len(data)
    if total_size % batch_size != 0:
        print(
            "Warning: batch size doesn't evenly divide total size.", file=stderr
        )

    for i in range(0, total_size, batch_size):
        if DISPLAY:
            showProgress(i, total_size)
        var start_time = perf_counter_ns()
        var results_tuple = trainBatchParallel(model, data[i : i + batch_size])
        var elapsed = perf_counter_ns() - start_time
        if logger:
            try:
                logger.value().logTrainingEpoch(
                    "CPU", i, elapsed, results_tuple[0], total_size, results_tuple[1], ALPHA, ftype
                )
            except e:
                print("logging error during CPU training:", e)


def training(
    mut model: LeNet5,
    data: List[Image],
    batch_size: Int,
    logger: Optional[MultiFileLogger] = None,
):
    if DISPLAY:
        print("Training: Single-Threaded")
    var total_size = len(data)
    if total_size % batch_size != 0:
        print(
            "Warning: batch size doesn't evenly divide total size.", file=stderr
        )
    for i in range(0, total_size, batch_size):
        if DISPLAY:
            showProgress(i, total_size)
        var start_time = perf_counter_ns()
        var results_tuple = trainBatch(model, data[i : i + batch_size])
        var elapsed = perf_counter_ns() - start_time
        if logger:
            try:
                logger.value().logTrainingEpoch(
                    "CPU", i, elapsed, results_tuple[0], total_size, results_tuple[1], ALPHA, ftype
                )
            except e:
                print("logging error during CPU training:", e)


def testing(model: LeNet5, data: List[Image]) -> Int:
    var correct = 0
    var feat_arena = CPU_ALLOCATOR(Feature.sizeInBytes())
    var feat = Feature(feat_arena)
    for i in range(len(data)):
        feat_arena.zero()
        var pred = model.predict(feat, data[i])
        var actual = Int(data[i].label)
        correct += 1 if pred == actual else 0
    benchmark.keep(feat_arena)
    return correct


def testingParallel(
    model: LeNet5, data: List[Image], batch_size: Int = 50
) -> Int:
    var correct = 0
    var feat_arena = CPU_ALLOCATOR(Feature.sizeInBytes() * batch_size)
    var feats = alloc[Feature](batch_size)

    for i in range(batch_size):
        # doing feats[i] = Feature() will try and __del__ what "was already there" - bad
        (feats + i).init_pointee_move(Feature(feat_arena))
    var corrects = List[Int](length=batch_size, fill=0)

    var n_full = len(data) // batch_size
    for i in range(0, n_full * batch_size, batch_size):
        feat_arena.zero()

        def work(tid: Int) {read, mut corrects}:
            var pred = model.predict(feats[tid], data[i + tid])
            var actual = Int(data[i + tid].label)
            corrects[tid] += 1 if pred == actual else 0

        parallelize(work, batch_size)
    benchmark.keep(feat_arena)
    for i in range(batch_size):
        correct += corrects[i]

    var remainder = len(data) % batch_size
    if remainder > 0:
        feat_arena.zero()
        var base = n_full * batch_size
        for j in range(remainder):
            var pred = model.predict(feats[j], data[base + j])
            if pred == Int(data[base + j].label):
                correct += 1

    return correct
