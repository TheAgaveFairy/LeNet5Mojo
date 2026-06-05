from std.gpu.host import DeviceContext
from std.sys import argv
from std.benchmark.compiler import keep

from constants import (
    ftype,
    act_fn,
    GPU_STREAM_BATCH_SIZE,
    NUM_GPU_STREAMS,
    IMAGE_SIZE,
)
from cpu.model import LeNet5
from cpu.ops import testing
from dataloader import MNISTDataRepository, MNISTBatch
from accel import (
    batchedForwardMultiStream,
    DeviceSession,
    GPUBumpArenaAllocator,
)
from accel.ops import (
    normalizeInputsKernel,
    conv1FusedKernel,
    maxPool1Kernel,
    conv2FusedKernel,
    maxPool2Kernel,
    conv3FusedKernel,
    matMulFusedKernel,
    gatherOutputsKernel,
    StreamSlot,
)

comptime model_path = "models/deleteme.test"
comptime saved_model_dtype = ftype
comptime batch_size = GPU_STREAM_BATCH_SIZE
comptime num_streams = NUM_GPU_STREAMS


def main() raises:
    var args = argv()
    var use_test_data = len(args) > 1 and args[1] == "--test"

    var data_repo = MNISTDataRepository()
    var model = LeNet5()
    model.loadFromFile[saved_model_dtype](model_path)

    var cpu_correct = testing(model, data_repo.test_data)
    print("CPU test accuracy:", cpu_correct, "/", MNISTDataRepository.COUNT_TEST)

    with DeviceContext() as ctx:
        var gpu_session = DeviceSession[GPUBumpArenaAllocator](ctx)
        gpu_session.bufs.loadCPUWeights(model)
        keep(gpu_session)

        var norm = ctx.compile_function[normalizeInputsKernel[batch_size]]()
        var conv1 = ctx.compile_function[conv1FusedKernel[batch_size]]()
        var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
        var conv2 = ctx.compile_function[conv2FusedKernel[batch_size]]()
        var pool2 = ctx.compile_function[maxPool2Kernel[batch_size]]()
        var conv3 = ctx.compile_function[conv3FusedKernel[batch_size]]()
        var matmul = ctx.compile_function[matMulFusedKernel[batch_size]]()
        var gather = ctx.compile_function[gatherOutputsKernel[batch_size]]()

        # Warmup: one throwaway all-zero forward pass through a single StreamSlot.
        # Absorbs JIT + lazy CUDA context init + first-launch cache warmup so the
        # timed run's first batch isn't a profiling outlier. Result discarded.
        var warm_slot = StreamSlot[batch_size]()
        var zero_pixels = InlineArray[
            UInt8, batch_size * IMAGE_SIZE * IMAGE_SIZE
        ](fill=0)
        var zero_labels = InlineArray[UInt8, batch_size](fill=0)
        warm_slot.loadBatch(Span(zero_pixels))
        warm_slot.doWork(
            norm, conv1, pool1, conv2, pool2, conv3, matmul, gather,
            gpu_session.model,
        )
        _ = warm_slot.getResults(Span(zero_labels))

        var data: MNISTBatch
        var total: Int
        if use_test_data:
            data = data_repo.getTestBatch(0, MNISTDataRepository.COUNT_TEST)
            total = MNISTDataRepository.COUNT_TEST
        else:
            data = data_repo.getTrainBatch(0, MNISTDataRepository.COUNT_TRAIN)
            total = MNISTDataRepository.COUNT_TRAIN

        var correct = batchedForwardMultiStream[batch_size, num_streams](
            ctx,
            data,
            gpu_session.model,
            norm, conv1, pool1, conv2, pool2, conv3, matmul, gather,
        )
        print("GPU", "test" if use_test_data else "train", "accuracy:", correct, "/", total)
        keep(gpu_session)
    # FIXME: MNISTBatch holds Spans into data_repo's arena. Mojo doesn't track this origin
    # dependency, so the compiler may destroy data_repo before inference completes.
    # keep() forces data_repo to survive. Fix: MNISTBatch should carry the origin of its
    # backing arena so the borrow checker can enforce the lifetime relationship.
    keep(data_repo)
