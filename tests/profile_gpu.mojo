from std.gpu.host import DeviceContext
from std.sys import argv
from std.benchmark.compiler import keep

from constants import (
    ftype,
    act_fn,
    GPU_STREAM_BATCH_SIZE,
    IMAGE_SIZE,
)
from cli import CliArgs
from cpu.model import LeNet5
from cpu.ops import testing
from dataloader import MNISTDataRepository, MNISTDataView
from accel import (
    batchedForwardMultiStream,
    DeviceSession,
    GPUBumpArenaAllocator,
)
from accel.ops import CompiledKernels, StreamSlot

comptime model_path = "models/deleteme.test"
comptime saved_model_dtype = ftype
comptime batch_size = GPU_STREAM_BATCH_SIZE


def main() raises:
    var args = argv()
    var use_test_data = len(args) > 1 and args[1] == "--test"
    var num_streams = CliArgs.parse().num_streams

    var data_repo = MNISTDataRepository()
    var model = LeNet5()
    model.loadFromFile[saved_model_dtype](model_path)

    var cpu_correct = testing(model, data_repo.test_data)
    print(
        "CPU test accuracy:", cpu_correct, "/", MNISTDataRepository.COUNT_TEST
    )

    with DeviceContext() as ctx:
        var gpu_session = DeviceSession[GPUBumpArenaAllocator](ctx)
        gpu_session.bufs.loadCPUWeights(model)
        keep(gpu_session)

        var kernels = CompiledKernels[batch_size](ctx)

        # Warmup: one throwaway all-zero forward pass through a single StreamSlot.
        # Absorbs JIT + lazy CUDA context init + first-launch cache warmup so the
        # timed run's first batch isn't a profiling outlier. Result discarded.
        var warm_slot = StreamSlot[batch_size]()
        var zero_pixels = InlineArray[
            UInt8, batch_size * IMAGE_SIZE * IMAGE_SIZE
        ](fill=0)
        var zero_labels = InlineArray[UInt8, batch_size](fill=0)
        warm_slot.loadBatch(Span(zero_pixels))
        warm_slot.doWork(kernels, gpu_session.model)
        _ = warm_slot.getResults(Span(zero_labels))

        var total = MNISTDataRepository.COUNT_TEST if use_test_data else MNISTDataRepository.COUNT_TRAIN
        var data = data_repo.getTestBatch(0, MNISTDataRepository.COUNT_TEST) if use_test_data else data_repo.getTrainBatch(0, MNISTDataRepository.COUNT_TRAIN)

        var correct = batchedForwardMultiStream[batch_size](
            ctx, data, gpu_session.model, kernels, num_streams
        )
        print(
            "GPU",
            "test" if use_test_data else "train",
            "accuracy:",
            correct,
            "/",
            total,
        )
        keep(gpu_session)
