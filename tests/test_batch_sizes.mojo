"""Weird-batch-size coverage for the GPU pipeline.

`_batchRun` drops the tail that doesn't fill a full batch (until the
pad-the-tail TODO lands). These tests pin down the CURRENT contract for
dataset slices that don't divide evenly:

  - processed = (n // batch_size) * batch_size images, remainder dropped
  - GPU correct-count == CPU correct-count over exactly those images
  - n < batch_size  ->  zero batches, zero correct, no crash/OOB

Run: pixi run mojo -I src tests/test_batch_sizes.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_equal

from constants import ftype, GPU_STREAM_BATCH_SIZE
from cpu.model import LeNet5
from cpu.ops import testingParallel
from dataloader import MNISTDataRepository
from accel import DeviceSession, GPUBumpArenaAllocator
from accel.ops import CompiledKernels, StreamSlot, _batchRun
from image import Image

comptime model_path = "models/deleteme.test"
comptime batch_size = GPU_STREAM_BATCH_SIZE
comptime num_streams = 3


def cpuCorrectPrefix(model: LeNet5, data: List[Image], n: Int) raises -> Int:
    """CPU ground truth: correct-count over the first n test images."""
    if n == 0:
        return 0
    var prefix = List[Image](capacity=n)
    for i in range(n):
        prefix.append(data[i])
    return testingParallel(model, prefix)


def main() raises:
    var data_repo = MNISTDataRepository()
    var model = LeNet5()
    model.loadFromFile[ftype](model_path)

    with DeviceContext() as ctx:
        var gpu_session = DeviceSession[GPUBumpArenaAllocator](ctx)
        gpu_session.bufs.loadCPUWeights(model)
        var kernels = CompiledKernels[batch_size](ctx)

        var slots = alloc[StreamSlot[batch_size]](num_streams)
        for s in range(num_streams):
            (slots + s).init_pointee_move(StreamSlot[batch_size]())

        # n < bs, n < bs (just under), exactly one batch, partial tail,
        # several batches + tail, almost-full, full
        var sizes = [17, 99, 100, 250, 999, 5000, 9999, 10000]
        for n in sizes:
            var view = data_repo.getTestBatch(0, n)
            var gpu_correct = _batchRun[batch_size](
                slots, view, gpu_session.model, kernels, num_streams
            )
            var processed = (n // batch_size) * batch_size
            var cpu_correct = cpuCorrectPrefix(
                model, data_repo.test_data, processed
            )
            print(
                t"n={n}: processed={processed},"
                t" gpu={gpu_correct}, cpu={cpu_correct}"
            )
            assert_equal(gpu_correct, cpu_correct)

        for s in range(num_streams):
            (slots + s).destroy_pointee()
        slots.free()

    print("\nAll batch-size tests passed!")
