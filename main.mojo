from std.random import seed
from std.sys.info import num_logical_cores
from std.sys import stderr
from std.time import perf_counter_ns
import std.os as os
import std.benchmark as benchmark
from std.gpu.host import DeviceContext

from image import Image
from cpu.model import LeNet5
from constants import ftype
from cpu.ops import training, testing, trainBatch
from cpu.arena import CPUBumpArenaAllocator
#
# from accel import (
#     LeNet5GPU,
#     conv1FusedKernel,
#     conv2FusedKernel,
#     conv3FusedKernel,
#     maxPool1Kernel,
#     maxPool2Kernel,
#     matMulFusedKernel,
#     batchedForward,
# )

from helpers import showProgress
from dataloader import MNISTDataRepository
from resultlogger import MultiFileLogger

# note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper
# the penultimate layer of size 84 isnt included either, see their paper

comptime COUNT_TRAIN = MNISTDataRepository.COUNT_TRAIN
comptime COUNT_TEST = MNISTDataRepository.COUNT_TEST


def main():
    print("CPU Testing")  # , num_logical_cores())
    var data_repo = MNISTDataRepository()
    var logger = MultiFileLogger("results/")

    var batch_sizes = [100]  # 100, 300, 600, 1000]
    print(len(batch_sizes), "Batch size test[s] to run")
    for b_sz in batch_sizes:  # range(tests_to_run):
        print("\tBatch size:", b_sz)
        seed(0)  # seeds 'random', we could 'search' for a better seed
        data_repo.shuffle()

        var arena = CPUBumpArenaAllocator(LeNet5._calcArenaSize())
        var model = LeNet5(arena)
        model.randomizeWeights()

        var start_time = perf_counter_ns()
        training(model, data_repo.train_data, b_sz, logger)
        #for _ in range(10):
        #    _ = trainBatch(model, data_repo.train_data[:b_sz])
        var training_time = perf_counter_ns()
        var elapsed = training_time - start_time
        print(
            "\n\tTraining done in", elapsed // 1_000_000, "ms. Now testing..."
        )
    
        var correct = testing(model, data_repo.test_data)
        var end_time = perf_counter_ns()
        elapsed = end_time - training_time
        try:
            logger.logInferenceResult(
                "CPU", elapsed, correct, COUNT_TEST, 1, ftype
            )
            print(
                "\t",
                correct,
                "/",
                COUNT_TEST,
                "correct\n\t",
                elapsed // 1_000_000,
                "ms for testing.",
            )
        except e:
            print(e, file=stderr)
        # TODO: SAVE THE MODEL TO A FILE
        benchmark.keep(arena)

    _ = """
    # TESTING A PRETRAINED VERSION FROM OLD FILE
        
    comptime model_name = "models/model_f64.dat"
    comptime saved_model_dtype = DType.float64

    print("Loading and testing a saved model: '" + model_name + "'")
    var modelCPU = LeNet5()
    modelCPU.loadFromFile[saved_model_dtype](model_name)
    start_time = perf_counter_ns()
    var correct = testing(modelCPU, data_repo.train_data)
    end_time = perf_counter_ns()
    print("\t", correct, "/", COUNT_TRAIN, "correct")
    elapsed = end_time - start_time  # // 1_000_000
    print("\t", elapsed // 1_000_000, "ms")
    try:
        logger.logInferenceResult(
            "CPU", elapsed, correct, COUNT_TRAIN, 1, saved_model_dtype
        )
    except e:
        print(e, file=stderr)
    

    # print("Kernel Length:", LENGTH_KERNEL)
    # print("Feature 0->5:", LENGTH_FEATURE0, LENGTH_FEATURE1, LENGTH_FEATURE2, LENGTH_FEATURE3, LENGTH_FEATURE4, LENGTH_FEATURE5)
    # print("Input Channels, Layer1->5, Output:", INPUT, LAYER1, LAYER2, LAYER3, LAYER4, LAYER5, OUTPUT)

    try:
        with DeviceContext() as ctx:
            var modelGPUfromCPU = LeNet5GPU(ctx, modelCPU)
            var device_name = ctx.name()
            print(
                "\nDevice found:",
                device_name,
                ". Compiling kernels and testing...",
            )
            comptime batch_size = 50  # more than ~75 fails "uses too much parameter space"

            var conv1 = ctx.compile_function[
                conv1FusedKernel[batch_size, reLu]
            ]()
            var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
            var conv2 = ctx.compile_function[
                conv2FusedKernel[batch_size, reLu]
            ]()
            var pool2 = ctx.compile_function[maxPool2Kernel[batch_size]]()
            var conv3 = ctx.compile_function[
                conv3FusedKernel[batch_size, reLu]
            ]()
            var matmul = ctx.compile_function[
                matMulFusedKernel[batch_size, reLu]
            ]()

            var start_time = perf_counter_ns()

            var correct = batchedForward[COUNT_TRAIN, batch_size](
                train_data,
                modelGPUfromCPU,
                conv1,
                pool1,
                conv2,
                pool2,
                conv3,
                matmul,
            )
            var end_time = perf_counter_ns()
            var elapsed = end_time - start_time  # // 1_000_000

            print("\t", correct, "/", COUNT_TRAIN, "correct")
            print("\t", elapsed // 1_000_000, "ms")
            logger.logInferenceResult(
                device_name, elapsed, correct, COUNT_TRAIN, batch_size, ftype
            )
    except e:
        print("ERROR IN MAIN", e, file=stderr)
        # don't forget to tell "raise" what to raise, compiler doesn't handle that well
    """
