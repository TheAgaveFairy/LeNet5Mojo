from std.subprocess import run as subProcessRun
from std.random import seed
from std.sys.info import num_logical_cores
from std.sys import stderr
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
import std.os as os
import std.benchmark as benchmark
from std.reflection import get_type_name, reflect # TODO: remove unused imports
from std.gpu.host import DeviceContext

from image import Image
from cpu.model import LeNet5
from constants import ftype, act_fn, ALPHA, DISPLAY
from cpu.ops import training, trainingParallel, testing, trainBatch
from cpu.arena import CPUBumpArenaAllocator, CPUSystemAllocator

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

comptime act_fn_name = reflect[act_fn]().name()

def trainAndTest(
    mut model: LeNet5,
    data_repo: MNISTDataRepository,
    alloc: String,
    #thread: String,
    run_id: String,
    *,
    parallel: Bool = False,
    batch_size: Int = 300,
):
    var act_name = materialize[act_fn_name]().split('.')[1] # ex: activation_fn.GELU, only need the end portion
    var threads = 1 if not parallel else num_logical_cores()
    #var name = t"{alloc} with {threads} threads"
    #print(t"{name} begins, parallel = {parallel}, batch_size = {batch_size}")
    var infer_name = t"mode=infer_alloc={alloc}_thread={threads}_bs={batch_size}_act={act_name}_run={run_id}"
    var train_name = t"mode=train_alloc={alloc}_thread={threads}_bs={batch_size}_act={act_name}_run={run_id}"
    var logger = MultiFileLogger(
        "results/", String(infer_name), String(train_name)
    )
    var start_time = perf_counter_ns()
    if parallel:
        trainingParallel(model, data_repo.train_data, batch_size, logger)
    else:
        training(model, data_repo.train_data, batch_size, logger)
    var mid_time = perf_counter_ns()
    var training_ns = mid_time - start_time
    if DISPLAY:
        print(
                t"\n\t{alloc} training:",
            training_ns // 1_000_000,
            "ms.",
        )

    var correct = testing(model, data_repo.test_data)
    var end_time = perf_counter_ns()
    var testing_ns = end_time - mid_time
    try:
        logger.logInferenceResult(
            "CPU", testing_ns, correct, COUNT_TEST, 1, ftype
        )
        if DISPLAY:
            print(
                "\t",
                correct,
                "/",
                COUNT_TEST,
                "correct\n\t",
                testing_ns // 1_000_000,
                "ms for testing.",
            )
    except e:
        print(e, file=stderr)

    var training_ms = training_ns // 1_000_000
    var testing_ms = testing_ns // 1_000_000
    print(t"alloc={alloc}, act_fn={act_name}, threads={threads}, ALPHA={ALPHA}, correct={correct}, total_count={len(data_repo.test_data)}, ftype={ftype}, batch_size={batch_size}, training_ms={training_ms}, testing_ms={testing_ms}")

def main():
    var run_id: String
    try:
        run_id = subProcessRun("date +%s")
    except:
        run_id = "unknown"

    var act_name = materialize[act_fn_name]().split('.')[1]
    #print(t"CPU Testing with {num_logical_cores()} cores. Activation function is: {act_name}. Alpha = {ALPHA}, ftype = {ftype}")
    var data_repo = MNISTDataRepository()

    var batch_sizes = [300]  # 100, 300, 600, 1000] # prefer 300
    #print(len(batch_sizes), "Batch size test[s] to run")
    for b_sz in batch_sizes:  # range(tests_to_run):
        #print("\tBatch size:", b_sz)
        seed(42069)  # seeds 'random', we could 'search' for a better seed
        data_repo.shuffle()

        var arena = CPUBumpArenaAllocator(LeNet5._calcArenaSize())
        #var arena = CPUSystemAllocator()
        var arena_model = LeNet5(arena)
        #arena_model.randomizeWeights()

        #var model = LeNet5()
        #model.randomizeWeights()

        arena_model.zero()
        arena_model.randomizeWeights()
        #model.zero()
        #model.randomizeWeights()

        trainAndTest(arena_model, data_repo, "arena", run_id, parallel = True, batch_size = b_sz)
        #trainAndTest(model, data_repo, "alloc", run_id, parallel = False, batch_size = b_sz)
        _  = """
        var start_time = perf_counter_ns()
        training(model, data_repo.train_data, b_sz, logger)
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
                t"\t{correct}/{COUNT_TEST} correct\n\t{elapsed // 1_000_000}ms"
                t" for testing."
            )
        except e:
            print(e, file=stderr)
        # TODO: SAVE THE MODEL TO A FILE
        """
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
