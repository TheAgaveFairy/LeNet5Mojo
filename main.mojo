from std.subprocess import run as subProcessRun
from std.random import seed
from std.sys.info import num_logical_cores
from std.sys import stderr
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from std.pathlib import Path
from std.sys import argv
import std.os as os
import std.benchmark as benchmark
from std.reflection.reflect import reflect  # TODO: remove unused imports
from std.gpu.host import DeviceContext

from image import Image
from cpu.model import LeNet5
from constants import ftype, act_fn, ALPHA, DISPLAY
from cpu.ops import training, trainingParallel, testing, trainBatch
from cpu.arena import CPUBumpArenaAllocator, CPUSystemAllocator

from accel import (
    # LeNet5GPU,
    batchedForward,
    DeviceSession,
    GPUBumpArenaAllocator,
)
from accel.ops import (
    conv1FusedKernel,
    maxPool1Kernel,
    conv2FusedKernel,
    maxPool2Kernel,
    conv3FusedKernel,
    matMulFusedKernel,
    gatherOutputsKernel,
    compareBuffers,
)

from helpers import showProgress
from dataloader import MNISTDataRepository
from resultlogger import MultiFileLogger

# note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper
# the penultimate layer of size 84 isnt included either, see their paper

comptime COUNT_TRAIN = MNISTDataRepository.COUNT_TRAIN
comptime COUNT_TEST = MNISTDataRepository.COUNT_TEST

comptime act_fn_name = reflect[act_fn].base_name()


@fieldwise_init
struct TrainingSummary(Copyable, Movable):
    var elapsed_ns: UInt


@fieldwise_init
struct InferenceSummary(Copyable, Movable):
    var correct: Int
    var count: Int
    var elapsed_ns: UInt


def runTrain(
    mut model: LeNet5,
    data: List[Image],
    *,
    parallel: Bool = False,
    batch_size: Int = 300,
) -> TrainingSummary:
    var start_time = perf_counter_ns()
    if parallel:
        trainingParallel(model, data, batch_size)
    else:
        training(model, data, batch_size)
    return TrainingSummary(perf_counter_ns() - start_time)


def runTrain(
    mut model: LeNet5,
    data: List[Image],
    mut logger: MultiFileLogger,
    *,
    parallel: Bool = False,
    batch_size: Int = 300,
) -> TrainingSummary:
    var start_time = perf_counter_ns()
    if parallel:
        trainingParallel(model, data, batch_size, logger)
    else:
        training(model, data, batch_size, logger)
    return TrainingSummary(perf_counter_ns() - start_time)


def runTest(
    model: LeNet5,
    data: List[Image],
) -> InferenceSummary:
    var start_time = perf_counter_ns()
    var correct = testing(model, data)
    return InferenceSummary(correct, len(data), perf_counter_ns() - start_time)


def runTest(
    model: LeNet5,
    data: List[Image],
    mut logger: MultiFileLogger,
) -> InferenceSummary:
    var start_time = perf_counter_ns()
    var correct = testing(model, data)
    var elapsed_ns = perf_counter_ns() - start_time
    try:
        logger.logInferenceResult(
            "CPU", elapsed_ns, correct, len(data), 1, ftype
        )
    except e:
        print(e, file=stderr)
    return InferenceSummary(correct, len(data), elapsed_ns)


def trainAndTest(
    mut model: LeNet5,
    data_repo: MNISTDataRepository,
    alloc: String,
    run_id: String,
    *,
    parallel: Bool = False,
    batch_size: Int = 300,
) raises:
    var act_name = materialize[act_fn_name]()
    var threads = 1 if not parallel else num_logical_cores()
    var infer_name = t"mode=infer_alloc={alloc}_thread={threads}_bs={batch_size}_act={act_name}_run={run_id}"
    var train_name = t"mode=train_alloc={alloc}_thread={threads}_bs={batch_size}_act={act_name}_run={run_id}"
    var logger = MultiFileLogger(
        "results/", String(infer_name), String(train_name)
    )
    var train_res = runTrain(
        model,
        data_repo.train_data,
        logger,
        parallel=parallel,
        batch_size=batch_size,
    )
    if DISPLAY:
        print(
            t"\n\t{alloc} training:", train_res.elapsed_ns // 1_000_000, "ms."
        )
    var infer_res = runTest(model, data_repo.test_data, logger)
    if DISPLAY:
        print(
            "\t",
            infer_res.correct,
            "/",
            infer_res.count,
            "correct\n\t",
            infer_res.elapsed_ns // 1_000_000,
            "ms for testing.",
        )
    var training_ms = train_res.elapsed_ns // 1_000_000
    var testing_ms = infer_res.elapsed_ns // 1_000_000
    print(
        t"alloc={alloc}, act_fn={act_name}, threads={threads}, ALPHA={ALPHA},"
        t" correct={infer_res.correct}, total_count={infer_res.count},"
        t" ftype={ftype}, batch_size={batch_size}, training_ms={training_ms},"
        t" testing_ms={testing_ms}"
    )


def main() raises:
    var args = argv()
    var argc = len(args)
    if argc > 1:
        if args[1] == "--help":
            print("-D ALPHA=[1..1000], -D ACT_FN, see constants.mojo")
            return

    var run_id: String
    try:
        run_id = subProcessRun("date +%s")
    except:
        run_id = "unknown"

    var data_repo = MNISTDataRepository()

    var batch_sizes = [300]  # 100, 300, 600, 1000] # prefer 300
    for b_sz in batch_sizes:  # range(tests_to_run):
        seed(42069)  # seeds 'random', we could 'search' for a better seed
        data_repo.shuffle()

        var arena = CPUBumpArenaAllocator(LeNet5._calcArenaSize())
        var arena_model = LeNet5(arena)
        arena_model.zero()
        arena_model.randomizeWeights()

        trainAndTest(
            arena_model,
            data_repo,
            "arena",
            run_id,
            parallel=True,
            batch_size=b_sz,
        )

        try:
            arena_model.saveToFile(Path("models/deleteme.test"))
        except e:
            print(e, file=stderr)
        benchmark.keep(
            arena
        )  # FIXME: joint origins have been discussed. could also wrap together into an AllocatorBox[LeNet5, Arena] so the lifetimes are tied

    # TESTING A PRETRAINED VERSION FROM OLD FILE

    comptime model_name = "models/deleteme.test"
    comptime saved_model_dtype = ftype

    print("Loading and testing a saved model: '" + model_name + "'")
    var modelCPU = LeNet5()
    modelCPU.loadFromFile[saved_model_dtype](model_name)
    var saved_res = runTest(modelCPU, data_repo.test_data)
    print("\t", saved_res.correct, "/", saved_res.count, "correct")
    print("\t", saved_res.elapsed_ns // 1_000_000, "ms")

    try:
        with DeviceContext() as ctx:
            var gpu_session = DeviceSession[GPUBumpArenaAllocator](ctx)
            gpu_session.bufs.loadCPUWeights(modelCPU)
            # compareBuffers[LeNet5.w01_layout](ctx, gpu_session.bufs.w01_storage, modelCPU.weight0_1.ptr, label = "layer1")
            var device_name = ctx.name()
            print(
                "\nDevice found:",
                device_name,
                ". Compiling kernels and testing...",
            )
            comptime batch_size = 75  # more than ~75 fails "uses too much parameter space"

            var conv1 = ctx.compile_function[conv1FusedKernel[batch_size]]()
            var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
            var conv2 = ctx.compile_function[conv2FusedKernel[batch_size]]()
            var pool2 = ctx.compile_function[maxPool2Kernel[batch_size]]()
            var conv3 = ctx.compile_function[conv3FusedKernel[batch_size]]()
            var matmul = ctx.compile_function[matMulFusedKernel[batch_size]]()
            var gather = ctx.compile_function[gatherOutputsKernel[batch_size]]()

            var start_time = perf_counter_ns()

            var correct = batchedForward[batch_size](
                data_repo.train_data,
                gpu_session.model,
                conv1,
                pool1,
                conv2,
                pool2,
                conv3,
                matmul,
                gather,
            )
            var end_time = perf_counter_ns()
            var elapsed = end_time - start_time  # // 1_000_000

            print("\t", correct, "/", COUNT_TRAIN, "correct")
            print("\t", elapsed // 1_000_000, "ms")
            # TODO: wire up logger for GPU inference result
            # logger.logInferenceResult(
            #     device_name, elapsed, correct, COUNT_TRAIN, batch_size, ftype
            # )
    except e:
        print("ERROR IN MAIN", e, file=stderr)
