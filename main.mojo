from std.subprocess import run as subProcessRun
from std.random import seed
from std.sys.info import num_logical_cores
from std.sys import stderr
from std.time import perf_counter_ns
from std.pathlib import Path
from std.sys import argv
import std.benchmark as benchmark
from std.reflection.reflect import reflect
from std.gpu.host import DeviceContext

from image import Image
from cpu.model import LeNet5
from constants import (
    ftype,
    act_fn,
    ALPHA,
    DISPLAY,
    GPU_STREAM_BATCH_SIZE,
    NUM_GPU_STREAMS,
)
from cpu.ops import (
    training,
    trainingParallel,
    testing,
    testingParallel,
)
from cpu.arena import CPUBumpArenaAllocator

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
)

from dataloader import MNISTDataRepository
from resultlogger import MultiFileLogger, ResultLogger

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
    parallel: Bool = True,
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
    parallel: Bool = True,
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
    *,
    parallel: Bool = True,
) -> InferenceSummary:
    var start_time = perf_counter_ns()
    var correct: Int
    if parallel:
        correct = testingParallel(model, data)
    else:
        correct = testing(model, data)
    return InferenceSummary(correct, len(data), perf_counter_ns() - start_time)


def runTest(
    model: LeNet5,
    data: List[Image],
    mut logger: MultiFileLogger,
    *,
    parallel: Bool = True,
) -> InferenceSummary:
    var start_time = perf_counter_ns()
    var correct: Int
    if parallel:
        correct = testingParallel(model, data)
    else:
        correct = testing(model, data)
    var elapsed_ns = perf_counter_ns() - start_time
    try:
        logger.logInferenceResult(
            "CPU", elapsed_ns, correct, len(data), 1, 1, ftype
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
    parallel: Bool = True,
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
    var infer_res = runTest(
        model, data_repo.test_data, logger, parallel=parallel
    )
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


def runGPUTest(
    model: LeNet5,
    data_repo: MNISTDataRepository,
    run_id: String,
) raises:
    comptime batch_size = GPU_STREAM_BATCH_SIZE  # more than ~120 fails on my RTX3070
    with DeviceContext() as ctx:
        var gpu_session = DeviceSession[GPUBumpArenaAllocator](ctx)
        gpu_session.bufs.loadCPUWeights(model)
        print(
            "\nDevice found:", ctx.name(), ". Compiling kernels and testing..."
        )

        var norm = ctx.compile_function[normalizeInputsKernel[batch_size]]()
        var conv1 = ctx.compile_function[conv1FusedKernel[batch_size]]()
        var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
        var conv2 = ctx.compile_function[conv2FusedKernel[batch_size]]()
        var pool2 = ctx.compile_function[maxPool2Kernel[batch_size]]()
        var conv3 = ctx.compile_function[conv3FusedKernel[batch_size]]()
        var matmul = ctx.compile_function[matMulFusedKernel[batch_size]]()
        var gather = ctx.compile_function[gatherOutputsKernel[batch_size]]()

        var batched_data = data_repo.getTrainBatch(0, COUNT_TRAIN)
        var gpu_logger = ResultLogger(
            String(
                t"results/gpu_infer_bs{batch_size}_act={act_fn_name}_run={run_id}.csv"
            )
        )

        var t0 = perf_counter_ns()
        var correct_s1 = batchedForwardMultiStream[batch_size, 1](
            ctx,
            batched_data,
            gpu_session.model,
            norm,
            conv1,
            pool1,
            conv2,
            pool2,
            conv3,
            matmul,
            gather,
        )
        var t1 = perf_counter_ns()
        print(
            t"batchedForwardMultiStream[s=1]: "
            t" {correct_s1}/{COUNT_TRAIN} correct, {(t1-t0)//1_000_000}ms"
        )
        try:
            gpu_logger.logInferenceResult(
                "GPU", t1 - t0, correct_s1, COUNT_TRAIN, batch_size, 1, ftype
            )
        except e:
            print(e, file=stderr)

        var correct_ms = batchedForwardMultiStream[batch_size, NUM_GPU_STREAMS](
            ctx,
            batched_data,
            gpu_session.model,
            norm,
            conv1,
            pool1,
            conv2,
            pool2,
            conv3,
            matmul,
            gather,
        )
        var t2 = perf_counter_ns()
        print(
            t"batchedForwardMultiStream[s={NUM_GPU_STREAMS}]:"
            t" {correct_ms}/{COUNT_TRAIN} correct, {(t2-t1)//1_000_000}ms"
        )
        try:
            gpu_logger.logInferenceResult(
                "GPU",
                t2 - t1,
                correct_ms,
                COUNT_TRAIN,
                batch_size,
                NUM_GPU_STREAMS,
                ftype,
            )
        except e:
            print(e, file=stderr)
        benchmark.keep(gpu_session)


def main() raises:
    var args = argv()
    if len(args) > 1 and args[1] == "--help":
        print("-D ALPHA=[1..1000], -D ACT_FN, see constants.mojo")
        return

    var run_id: String
    try:
        run_id = subProcessRun("date +%s")
    except:
        run_id = "unknown"

    var data_repo = MNISTDataRepository()

    comptime cpu_batch_size = 300
    seed(42069)

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
        batch_size=cpu_batch_size,
    )

    try:
        arena_model.saveToFile(Path("models/deleteme.test"))
    except e:
        print(e, file=stderr)
    benchmark.keep(
        arena
    )  # FIXME: joint origins — could wrap into CPUSession.{LeNet5, Arena}

    # load the model saved above and test it independently
    comptime model_name = "models/deleteme.test"
    print("Loading and testing a saved model: '" + model_name + "'")
    var modelCPU = LeNet5()
    modelCPU.loadFromFile[ftype](model_name)
    var saved_res = runTest(modelCPU, data_repo.test_data)
    print("\t", saved_res.correct, "/", saved_res.count, "correct")
    print("\t", saved_res.elapsed_ns // 1_000_000, "ms")

    runGPUTest(modelCPU, data_repo, run_id)
    benchmark.keep(data_repo)
