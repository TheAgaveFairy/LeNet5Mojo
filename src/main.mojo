from std.subprocess import run as subProcessRun
from std.random import seed
from std.sys.info import num_logical_cores
from std.sys import stderr
from std.time import perf_counter_ns
from std.pathlib import Path
import std.benchmark as benchmark
import std.os as os
from std.reflection.reflect import reflect
from std.gpu.host import DeviceContext
import std.sys.defines as defines

from image import Image
from cli import printHelp, CliArgs
from cpu.model import LeNet5, CPUSession
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

from accel import (
    DeviceSession,
    GPUBumpArenaAllocator,
)
from accel.ops import (
    _batchRun,
    StreamSlot,
    normalizeInputsKernel,
    conv1FusedKernel,
    maxPool1Kernel,
    conv2FusedKernel,
    maxPool2Kernel,
    conv3FusedKernel,
    matMulFusedKernel,
)

from dataloader import MNISTDataRepository
from resultlogger import MultiFileLogger, ResultLogger

# note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper
# the penultimate layer of size 84 isnt included either, see their paper

comptime COUNT_TRAIN = MNISTDataRepository.COUNT_TRAIN
comptime COUNT_TEST = MNISTDataRepository.COUNT_TEST

comptime act_fn_name = reflect[act_fn].base_name()

comptime N_WARMUP = defines.get_defined_int["N_WARMUP", 3]()
comptime N_PASSES = defines.get_defined_int["N_PASSES", 10]()


def main() raises:
    var cli = CliArgs.parse()
    if cli.help:
        printHelp()
        return
    var num_streams = cli.num_streams

    var run_id: String
    try:
        run_id = subProcessRun("date +%s")
    except:
        run_id = "unknown"

    var data_repo = MNISTDataRepository()

    if cli.bench_only: # cli.bench_only
        var model_path = String("models/deleteme.test")
        if not os.path.exists(model_path):
            print("cannot load model: file not found:", model_path, file=stderr)
            print(
                "run without --bench-only / -D BENCH_ONLY first to train and save.",
                file=stderr,
            )
            return
        print("BENCH_ONLY: loading '" + model_path + "'...")
        var modelCPU = LeNet5()
        modelCPU.loadFromFile[ftype](Path(model_path))
        var act_name = materialize[act_fn_name]()
        var threads = num_logical_cores()
        var bench_logger = MultiFileLogger(
            "results/",
            String(t"mode=bench_thread={threads}_act={act_name}_run={run_id}"),
            String(t"mode=bench_train_noop_run={run_id}"),
        )
        var infer_res = benchCPUInference(
            modelCPU, data_repo.test_data, bench_logger
        )
        var testing_ms = infer_res.elapsed_ns // 1_000_000
        var cpu_fps = UInt(infer_res.count) * 1_000_000_000 // infer_res.elapsed_ns
        var accuracy_pct = infer_res.correct * 100 // infer_res.count
        print(
            t"alloc=bench, act_fn={act_name}, threads={threads},"
            t" correct={infer_res.correct}, total_count={infer_res.count},"
            t" ftype={ftype}, testing_ms={testing_ms},"
            t" fps={cpu_fps}, accuracy_pct={accuracy_pct}"
        )
        runGPUTest(modelCPU, data_repo, run_id, num_streams)
        benchmark.keep(data_repo)
    else:
        comptime cpu_batch_size = 300
        comptime train_name = "models/deleteme.test"
        seed(42069)

        var session = CPUSession()
        session.model.zero()
        session.model.randomizeWeights()

        trainAndTest(
            session.model,
            data_repo,
            "arena",
            run_id,
            parallel=True,
            batch_size=cpu_batch_size,
        )

        try:
            session.model.saveToFile(Path(train_name))
        except e:
            print(e, file=stderr)

        # load the model saved above and test it independently
        comptime model_name = train_name#"models/deleteme.test"
        print("Loading and testing a saved model: '" + model_name + "'")
        var modelCPU = LeNet5()
        modelCPU.loadFromFile[ftype](model_name)
        var saved_res = runTest(modelCPU, data_repo.test_data)
        print("\t", saved_res.correct, "/", saved_res.count, "correct")
        print("\t", saved_res.elapsed_ns // 1_000_000, "ms")

        runGPUTest(modelCPU, data_repo, run_id, num_streams)
        #benchmark.keep(data_repo)


@fieldwise_init
struct TimingStats(Copyable, Movable):
    var median_ns: UInt
    var min_ns: UInt
    var max_ns: UInt


def _timing_stats(mut times: List[UInt]) -> TimingStats:
    @parameter
    def less_than(a: UInt, b: UInt) capturing -> Bool:
        return a < b

    sort[cmp_fn=less_than](Span(times))
    var n = len(times)
    var median = (times[n // 2] + times[(n - 1) // 2]) // 2
    return TimingStats(median, times[0], times[n - 1])


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
    logger: Optional[MultiFileLogger] = None,
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
    logger: Optional[MultiFileLogger] = None,
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
    if logger:
        try:
            logger.value().logInferenceResult(
                "CPU", elapsed_ns, correct, len(data), 1, 1, ftype
            )
        except e:
            print(e, file=stderr)
    return InferenceSummary(correct, len(data), elapsed_ns)


def benchCPUInference(
    model: LeNet5,
    data: List[Image],
    logger: Optional[MultiFileLogger] = None,
    *,
    parallel: Bool = True,
) raises -> InferenceSummary:
    for _ in range(N_WARMUP):
        var warmup = runTest(model, data, parallel=parallel)
        benchmark.keep(warmup)
    var times = List[UInt]()
    var correct = 0
    for i in range(N_PASSES):
        var res = runTest(model, data, parallel=parallel)
        times.append(res.elapsed_ns)
        if i == 0:
            correct = res.correct
    var stats = _timing_stats(times)
    if logger:
        try:
            logger.value().logInferenceResult(
                "CPU", stats.median_ns, correct, len(data), 1, 1, ftype
            )
        except e:
            print(e, file=stderr)
    var ns_per_img = stats.median_ns // UInt(len(data))
    print(
        t"  median={stats.median_ns//1_000_000}ms ({ns_per_img}ns/img),"
        t" min={stats.min_ns//1_000_000}ms, max={stats.max_ns//1_000_000}ms"
    )
    return InferenceSummary(correct, len(data), stats.median_ns)


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
    var infer_res = benchCPUInference(
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
    var cpu_fps = UInt(infer_res.count) * 1_000_000_000 // infer_res.elapsed_ns
    var accuracy_pct = infer_res.correct * 100 // infer_res.count
    print(
        t"alloc={alloc}, act_fn={act_name}, threads={threads}, ALPHA={ALPHA},"
        t" correct={infer_res.correct}, total_count={infer_res.count},"
        t" ftype={ftype}, batch_size={batch_size}, training_ms={training_ms},"
        t" testing_ms={testing_ms}, fps={cpu_fps}, accuracy_pct={accuracy_pct}"
    )


def runGPUTest(
    model: LeNet5,
    data_repo: MNISTDataRepository,
    run_id: String,
    num_streams: Int = NUM_GPU_STREAMS,
) raises:
    comptime batch_size = GPU_STREAM_BATCH_SIZE
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

        var batched_data = data_repo.getTestBatch(0, COUNT_TEST)
        var gpu_logger = ResultLogger(
            String(
                t"results/gpu_infer_bs{batch_size}_act={act_fn_name}_run={run_id}.csv"
            )
        )
        # actual images processed: drop remainder that doesn't fill a full batch
        comptime n_proc = (COUNT_TEST // batch_size) * batch_size
        var eff_batch = batch_size * num_streams

        # Allocate slots once — reused across warmup and timed passes
        var slots = alloc[StreamSlot[batch_size]](num_streams)
        for s in range(num_streams):
            (slots + s).init_pointee_move(StreamSlot[batch_size]())

        # warmup
        for _ in range(N_WARMUP):
            var wc = _batchRun[batch_size](
                slots, batched_data, gpu_session.model,
                norm, conv1, pool1, conv2, pool2, conv3, matmul, num_streams,
            )
            benchmark.keep(wc)

        # N_PASSES timed, take median
        var times = List[UInt]()
        var correct = 0
        for i in range(N_PASSES):
            var t = perf_counter_ns()
            var c = _batchRun[batch_size](
                slots, batched_data, gpu_session.model,
                norm, conv1, pool1, conv2, pool2, conv3, matmul, num_streams,
            )
            times.append(perf_counter_ns() - t)
            if i == 0:
                correct = c
        var stats = _timing_stats(times)
        var fps = UInt(n_proc) * 1_000_000_000 // stats.median_ns
        var ns = stats.median_ns // UInt(n_proc)
        var acc = correct * 100 // n_proc
        print(
            t"batchedForwardMultiStream[s={num_streams}]:"
            t" eff_batch={eff_batch}, {correct}/{n_proc} ({acc}%)"
            t" correct, {stats.median_ns//1_000_000}ms ({ns}ns/img),"
            t" {fps} fps [min={stats.min_ns//1_000_000}ms"
            t" max={stats.max_ns//1_000_000}ms]"
        )
        try:
            gpu_logger.logInferenceResult(
                "GPU",
                stats.median_ns,
                correct,
                n_proc,
                batch_size,
                num_streams,
                ftype,
            )
        except e:
            print(e, file=stderr)

        for s in range(num_streams):
            (slots + s).destroy_pointee()
        slots.free()

        benchmark.keep(gpu_session)
