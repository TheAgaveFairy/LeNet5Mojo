# A/B microbenchmark: branchless vs branchy argmax in maxpool (forward + backward).
#
#   pixi run mojo -I src tests/bench_maxpool.mojo
#
# The production maxPoolForward/maxPoolBackward track the window argmax without an `if`:
#     x0 += ismax * (l0 - x0)          # ismax in {0,1}
# This benchmarks that trick against the natural branchy form (`if v > best: best = v; ...`)
# at the two real LeNet pool shapes: (6,28)->(6,14) and (16,10)->(16,5), both stride 2.
# The production (branchless) ops are imported so we measure the real code, not a copy.

from cpu.ops import maxPoolForward, maxPoolBackward
from cpu.arena import CPUBumpArenaAllocator as CPUArena
from layout import Layout, LayoutTensor
from constants import (
    ftype,
    sftype,
    LAYER1,
    LAYER3,
    LENGTH_FEATURE1,
    LENGTH_FEATURE2,
    LENGTH_FEATURE3,
    LENGTH_FEATURE4,
)
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.benchmark.compiler import keep
from std.sys.info import size_of

comptime Tens[layout: Layout] = LayoutTensor[ftype, layout, MutAnyOrigin]


# Deterministic, varied fill so argmax lands in different window slots (exercises the branch).
def fillVaried(ptr: UnsafePointer[sftype, MutAnyOrigin], count: Int):
    for idx in range(count):
        ptr[idx] = sftype((idx * 2654435761) % 1009)


# --- branchy ("normal") variants, mirroring the production signatures --------
def maxPoolForwardBranchy[
    num_channels: Int, in_feat_size: Int, out_feat_size: Int
](
    input: Tens[Layout.row_major(num_channels, in_feat_size, in_feat_size)],
    output: Tens[Layout.row_major(num_channels, out_feat_size, out_feat_size)],
):
    comptime lenx = in_feat_size // out_feat_size
    comptime leny = in_feat_size // out_feat_size
    comptime for c in range(num_channels):
        comptime for i in range(out_feat_size):
            comptime for j in range(out_feat_size):
                var best = rebind[sftype](input[c, i * lenx, j * leny])
                comptime for x in range(lenx):
                    comptime for y in range(leny):
                        var v = rebind[sftype](
                            input[c, i * lenx + x, j * leny + y]
                        )
                        if v > best:
                            best = v
                output[c, i, j] = best


def maxPoolBackwardBranchy[
    num_channels: Int, in_feat_size: Int, out_feat_size: Int
](
    input: Tens[Layout.row_major(num_channels, in_feat_size, in_feat_size)],
    inerror: Tens[Layout.row_major(num_channels, in_feat_size, in_feat_size)],
    outerror: Tens[Layout.row_major(num_channels, out_feat_size, out_feat_size)],
):
    comptime len0 = in_feat_size // out_feat_size
    comptime len1 = in_feat_size // out_feat_size
    comptime for i in range(num_channels):
        for o0 in range(out_feat_size):
            for o1 in range(out_feat_size):
                var bx = 0
                var by = 0
                var best = rebind[sftype](input[i, o0 * len0, o1 * len1])
                for l0 in range(len0):
                    for l1 in range(len1):
                        var v = rebind[sftype](
                            input[i, o0 * len0 + l0, o1 * len1 + l1]
                        )
                        if v > best:
                            best = v
                            bx = l0
                            by = l1
                inerror[i, o0 * len0 + bx, o1 * len1 + by] = outerror[i, o0, o1]


# --- benchmark scaffolding ---------------------------------------------------
# Two pool stages; alloc once per bench_fn, time only the op call in b.iter.
comptime IC1 = LAYER1  # 6
comptime IN1 = LENGTH_FEATURE1  # 28
comptime OUT1 = LENGTH_FEATURE2  # 14
comptime IC2 = LAYER3  # 16
comptime IN2 = LENGTH_FEATURE3  # 10
comptime OUT2 = LENGTH_FEATURE4  # 5


@parameter
def bench_fwd_branchless(mut b: Bencher) raises:
    var arena = CPUArena(
        (IC1 * IN1 * IN1 + IC1 * OUT1 * OUT1 + IC2 * IN2 * IN2 + IC2 * OUT2 * OUT2)
        * size_of[ftype]()
    )
    var in1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1))
    var out1 = Tens[Layout.row_major(IC1, OUT1, OUT1)](arena.alloc[sftype](IC1 * OUT1 * OUT1))
    var in2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2))
    var out2 = Tens[Layout.row_major(IC2, OUT2, OUT2)](arena.alloc[sftype](IC2 * OUT2 * OUT2))
    fillVaried(in1.ptr, IC1 * IN1 * IN1)
    fillVaried(in2.ptr, IC2 * IN2 * IN2)

    @parameter
    def work():
        maxPoolForward[IC1, IN1, OUT1](in1, out1)
        maxPoolForward[IC2, IN2, OUT2](in2, out2)
        keep(out1.ptr)
        keep(out2.ptr)

    b.iter[work]()


@parameter
def bench_fwd_branchy(mut b: Bencher) raises:
    var arena = CPUArena(
        (IC1 * IN1 * IN1 + IC1 * OUT1 * OUT1 + IC2 * IN2 * IN2 + IC2 * OUT2 * OUT2)
        * size_of[ftype]()
    )
    var in1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1))
    var out1 = Tens[Layout.row_major(IC1, OUT1, OUT1)](arena.alloc[sftype](IC1 * OUT1 * OUT1))
    var in2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2))
    var out2 = Tens[Layout.row_major(IC2, OUT2, OUT2)](arena.alloc[sftype](IC2 * OUT2 * OUT2))
    fillVaried(in1.ptr, IC1 * IN1 * IN1)
    fillVaried(in2.ptr, IC2 * IN2 * IN2)

    @parameter
    def work():
        maxPoolForwardBranchy[IC1, IN1, OUT1](in1, out1)
        maxPoolForwardBranchy[IC2, IN2, OUT2](in2, out2)
        keep(out1.ptr)
        keep(out2.ptr)

    b.iter[work]()


@parameter
def bench_bwd_branchless(mut b: Bencher) raises:
    var arena = CPUArena(
        (2 * IC1 * IN1 * IN1 + IC1 * OUT1 * OUT1 + 2 * IC2 * IN2 * IN2 + IC2 * OUT2 * OUT2)
        * size_of[ftype]()
    )
    var in1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1))
    var ie1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1)).fill(0.0)
    var oe1 = Tens[Layout.row_major(IC1, OUT1, OUT1)](arena.alloc[sftype](IC1 * OUT1 * OUT1)).fill(1.0)
    var in2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2))
    var ie2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2)).fill(0.0)
    var oe2 = Tens[Layout.row_major(IC2, OUT2, OUT2)](arena.alloc[sftype](IC2 * OUT2 * OUT2)).fill(1.0)
    fillVaried(in1.ptr, IC1 * IN1 * IN1)
    fillVaried(in2.ptr, IC2 * IN2 * IN2)

    @parameter
    def work():
        maxPoolBackward[IC1, IN1, OUT1](in1, ie1, oe1)
        maxPoolBackward[IC2, IN2, OUT2](in2, ie2, oe2)
        keep(ie1.ptr)
        keep(ie2.ptr)

    b.iter[work]()


@parameter
def bench_bwd_branchy(mut b: Bencher) raises:
    var arena = CPUArena(
        (2 * IC1 * IN1 * IN1 + IC1 * OUT1 * OUT1 + 2 * IC2 * IN2 * IN2 + IC2 * OUT2 * OUT2)
        * size_of[ftype]()
    )
    var in1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1))
    var ie1 = Tens[Layout.row_major(IC1, IN1, IN1)](arena.alloc[sftype](IC1 * IN1 * IN1)).fill(0.0)
    var oe1 = Tens[Layout.row_major(IC1, OUT1, OUT1)](arena.alloc[sftype](IC1 * OUT1 * OUT1)).fill(1.0)
    var in2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2))
    var ie2 = Tens[Layout.row_major(IC2, IN2, IN2)](arena.alloc[sftype](IC2 * IN2 * IN2)).fill(0.0)
    var oe2 = Tens[Layout.row_major(IC2, OUT2, OUT2)](arena.alloc[sftype](IC2 * OUT2 * OUT2)).fill(1.0)
    fillVaried(in1.ptr, IC1 * IN1 * IN1)
    fillVaried(in2.ptr, IC2 * IN2 * IN2)

    @parameter
    def work():
        maxPoolBackwardBranchy[IC1, IN1, OUT1](in1, ie1, oe1)
        maxPoolBackwardBranchy[IC2, IN2, OUT2](in2, ie2, oe2)
        keep(ie1.ptr)
        keep(ie2.ptr)

    b.iter[work]()


def main() raises:
    var bench = Bench(BenchConfig(max_iters=200000))
    bench.bench_function[bench_fwd_branchless](BenchId("fwd_branchless"))
    bench.bench_function[bench_fwd_branchy](BenchId("fwd_branchy"))
    bench.bench_function[bench_bwd_branchless](BenchId("bwd_branchless"))
    bench.bench_function[bench_bwd_branchy](BenchId("bwd_branchy"))
    bench.dump_report()
