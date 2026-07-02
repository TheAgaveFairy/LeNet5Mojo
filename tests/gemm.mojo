from layout import Layout, LayoutTensor, lt_to_tt
from std.bit import next_power_of_two  # prev_power_of_two
from std.math import ceildiv, abs, max
import std.sys.defines as defines
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.benchmark.compiler import keep

from std.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    HostBuffer,
    DeviceFunction,
)
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives import block
from std.gpu.memory import AddressSpace

from constants import (
    ftype,
    sftype,
    nelts,
    act_fn,
    FeatureLayouts,
    WeightLayouts,
    BiasLayouts,
)
from accel.model import LeNet5GPU
from accel.feature import FeatureGPU, FeatureGPUBuffers
from origin_util import untrack, untrack_imm

from linalg.matmul import matmul # (out_tt, a_tt, b_tt, ctx = None)

# override at build time: -D TILE_SIZE=16
comptime TILE_SIZE = defines.get_defined_int["TILE_SIZE", nelts]()

def naiveCPU[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    bias_layout: Layout,
    epilogue_act: Bool = False,
](
    a: LayoutTensor[ftype, a_layout, _],
    b: LayoutTensor[ftype, b_layout, _],
    c: LayoutTensor[ftype, c_layout, MutUntrackedOrigin],
    bias: LayoutTensor[ftype, bias_layout, _],
) -> None:
    # a = (M, K)
    # b = (K, N)
    # c = (M, N), bias = (M,), act_fn epilogue optional
    comptime M = a.shape[0]()
    comptime K = a.shape[1]()
    comptime N = b.shape[1]()
    comptime assert b.shape[0]() == K, "bad shapes (a or b)"
    comptime assert c.shape[0]() == M, "bad C shape[0]"
    comptime assert c.shape[1]() == N, "bad C shape[1]"
    comptime assert bias_layout.size() == M, "bias must be (M,)"

    for i in range(M):
        for j in range(N):
            var accum = rebind[sftype](bias[i])
            for k in range(K):
                accum += rebind[sftype](a[i, k] * b[k, j])
            comptime if epilogue_act:
                accum = act_fn.simdForward[ftype, 1](accum)
            c[i, j] = accum

def tiledCPU[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    bias_layout: Layout,
    epilogue_act: Bool = False,
](
    a: LayoutTensor[ftype, a_layout, _],
    b: LayoutTensor[ftype, b_layout, _],
    c: LayoutTensor[ftype, c_layout, MutUntrackedOrigin],
    bias: LayoutTensor[ftype, bias_layout, _],
) -> None:
    # a(M, K) @ b(K, N) + bias(M,) = c(M, N), act_fn epilogue optional
    comptime M = a.shape[0]()
    comptime K = a.shape[1]()
    comptime N = b.shape[1]()

    comptime assert bias_layout.size() == M, "bias must be (M,)"
    # zero-padded tiles mean no ragged SIMD tail — divisibility is all we need
    comptime assert TILE_SIZE % nelts == 0, "TILE_SIZE must be a multiple of nelts"

    comptime tile_layout = Layout.row_major(TILE_SIZE, TILE_SIZE)
    comptime BM = ceildiv(M, TILE_SIZE)
    comptime BN = ceildiv(N, TILE_SIZE)
    comptime BK = ceildiv(K, TILE_SIZE)

    # prep tiles
    var ta = LayoutTensor[ftype, tile_layout, MutUntrackedOrigin](alloc[sftype](TILE_SIZE * TILE_SIZE))
    var tb = LayoutTensor[ftype, tile_layout, MutUntrackedOrigin](alloc[sftype](TILE_SIZE * TILE_SIZE))
    var tc = LayoutTensor[ftype, tile_layout, MutUntrackedOrigin](alloc[sftype](TILE_SIZE * TILE_SIZE))

    for bm in range(BM):
        for bn in range(BN):
            # super important to zero this
            _ = tc.fill(0.0)
            for bk in range(BK):

                # pack tile a
                var global_m = bm * TILE_SIZE
                for i in range(TILE_SIZE):
                    var global_k = bk * TILE_SIZE
                    for j in range(TILE_SIZE):
                        if global_m >= M or global_k >= K:
                            ta[i, j] = 0.0
                        else:
                            ta[i, j] = a[global_m, global_k]
                        global_k += 1
                    global_m += 1

                # pack tile b
                var global_k = bk * TILE_SIZE
                for i in range(TILE_SIZE):
                    var global_n = bn * TILE_SIZE
                    for j in range(TILE_SIZE):
                        if global_k >= K or global_n >= N:
                            tb[j, i] = 0.0
                        else:
                            tb[j, i] = b[global_k, global_n] # transpose
                        global_n += 1
                    global_k += 1

                # can combine those two packs into the same n^2 control flow or make into a fn()

                # SIMD dot-product microkernel: both tile rows unit-stride
                # (tb transposed), FMA accumulate, one horizontal reduce per output
                for ti in range(TILE_SIZE):
                    for tj in range(TILE_SIZE):
                        var accum = SIMD[ftype, nelts](0.0)
                        comptime for tk in range(0, TILE_SIZE, nelts):
                            var x = ta.ptr.load[width=nelts](ti * TILE_SIZE + tk)
                            var y = tb.ptr.load[width=nelts](tj * TILE_SIZE + tk)
                            accum = x.fma(y, accum)
                        tc[ti, tj] += accum.reduce_add()

            # store tc into global c output (bias + optional act epilogue here)
            var global_m = bm * TILE_SIZE
            var global_n = bn * TILE_SIZE
            for ti in range(TILE_SIZE):
                for tj in range(TILE_SIZE):
                    if global_m + ti < M and global_n + tj < N:
                        var v = rebind[sftype](tc[ti, tj]) + rebind[sftype](bias[global_m + ti])
                        comptime if epilogue_act:
                            v = act_fn.simdForward[ftype, 1](v)
                        c[global_m + ti, global_n + tj] = v

    ta.ptr.free()
    tb.ptr.free()
    tc.ptr.free()

# --- benchmarking -------------------------------------------------------------
#   pixi run mojo -I src tests/gemm.mojo
#   pixi run mojo -D TILE_SIZE=16 -I src tests/gemm.mojo   # sweep tile size
#
# Verifies naive/tiled against linalg.matmul first, then benchmarks each at a
# few sizes (squares + one LeNet-ish im2col shape). Add a new hand-written
# kernel (SIMD etc.) by dropping another bench_function into benchGemms.

comptime Tens[layout: Layout] = LayoutTensor[ftype, layout, MutUntrackedOrigin]


# Deterministic fill in [-1, 1) — signed so ReLU-style epilogues actually clip,
# small so float32 accumulation stays well-conditioned.
def fillVaried(ptr: UnsafePointer[sftype, MutUntrackedOrigin], count: Int):
    for idx in range(count):
        ptr[idx] = sftype((idx * 2654435761) % 1009 - 504) / 504


def maxAbsDiff(
    x: UnsafePointer[sftype, MutUntrackedOrigin],
    y: UnsafePointer[sftype, MutUntrackedOrigin],
    count: Int,
) -> sftype:
    var worst = sftype(0.0)
    for idx in range(count):
        worst = max(worst, abs(x[idx] - y[idx]))
    return worst


def verify[M: Int, K: Int, N: Int, epilogue_act: Bool = False]() raises:
    comptime al = Layout.row_major(M, K)
    comptime bl = Layout.row_major(K, N)
    comptime cl = Layout.row_major(M, N)
    comptime biasl = Layout.row_major(M)
    var a = Tens[al](alloc[sftype](M * K))
    var b = Tens[bl](alloc[sftype](K * N))
    var bias = Tens[biasl](alloc[sftype](M))
    var c_ref = Tens[cl](alloc[sftype](M * N))
    var c_out = Tens[cl](alloc[sftype](M * N))
    fillVaried(a.ptr, M * K)
    fillVaried(b.ptr, K * N)
    fillVaried(bias.ptr, M)

    # reference: linalg matmul, then fold in the same bias + act epilogue
    matmul[transpose_b=False](lt_to_tt(c_ref), lt_to_tt(a), lt_to_tt(b), None)
    for i in range(M):
        for j in range(N):
            var v = rebind[sftype](c_ref[i, j]) + rebind[sftype](bias[i])
            comptime if epilogue_act:
                v = act_fn.simdForward[ftype, 1](v)
            c_ref[i, j] = v

    naiveCPU[epilogue_act=epilogue_act](a, b, c_out, bias)
    var naive_diff = maxAbsDiff(c_ref.ptr, c_out.ptr, M * N)

    tiledCPU[epilogue_act=epilogue_act](a, b, c_out, bias)
    var tiled_diff = maxAbsDiff(c_ref.ptr, c_out.ptr, M * N)

    a.ptr.free()
    b.ptr.free()
    bias.ptr.free()
    c_ref.ptr.free()
    c_out.ptr.free()

    comptime tol = sftype(1e-4)
    print(t"verify {M}x{K}x{N} act={epilogue_act}: naive diff {naive_diff}, tiled diff {tiled_diff}")
    if naive_diff > tol or tiled_diff > tol:
        raise Error("GEMM verification failed vs linalg.matmul")


# One call registers all three impls at (M, K, N). Alloc/fill once per bench_fn;
# only the matmul call is timed inside b.iter.
def benchGemms[M: Int, K: Int, N: Int](mut bench: Bench) raises:
    comptime al = Layout.row_major(M, K)
    comptime bl = Layout.row_major(K, N)
    comptime cl = Layout.row_major(M, N)
    comptime biasl = Layout.row_major(M)
    comptime flops = 2 * M * N * K
    var suffix = String(M) + "x" + String(K) + "x" + String(N)

    @parameter
    def bench_naive(mut b: Bencher) raises:
        var a = Tens[al](alloc[sftype](M * K))
        var bt = Tens[bl](alloc[sftype](K * N))
        var c = Tens[cl](alloc[sftype](M * N))
        var bias = Tens[biasl](alloc[sftype](M))
        fillVaried(a.ptr, M * K)
        fillVaried(bt.ptr, K * N)
        fillVaried(bias.ptr, M)

        @parameter
        def work():
            naiveCPU(a, bt, c, bias)
            keep(c.ptr)

        b.iter[work]()
        a.ptr.free()
        bt.ptr.free()
        c.ptr.free()
        bias.ptr.free()

    @parameter
    def bench_tiled(mut b: Bencher) raises:
        var a = Tens[al](alloc[sftype](M * K))
        var bt = Tens[bl](alloc[sftype](K * N))
        var c = Tens[cl](alloc[sftype](M * N))
        var bias = Tens[biasl](alloc[sftype](M))
        fillVaried(a.ptr, M * K)
        fillVaried(bt.ptr, K * N)
        fillVaried(bias.ptr, M)

        @parameter
        def work():
            tiledCPU(a, bt, c, bias)
            keep(c.ptr)

        b.iter[work]()
        a.ptr.free()
        bt.ptr.free()
        c.ptr.free()
        bias.ptr.free()

    @parameter
    def bench_linalg(mut b: Bencher) raises:
        var a = Tens[al](alloc[sftype](M * K))
        var bt = Tens[bl](alloc[sftype](K * N))
        var c = Tens[cl](alloc[sftype](M * N))
        fillVaried(a.ptr, M * K)
        fillVaried(bt.ptr, K * N)

        @parameter
        def work() raises:
            matmul[transpose_b=False](lt_to_tt(c), lt_to_tt(a), lt_to_tt(bt), None)
            keep(c.ptr)

        b.iter[work]()
        a.ptr.free()
        bt.ptr.free()
        c.ptr.free()

    var measures = [ThroughputMeasure(BenchMetric.flops, flops)]
    bench.bench_function[bench_naive](BenchId("naive_" + suffix), measures=measures.copy())
    bench.bench_function[bench_tiled](BenchId("tiled_" + suffix), measures=measures.copy())
    bench.bench_function[bench_linalg](BenchId("linalg_" + suffix), measures=measures.copy())


def main() raises:
    # correctness first: aligned size, ragged size (tile edge padding), act epilogue
    verify[32, 32, 32]()
    verify[33, 50, 17]()
    verify[33, 50, 17, epilogue_act=True]()

    var bench = Bench(BenchConfig(max_iters=10000))
    benchGemms[32, 32, 32](bench)
    benchGemms[128, 128, 128](bench)
    benchGemms[256, 256, 256](bench)
    # LeNet-ish im2col shape: conv2 weights (16, 6*5*5) @ cols (150, batch*10*10)
    benchGemms[16, 150, 1600](bench)
    bench.dump_report()
