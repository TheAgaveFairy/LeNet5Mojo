"""Architecture dimensions, dtype, and compile-time activation/allocator selection."""

from std.sys import simd_width_of
import std.sys.defines as defines
from layout import Layout

from activation_fn import *
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator, CPUSystemAllocator
from accel.arena import GPUAllocator, GPUBumpArenaAllocator, GPUSystemAllocator


# Architecture dimensions
comptime LENGTH_KERNEL = 5
comptime LENGTH_KERNEL_SQ = LENGTH_KERNEL * LENGTH_KERNEL

comptime LENGTH_FEATURE0 = 32
comptime LENGTH_FEATURE1 = LENGTH_FEATURE0 - LENGTH_KERNEL + 1
comptime LENGTH_FEATURE2 = LENGTH_FEATURE1 >> 1
comptime LENGTH_FEATURE3 = LENGTH_FEATURE2 - LENGTH_KERNEL + 1
comptime LENGTH_FEATURE4 = LENGTH_FEATURE3 >> 1
comptime LENGTH_FEATURE5 = LENGTH_FEATURE4 - LENGTH_KERNEL + 1

comptime INPUT = 1
comptime LAYER1 = 6
comptime LAYER2 = LAYER1
comptime LAYER3 = 16  # 16
comptime LAYER4 = LAYER3
comptime LAYER5 = 120
comptime OUTPUT = 10

comptime NUM_WEIGHTS = 51902  # hardcoding here for simple referencing, can be calculated

comptime ALPHA = Float32(defines.get_defined_int["ALPHA", 500]()) / 1000
comptime PADDING = 2

# RNG seed — single source of truth (weight init + shuffle). Override at runtime with --seed.
comptime DEFAULT_SEED = 42

comptime IMAGE_SIZE = 28
comptime PADDED_SIZE = IMAGE_SIZE + 2 * PADDING  # == LENGTH_FEATURE0

# Numeric type — change 'ftype' (floating point type) here to switch the whole model (float64, bf16, etc.)
# GPU doesn't like fp64, fp16 is making some things mad, too # TODO: make it work - shuffles might not work
comptime ftype = DType.float32  # DType.float64 if defines.is_defined["float64"]() else DType.float32
comptime sftype = Scalar[ftype]
comptime nelts = simd_width_of[ftype]()

comptime CPU_TILE_SIZE = defines.get_defined_int["CPU_TILE_SIZE", nelts]()

comptime GPU_TILE_SIZE = defines.get_defined_int["GPU_TILE_SIZE", 16]()
comptime GPU_STREAM_BATCH_SIZE = defines.get_defined_int[
    "GPU_STREAM_BATCH_SIZE", 100
]()
comptime NUM_GPU_STREAMS = defines.get_defined_int[
    "NUM_GPU_STREAMS", 12
]()  # knee moved 5 -> 12 after conv3 GEMM (2026-07-02 sweep, bs=100: 1.42M fps @ s=12
# vs 1.30M @ s=8 vs 1.20M @ s=5; s=16 flat — knee). Lighter kernels pack more streams.
comptime MAX_GPU_STREAMS = 16  # sanity cap for --num-streams; raise freely, slots are runtime-alloc'd

comptime act_fn = GELU if defines.is_defined[
    "GELU"
]() else GELUTanh if defines.is_defined[
    "GELUTanh"
]() else GELUFast if defines.is_defined[
    "GELUFast"
]() else Sigmoid if defines.is_defined[
    "Sigmoid"
]() else Tanh if defines.is_defined[
    "Tanh"
]() else ReLU

# Compile-time allocator selection (mirrors act_fn). Bump arena by default; the
# system allocator is a benchmarking baseline. -D CPU_SYSTEM_ALLOC / -D GPU_SYSTEM_ALLOC.
comptime CPU_ALLOCATOR = CPUSystemAllocator if defines.is_defined[
    "CPU_SYSTEM_ALLOC"
]() else CPUBumpArenaAllocator
comptime GPU_ALLOCATOR = GPUSystemAllocator if defines.is_defined[
    "GPU_SYSTEM_ALLOC"
]() else GPUBumpArenaAllocator

comptime DISPLAY = True if defines.is_defined["DISPLAY"]() else False


struct FeatureLayouts:
    comptime input = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    comptime layer1 = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    comptime layer2 = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    comptime layer3 = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    comptime layer4 = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    comptime layer5 = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    comptime output = Layout.row_major(OUTPUT)


struct BatchedFeatureLayouts[bs: Int = GPU_STREAM_BATCH_SIZE]():
    comptime input = Layout.row_major(
        Self.bs, INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0
    )
    # layer1 fused away
    comptime layer2 = Layout.row_major(
        Self.bs, LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2
    )
    comptime layer3 = Layout.row_major(
        Self.bs, LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3
    )
    comptime layer4 = Layout.row_major(
        Self.bs, LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4
    )
    comptime layer5 = Layout.row_major(Self.bs, LAYER5)  # "GEMM" re-shaped


struct WeightLayouts:
    comptime w01 = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    comptime w23 = Layout.row_major(
        LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL
    )
    comptime w45 = Layout.row_major(
        LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL
    )
    # w45 transposed to GEMM b(K, N): row-major (ic*kw*kh, oc). One-time
    # device copy made at weight upload; w45's native (ic, oc, kw, kh) order
    # can't be a 2D strided view (k mixes ic and kw*kh at different strides).
    comptime w45g = Layout.row_major(
        LAYER4 * LENGTH_KERNEL * LENGTH_KERNEL, LAYER5
    )
    comptime w56 = Layout.row_major(
        LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT
    )


struct BiasLayouts:
    comptime b01 = Layout.row_major(LAYER1)
    comptime b23 = Layout.row_major(LAYER3)
    comptime b45 = Layout.row_major(LAYER5)
    comptime b56 = Layout.row_major(OUTPUT)
