from std.sys import simd_width_of
import std.sys.defines as defines
from std.utils.type_functions import ConditionalType
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

comptime NUM_WEIGHTS = 51902  # hardcoding here for simplicity, can be calculated

comptime ALPHA = Float32(defines.get_defined_int["ALPHA", 500]()) / 1000
comptime PADDING = 2

# RNG seed — single source of truth (weight init + shuffle). Override at runtime with --seed.
comptime DEFAULT_SEED = 42

comptime IMAGE_SIZE = 28
comptime PADDED_SIZE = IMAGE_SIZE + 2 * PADDING  # == LENGTH_FEATURE0

# Numeric type — change 'ftype' (floating point type) here to switch the whole model (float64, bf16, etc.)
# comptime ftype = DType.float64 #defines.get_defined_dtype["ftype", DType.float32]() # doesn't want to work
# GPU doesn't like fp64 # TODO: make it work - shuffles might not work
comptime ftype = DType.float32  # if defines.is_defined["float64"]() else DType.float32
comptime sftype = Scalar[ftype]
comptime nelts = simd_width_of[ftype]()

comptime GPU_STREAM_BATCH_SIZE = defines.get_defined_int[
    "GPU_STREAM_BATCH_SIZE", 100
]()
comptime NUM_GPU_STREAMS = defines.get_defined_int[
    "NUM_GPU_STREAMS", 5
]()  # saturation knee ~5-6 after Tier A conv3 (low-occupancy kernels leave headroom); 8 wasted

comptime act_fn = ConditionalType[
    Trait=ActivationFunction,
    If=defines.is_defined["GELU"](),
    Then=GELU,
    Else=ConditionalType[
        Trait=ActivationFunction,
        If=defines.is_defined["GELUTanh"](),
        Then=GELUTanh,
        Else=ConditionalType[
            Trait=ActivationFunction,
            If=defines.is_defined["GELUFast"](),
            Then=GELUFast,
            Else=ConditionalType[
                Trait=ActivationFunction,
                If=defines.is_defined["Sigmoid"](),
                Then=Sigmoid,
                Else=ConditionalType[
                    Trait=ActivationFunction,
                    If=defines.is_defined["Tanh"](),
                    Then=Tanh,
                    Else=ReLU,
                ],
            ],
        ],
    ],
]  # options: ReLU, GELU, GELUFast, GELUTanh, Sigmoid

# Compile-time allocator selection (mirrors act_fn). Bump arena by default; the
# system allocator is a benchmarking baseline. -D CPU_SYSTEM_ALLOC / -D GPU_SYSTEM_ALLOC.
comptime CPU_ALLOCATOR = ConditionalType[
    Trait=CPUAllocator,
    If=defines.is_defined["CPU_SYSTEM_ALLOC"](),
    Then=CPUSystemAllocator,
    Else=CPUBumpArenaAllocator,
]
comptime GPU_ALLOCATOR = ConditionalType[
    Trait=GPUAllocator,
    If=defines.is_defined["GPU_SYSTEM_ALLOC"](),
    Then=GPUSystemAllocator,
    Else=GPUBumpArenaAllocator,
]

comptime DISPLAY = True if defines.is_defined["DISPLAY"]() else False


# Shared shape definitions — one source of truth for CPU + GPU. LeNet5 has the
# same feature/weight/bias shapes on both paths, so both the CPU (`Feature`,
# `LeNet5`) and GPU (`FeatureGPU`, `LeNet5GPU`) structs point at these instead of
# re-deriving `Layout.row_major(...)` inline. Access as `FeatureLayouts.layer4`,
# `WeightLayouts.w45`, `BiasLayouts.b45` (the struct prefix reads well, so no
# `_layout` suffix).
struct FeatureLayouts:
    comptime input = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    comptime layer1 = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    comptime layer2 = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    comptime layer3 = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    comptime layer4 = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    comptime layer5 = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    comptime output = Layout.row_major(OUTPUT)


struct WeightLayouts:
    comptime w01 = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    comptime w23 = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    comptime w45 = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    comptime w56 = Layout.row_major(
        LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT
    )


struct BiasLayouts:
    comptime b01 = Layout.row_major(LAYER1)
    comptime b23 = Layout.row_major(LAYER3)
    comptime b45 = Layout.row_major(LAYER5)
    comptime b56 = Layout.row_major(OUTPUT)
