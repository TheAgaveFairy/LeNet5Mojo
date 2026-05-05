from std.sys import simd_width_of
import std.sys.defines as defines

from activation_fn import *


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
comptime LAYER3 = 16
comptime LAYER4 = LAYER3
comptime LAYER5 = 120
comptime OUTPUT = 10

comptime NUM_WEIGHTS = 51902 # hardcoding here for simplicity, can be calculated

comptime ALPHA = Float32(defines.get_defined_int["ALPHA", 500]()) / 1000
comptime PADDING = 2

comptime IMAGE_SIZE = 28
comptime PADDED_SIZE = IMAGE_SIZE + 2 * PADDING  # == LENGTH_FEATURE0

# Numeric type — change 'ftype' here to switch the whole model (float64, bf16, etc.)
comptime ftype = DType.float64 #defines.get_defined_dtype["ftype", DType.float32]() # doesn't want to work
comptime sftype = Scalar[ftype]
comptime nelts = simd_width_of[ftype]()

comptime act_fn: ActivationFunction = GELUFast # options: ReLU, GELU, GELUFast, GELUTanh
