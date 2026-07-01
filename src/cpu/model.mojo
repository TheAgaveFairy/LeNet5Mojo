from layout import Layout, LayoutTensor
from std.math import sqrt, exp, log
from std.random import random_float64, rand
from std.sys import stderr, is_big_endian, size_of, simd_width_of
from std.utils.index import IndexList
import std.os as os
from std.memory import memcpy
from std.pathlib import Path
from std.algorithm.functional import vectorize

from image import Image
from origin_util import untrack
from resultlogger import MultiFileLogger, LeNet5Logger
from cpu.ops import (
    convoluteForward,
    maxPoolForward,
    matmulForward,
    matmulBackward,
    convoluteBackward,
    maxPoolBackward,
    argMax,
)
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as CPUArena, ArenaSizable
from activation_fn import ActivationFunction
from constants import (
    ftype,
    sftype,
    nelts,
    act_fn,
    LENGTH_KERNEL,
    LENGTH_KERNEL_SQ,
    LENGTH_FEATURE0,
    LENGTH_FEATURE1,
    LENGTH_FEATURE2,
    LENGTH_FEATURE3,
    LENGTH_FEATURE4,
    LENGTH_FEATURE5,
    INPUT,
    LAYER1,
    LAYER2,
    LAYER3,
    LAYER4,
    LAYER5,
    OUTPUT,
    NUM_WEIGHTS,
    ALPHA,
    PADDING,
    IMAGE_SIZE,
    PADDED_SIZE,
    FeatureLayouts,
    WeightLayouts,
    BiasLayouts,
)


struct LeNet5(Movable, ArenaSizable):
    """
    The LeNet5 model. In the actual LeCun et al implementation, there is some
    notable sparsity in final layers that is not in this version, as well as
    another linear layer of size 84 just before output.

    Unlike my previous C project, these layers are all on the heap instead of
    the stack.
    """

    # var arena: Self.Allocator  # might not actually be an 'arena' per se, but that's the default
    var allocator_owns_memory: Bool

    # WEIGHTS
    var weight0_1: LayoutTensor[ftype, WeightLayouts.w01, MutUntrackedOrigin]
    var weight2_3: LayoutTensor[ftype, WeightLayouts.w23, MutUntrackedOrigin]
    var weight4_5: LayoutTensor[ftype, WeightLayouts.w45, MutUntrackedOrigin]
    var weight5_6: LayoutTensor[ftype, WeightLayouts.w56, MutUntrackedOrigin]

    # BIASES
    var bias0_1: LayoutTensor[ftype, BiasLayouts.b01, MutUntrackedOrigin]
    var bias2_3: LayoutTensor[ftype, BiasLayouts.b23, MutUntrackedOrigin]
    var bias4_5: LayoutTensor[ftype, BiasLayouts.b45, MutUntrackedOrigin]
    var bias5_6: LayoutTensor[ftype, BiasLayouts.b56, MutUntrackedOrigin]

    @staticmethod
    def sizeInBytes() -> Int:
        var weights = comptime (
            WeightLayouts.w01.size()
            + WeightLayouts.w23.size()
            + WeightLayouts.w45.size()
            + WeightLayouts.w56.size()
        )
        var biases = comptime (
            BiasLayouts.b01.size()
            + BiasLayouts.b23.size()
            + BiasLayouts.b45.size()
            + BiasLayouts.b56.size()
        )
        return (weights + biases) * size_of[ftype]()

    def __init__(out self):
        self.allocator_owns_memory = False
        # weights
        self.weight0_1 = untrack(LayoutTensor[ftype, WeightLayouts.w01](
            alloc[sftype](comptime (WeightLayouts.w01.size()))
        )).fill(0.0)
        self.weight2_3 = untrack(LayoutTensor[ftype, WeightLayouts.w23](
            alloc[sftype](comptime (WeightLayouts.w23.size()))
        )).fill(0.0)
        self.weight4_5 = untrack(LayoutTensor[ftype, WeightLayouts.w45](
            alloc[sftype](comptime (WeightLayouts.w45.size()))
        )).fill(0.0)
        self.weight5_6 = untrack(LayoutTensor[ftype, WeightLayouts.w56](
            alloc[sftype](comptime (WeightLayouts.w56.size()))
        )).fill(0.0)
        # biases
        self.bias0_1 = untrack(LayoutTensor[ftype, BiasLayouts.b01](
            alloc[sftype](comptime (BiasLayouts.b01.size()))
        )).fill(0.0)
        self.bias2_3 = untrack(LayoutTensor[ftype, BiasLayouts.b23](
            alloc[sftype](comptime (BiasLayouts.b23.size()))
        )).fill(0.0)
        self.bias4_5 = untrack(LayoutTensor[ftype, BiasLayouts.b45](
            alloc[sftype](comptime (BiasLayouts.b45.size()))
        )).fill(0.0)
        self.bias5_6 = untrack(LayoutTensor[ftype, BiasLayouts.b56](
            alloc[sftype](comptime (BiasLayouts.b56.size()))
        )).fill(0.0)

    def __init__(out self, mut arena: Some[CPUAllocator]):  # raises
        """
        Initialize to all zeros, for training you'll want to randomizeWeights(),
        or for inference, read in from a file. Only biases really need to be set
        to zeroes.
        """
        self.allocator_owns_memory = True
        # weights
        self.weight0_1 = untrack(LayoutTensor[ftype, WeightLayouts.w01](
            arena.alloc[sftype](comptime (WeightLayouts.w01.size()))
        )).fill(0.0)
        self.weight2_3 = untrack(LayoutTensor[ftype, WeightLayouts.w23](
            arena.alloc[sftype](comptime (WeightLayouts.w23.size()))
        )).fill(0.0)
        self.weight4_5 = untrack(LayoutTensor[ftype, WeightLayouts.w45](
            arena.alloc[sftype](comptime (WeightLayouts.w45.size()))
        )).fill(0.0)
        self.weight5_6 = untrack(LayoutTensor[ftype, WeightLayouts.w56](
            arena.alloc[sftype](comptime (WeightLayouts.w56.size()))
        )).fill(0.0)
        # biases
        self.bias0_1 = untrack(LayoutTensor[ftype, BiasLayouts.b01](
            arena.alloc[sftype](comptime (BiasLayouts.b01.size()))
        )).fill(0.0)
        self.bias2_3 = untrack(LayoutTensor[ftype, BiasLayouts.b23](
            arena.alloc[sftype](comptime (BiasLayouts.b23.size()))
        )).fill(0.0)
        self.bias4_5 = untrack(LayoutTensor[ftype, BiasLayouts.b45](
            arena.alloc[sftype](comptime (BiasLayouts.b45.size()))
        )).fill(0.0)
        self.bias5_6 = untrack(LayoutTensor[ftype, BiasLayouts.b56](
            arena.alloc[sftype](comptime (BiasLayouts.b56.size()))
        )).fill(0.0)

    def zero(mut self):
        _ = self.weight0_1.fill(0.0)
        _ = self.weight2_3.fill(0.0)
        _ = self.weight4_5.fill(0.0)
        _ = self.weight5_6.fill(0.0)
        _ = self.bias0_1.fill(0.0)
        _ = self.bias2_3.fill(0.0)
        _ = self.bias4_5.fill(0.0)
        _ = self.bias5_6.fill(0.0)

    def __init__(out self, *, deinit existing: Self):
        print("model move")
        self.allocator_owns_memory = existing.allocator_owns_memory
        self.weight0_1 = existing.weight0_1
        self.weight2_3 = existing.weight2_3
        self.weight4_5 = existing.weight4_5
        self.weight5_6 = existing.weight5_6
        self.bias0_1 = existing.bias0_1
        self.bias2_3 = existing.bias2_3
        self.bias4_5 = existing.bias4_5
        self.bias5_6 = existing.bias5_6

    def __del__(deinit self):
        if not self.allocator_owns_memory:
            self.weight0_1.ptr.free()
            self.weight2_3.ptr.free()
            self.weight4_5.ptr.free()
            self.weight5_6.ptr.free()
            self.bias0_1.ptr.free()
            self.bias2_3.ptr.free()
            self.bias4_5.ptr.free()
            self.bias5_6.ptr.free()

    @staticmethod
    def _accumHelper[
        x: Layout
    ](
        accum: LayoutTensor[ftype, x, MutAnyOrigin],
        other: LayoutTensor[ftype, x, _],
        lr: sftype,
    ):
        comptime N = x.size()
        _ = """
        # this is the "simple" way to do this
        var a = accum.ptr
        var b = other.ptr
        for i in range(N):
            a[i] += (b[i] * lr)
        """

        def vectorize_closure[width: Int](i: Int) {read}:
            var lrs = SIMD[ftype, width](lr)
            var a_nums = accum.ptr.load[width=width](i)
            var b_nums = other.ptr.load[width=width](i)
            var result = a_nums + b_nums * lrs
            accum.ptr.store[width=width](i, result)

        vectorize[nelts](comptime (N), vectorize_closure)

    def accumulateFromOther(
        mut self, other: Self, lr: sftype
    ):  # TODO: needs compiler / stdlib fix
        """
        For taking in errors / deltas during backward pass with learning rate.
        self.weight0_1 += other.weight0_1 * lr # EXPLODES COMPILE TIMES
        """
        Self._accumHelper(self.weight0_1, other.weight0_1, lr)
        Self._accumHelper(self.weight2_3, other.weight2_3, lr)
        Self._accumHelper(self.weight4_5, other.weight4_5, lr)
        Self._accumHelper(self.weight5_6, other.weight5_6, lr)

        Self._accumHelper(self.bias0_1, other.bias0_1, lr)
        Self._accumHelper(self.bias2_3, other.bias2_3, lr)
        Self._accumHelper(self.bias4_5, other.bias4_5, lr)
        Self._accumHelper(self.bias5_6, other.bias5_6, lr)

    @staticmethod
    def _randHelper[
        layout: Layout, o: MutOrigin
    ](mut tensor: LayoutTensor[ftype, layout, o], scale: sftype):
        comptime N = tensor.layout.size()
        var data = Span(ptr=tensor.ptr, length=comptime (N))
        rand(data, min=-1.0, max=1.0)  # uniform distribution
        # tensor *= sftype(sqrt(6.0)) / scale  # from the paper # compiler is slow for this
        # naive could look like this, but we want SIMD speedup
        _ = """
        for i in range(comptime(tensor.layout.size())):
            tensor.ptr[i] *= sftype(sqrt(6.0)) / scale
        """

        def vectorize_closure[width: Int](i: Int) {read}:
            comptime sixes = SIMD[ftype, width](6.0)
            var scales = SIMD[ftype, width](scale)
            var nums = tensor.ptr.load[width=width](i)
            var result = nums * sqrt(sixes / scales)
            tensor.ptr.store[width=width](i, result)

        vectorize[nelts](comptime (N), vectorize_closure)

    def randomizeWeights(mut self):
        """
        For initializing for training. Biases stay at zeros.
        There might be a better way to do this (SIMD, flatten and @parameter).
        """
        Self._randHelper(self.weight0_1, LENGTH_KERNEL_SQ * (INPUT + LAYER1))
        Self._randHelper(self.weight2_3, LENGTH_KERNEL_SQ * (LAYER2 + LAYER3))
        Self._randHelper(self.weight4_5, LENGTH_KERNEL_SQ * (LAYER4 + LAYER5))
        Self._randHelper(self.weight5_6, (LAYER5 + OUTPUT))

    def forward(self, features: Feature):
        convoluteForward(
            self.weight0_1, self.bias0_1, features.input, features.layer1
        )
        maxPoolForward(features.layer1, features.layer2)
        convoluteForward(
            self.weight2_3, self.bias2_3, features.layer2, features.layer3
        )
        maxPoolForward(features.layer3, features.layer4)
        convoluteForward(
            self.weight4_5, self.bias4_5, features.layer4, features.layer5
        )
        matmulForward(
            features.layer5, features.output, self.weight5_6, self.bias5_6
        )

    def backward(self, deltas: LeNet5, errors: Feature, features: Feature):
        matmulBackward(
            features.layer5,
            errors.layer5,
            errors.output,
            self.weight5_6,
            deltas.weight5_6,
            deltas.bias5_6,
        )
        convoluteBackward[kernel_size=LENGTH_KERNEL](
            features.layer4,
            errors.layer4,
            errors.layer5,
            self.weight4_5,
            deltas.weight4_5,
            deltas.bias4_5,
        )
        maxPoolBackward(features.layer3, errors.layer3, errors.layer4)
        convoluteBackward[kernel_size=LENGTH_KERNEL](
            features.layer2,
            errors.layer2,
            errors.layer3,
            self.weight2_3,
            deltas.weight2_3,
            deltas.bias2_3,
        )
        maxPoolBackward(features.layer1, errors.layer1, errors.layer2)
        convoluteBackward[kernel_size=LENGTH_KERNEL](
            features.input,
            errors.input,
            errors.layer1,
            self.weight0_1,
            deltas.weight0_1,
            deltas.bias0_1,
        )

    # TODO: make feat explicit Optional[Feature] and combine these two
    def predict(self, image: Image) -> Int:
        var feat_arena = CPUArena(Feature.sizeInBytes())
        var feat = Feature(feat_arena)
        feat.loadInput(image)
        self.forward(feat)
        return argMax(feat.output)

    def predict(self, feat: Feature, image: Image) -> Int:
        feat.loadInput(image)
        self.forward(feat)
        return argMax(feat.output)

    @staticmethod
    def bytesToFType[
        filetype: DType,
        num_bytes: Int,
        layout: Layout,
        big_e: Bool = is_big_endian(),
    ](
        bytes: InlineArray[Scalar[DType.uint8], num_bytes],
        tensor: LayoutTensor[ftype, layout, MutAnyOrigin],
    ) -> None:
        """
        Helper function that takes in an array of bytes from a "model.dat" file
        and converts them to the correct datatype and fills the associated layer.
        """

        comptime f_sz = size_of[
            filetype
        ]()  # 4 bytes for Float32, 8 for F64, etc
        comptime num_elems = num_bytes // f_sz

        comptime assert (
            num_elems == tensor.layout.size()
        ), "FATAL ERROR CONVERTING BYTES TO TENSOR"

        for i in range(comptime (tensor.layout.size())):
            var buffer = InlineArray[Byte, size_of[Scalar[filetype]]()](fill=0)
            comptime for bi in range(f_sz):
                var temp_idx = i * f_sz + bi
                buffer[bi] = bytes[temp_idx]
            # var value = Self._bytesHelper[filetype](buffer)
            # FIXME: big_endian flag was having compiler issues, investigate or file bug etc
            var value = Scalar[filetype].from_bytes(buffer)
            tensor.ptr[i] = sftype(
                value
            )  # ftype might not match file precision, that's a feature! we can convert!

    def loadFromFile[filetype: DType](mut self, filename: Path):
        """
        Reads in a "model.dat" file and loads it into self.
        The 'filetype' parameter is designating the floating point type of the
        saved binary file. This doesn't need to match constants.ftype.
        """
        comptime bytes_per_file_weight = size_of[
            filetype
        ]()

        try:
            with open(filename, "r") as model_file:

                def helper[
                    layout: Layout
                ](weights: LayoutTensor[ftype, layout, MutAnyOrigin]) {mut}:
                    comptime size_of_layer = layout.size()
                    comptime bytes_to_read = size_of_layer * bytes_per_file_weight
                    var bytes: List[UInt8]
                    try:
                        bytes = model_file.read_bytes(bytes_to_read)
                    except ee:
                        print("helper fromFile", ee)
                        bytes = type_of(bytes)()
                    var buffer = InlineArray[
                        Scalar[DType.uint8], bytes_to_read
                    ](uninitialized=True)
                    # TODO: get rid of extra copy, etc
                    # for i in range(bytes_to_read):
                    #    buffer[i] = bytes[i]  # memcpy
                    memcpy(
                        src=bytes.unsafe_ptr(),
                        dest=buffer.unsafe_ptr(),
                        count=bytes_to_read,
                    )
                    Self.bytesToFType[filetype, bytes_to_read, layout](
                        buffer, weights
                    )

                helper(self.weight0_1)
                helper(self.weight2_3)
                helper(self.weight4_5)
                helper(self.weight5_6)
                helper(self.bias0_1)
                helper(self.bias2_3)
                helper(self.bias4_5)
                helper(self.bias5_6)

        except e:
            print("error at reading lenet5 from file", e)

    @staticmethod
    def _writeTensor[
        layout: Layout
    ](
        tensor: LayoutTensor[ftype, layout, MutAnyOrigin], mut f: FileHandle
    ) raises:
        comptime fbs = size_of[ftype]()  # float byte size
        var ptr_bytes = tensor.ptr.bitcast[UInt8]()
        var ptr_len = comptime (layout.size()) * fbs
        var temp_buf = alloc[UInt8](ptr_len)
        var bytes_span = Span(ptr=temp_buf, length=ptr_len)
        memcpy(src=ptr_bytes, dest=temp_buf, count=ptr_len)
        comptime if is_big_endian(): # TODO: maybe make this a function parameter / arg check
            for i in range(0, ptr_len, fbs):
                comptime for j in range(fbs // 2):
                    swap(temp_buf[i + j], temp_buf[i + (fbs - 1) - j])
        f.write_all(bytes_span)

    def saveToFile(mut self, filename: Path) raises:
        """Alternatively, could write an Arena to file, etc."""
        with open(filename, "w") as f:
            Self._writeTensor(self.weight0_1, f)
            Self._writeTensor(self.weight2_3, f)
            Self._writeTensor(self.weight4_5, f)
            Self._writeTensor(self.weight5_6, f)
            Self._writeTensor(self.bias0_1, f)
            Self._writeTensor(self.bias2_3, f)
            Self._writeTensor(self.bias4_5, f)
            Self._writeTensor(self.bias5_6, f)


struct Feature(Movable, ArenaSizable):
    """
    These buffers hold intermediate results.
    """

    var input: LayoutTensor[ftype, FeatureLayouts.input, MutUntrackedOrigin]
    var layer1: LayoutTensor[ftype, FeatureLayouts.layer1, MutUntrackedOrigin]
    var layer2: LayoutTensor[ftype, FeatureLayouts.layer2, MutUntrackedOrigin]
    var layer3: LayoutTensor[ftype, FeatureLayouts.layer3, MutUntrackedOrigin]
    var layer4: LayoutTensor[ftype, FeatureLayouts.layer4, MutUntrackedOrigin]
    var layer5: LayoutTensor[ftype, FeatureLayouts.layer5, MutUntrackedOrigin]
    var output: LayoutTensor[ftype, FeatureLayouts.output, MutUntrackedOrigin]

    @staticmethod
    def sizeInBytes() -> Int:
        var n = comptime (
            FeatureLayouts.input.size()
            + FeatureLayouts.layer1.size()
            + FeatureLayouts.layer2.size()
            + FeatureLayouts.layer3.size()
            + FeatureLayouts.layer4.size()
            + FeatureLayouts.layer5.size()
            + FeatureLayouts.output.size()
        )
        return n * size_of[ftype]()

    def __init__(out self):
        """
        Needs to start as all zeros.
        """
        self.input = untrack(LayoutTensor[ftype, FeatureLayouts.input](
            alloc[sftype](comptime (FeatureLayouts.input.size()))
        )).fill(0.0)
        self.layer1 = untrack(LayoutTensor[ftype, FeatureLayouts.layer1](
            alloc[sftype](comptime (FeatureLayouts.layer1.size()))
        )).fill(0.0)
        self.layer2 = untrack(LayoutTensor[ftype, FeatureLayouts.layer2](
            alloc[sftype](comptime (FeatureLayouts.layer2.size()))
        )).fill(0.0)
        self.layer3 = untrack(LayoutTensor[ftype, FeatureLayouts.layer3](
            alloc[sftype](comptime (FeatureLayouts.layer3.size()))
        )).fill(0.0)
        self.layer4 = untrack(LayoutTensor[ftype, FeatureLayouts.layer4](
            alloc[sftype](comptime (FeatureLayouts.layer4.size()))
        )).fill(0.0)
        self.layer5 = untrack(LayoutTensor[ftype, FeatureLayouts.layer5](
            alloc[sftype](comptime (FeatureLayouts.layer5.size()))
        )).fill(0.0)
        self.output = untrack(LayoutTensor[ftype, FeatureLayouts.output](
            alloc[sftype](comptime (FeatureLayouts.output.size()))
        )).fill(0.0)

    def __init__(out self, mut arena: Some[CPUAllocator]):
        """
        Needs to start as all zeros.
        """
        self.input = untrack(LayoutTensor[ftype, FeatureLayouts.input](
            arena.alloc[sftype](comptime (FeatureLayouts.input.size()))
        )).fill(0.0)
        self.layer1 = untrack(LayoutTensor[ftype, FeatureLayouts.layer1](
            arena.alloc[sftype](comptime (FeatureLayouts.layer1.size()))
        )).fill(0.0)
        self.layer2 = untrack(LayoutTensor[ftype, FeatureLayouts.layer2](
            arena.alloc[sftype](comptime (FeatureLayouts.layer2.size()))
        )).fill(0.0)
        self.layer3 = untrack(LayoutTensor[ftype, FeatureLayouts.layer3](
            arena.alloc[sftype](comptime (FeatureLayouts.layer3.size()))
        )).fill(0.0)
        self.layer4 = untrack(LayoutTensor[ftype, FeatureLayouts.layer4](
            arena.alloc[sftype](comptime (FeatureLayouts.layer4.size()))
        )).fill(0.0)
        self.layer5 = untrack(LayoutTensor[ftype, FeatureLayouts.layer5](
            arena.alloc[sftype](comptime (FeatureLayouts.layer5.size()))
        )).fill(0.0)
        self.output = untrack(LayoutTensor[ftype, FeatureLayouts.output](
            arena.alloc[sftype](comptime (FeatureLayouts.output.size()))
        )).fill(0.0)

    def loadInput(self, image: Image):
        var normed_tensor = untrack(
            LayoutTensor[ftype, Image.DataLayout](self.input.ptr)
        )
        image.normalized(normed_tensor)


struct CPUSession(): # TODO: maybe offer other constructors for other allocators
    """Ties arena and model lifetimes together — mirrors DeviceSession."""

    var arena: CPUArena
    var model: LeNet5

    def __init__(out self):
        self.arena = CPUArena(LeNet5.sizeInBytes())
        self.model = LeNet5(self.arena)
