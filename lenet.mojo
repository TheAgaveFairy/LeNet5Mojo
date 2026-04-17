from layout import Layout, LayoutTensor
from std.math import sqrt, exp, log
from std.random import random_float64
from std.sys.info import sizeof
from std.sys import stderr, is_big_endian
from std.utils.index import IndexList
import std.os as os
from std.memory import memcpy
from std.time import perf_counter_ns
from std.pathlib import Path

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier

from image import Image
from resultlogger import MultiFileLogger, LeNet5Logger
from helpers import showProgress, reLu, reLuGrad
from arena import BumpArenaAllocator as Arena

#struct ModelParams():
comptime LENGTH_KERNEL = 5
comptime LENGTH_KERNEL_SQ = LENGTH_KERNEL * LENGTH_KERNEL

comptime LENGTH_FEATURE0 = 32
comptime LENGTH_FEATURE1 = (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
comptime LENGTH_FEATURE2 = (LENGTH_FEATURE1 >> 1)
comptime LENGTH_FEATURE3 = (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
comptime LENGTH_FEATURE4 = (LENGTH_FEATURE3 >> 1)
comptime LENGTH_FEATURE5 = (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

comptime INPUT  =  1
comptime LAYER1 =  6
comptime LAYER2 =  LAYER1
comptime LAYER3 =  16
comptime LAYER4 =  LAYER3
comptime LAYER5 =  120
comptime OUTPUT =  10

comptime NUM_WEIGHTS =     51902 # can be calculated but we're just hardcoding for some easier checks at load/save

comptime ALPHA = 0.5
comptime PADDING = 2

comptime IMAGE_SIZE =      28 # as we read it in from the file, its padded to LENGTH_FEATURE0 (a.k.a. PADDED_SIZE)
comptime PADDED_SIZE = IMAGE_SIZE + 2 * PADDING # 32 x 32 is what we want eventually # this should equal LENGTH_FEATURE0

# try changing this to float64 or bf16, etc
comptime ftype = DType.float32 # model's internal float type
comptime sftype = Scalar[ftype]


struct LeNet5(Copyable):
    """
    The LeNet5 model. In the actual LeCun et al implementation, there is some
    notable sparsity in final layers that is not in this version, as well as
    another linear layer of size 84 just before output.

    Unlike my previous C project, these layers are all on the heap instead of
    the stack.
    """
    var arena: Arena

    # WEIGHTS
    comptime w01_layout = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight0_1: LayoutTensor[mut = True, ftype, Self.w01_layout, MutableAnyOrigin]
    
    comptime w23_layout = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight2_3: LayoutTensor[mut = True, ftype, Self.w23_layout, MutableAnyOrigin]
    
    comptime w45_layout = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight4_5: LayoutTensor[mut = True, ftype, Self.w45_layout, MutableAnyOrigin]
    
    comptime w56_layout = Layout.row_major(LAYER5 * LENGTH_FEATURE5 *  LENGTH_FEATURE5, OUTPUT)
    var weight5_6: LayoutTensor[mut = True, ftype, Self.w56_layout, MutableAnyOrigin]

    # BIASES
    comptime b01_layout = Layout.row_major(LAYER1)
    var bias0_1: LayoutTensor[mut = True, ftype, Self.b01_layout, MutableAnyOrigin]
    
    comptime b23_layout = Layout.row_major(LAYER3)
    var bias2_3: LayoutTensor[mut = True, ftype, Self.b23_layout, MutableAnyOrigin]
    
    comptime b45_layout = Layout.row_major(LAYER5)
    var bias4_5: LayoutTensor[mut = True, ftype, Self.b45_layout, MutableAnyOrigin]
    
    comptime b56_layout = Layout.row_major(OUTPUT)
    var bias5_6: LayoutTensor[mut = True, ftype, Self.b56_layout, MutableAnyOrigin]

    def _tensorHelper[layout: Layout](mut self) -> LayoutTensor[ftype, layout, MutAnyOrigin]:
        return LayoutTensor[ftype, layout, MutAnyOrigin](self.arena.alloc[ftype](layout.size())).fill(0.0)
        
    @staticmethod
    def _calcArenaSize() -> Int:
        var weights = w01_layout.size() + w23_layout.size() + w45_layout.size() + w56_layout.size()
        var biases =  b01_layout.size() + b23_layout.size() + b45_layout.size() + b56_layout.size()
        return (weights + biases) * size_of[ftype]()

    def __init__(out self):
        """
        Initialize to all zeros, for training you'll want to randomizeWeights(),
        or for inference, read in from a file. Only biases really need to be set
        to zeroes.
        """
        var num_bytes = Self._calcArenaSize()
        self.arena = Arena(num_bytes)

        # weights
        self.weight0_1 = self._tensorHelper[w01_layout]()
        self.weight2_3 = self._tensorHelper[w23_layout]()
        self.weight4_5 = self._tensorHelper[w45_layout]()
        self.weight5_6 = self._tensorHelper[w56_layout]()
        # biases
        self.bias0_1 = self._tensorHelper[b01_layout]()
        self.bias2_3 = self._tensorHelper[b23_layout]()
        self.bias4_5 = self._tensorHelper[b45_layout]()
        self.bias5_6 = self._tensorHelper[b56_layout]()

    def __init__(out self, *, copy: Self):
        self.arena = copy.arena # shallow copy
        self.weight0_1 = copy.weight0_1
        self.weight2_3 = copy.weight2_3
        self.weight4_5 = copy.weight4_5
        self.weight5_6 = copy.weight5_6
        self.bias0_1 = copy.bias0_1
        self.bias2_3 = copy.bias2_3
        self.bias4_5 = copy.bias4_5
        self.bias5_6 = copy.bias5_6

    def __del__(deinit self):
        self.arena.clear()

    def accumulateFromOther(mut self, other: Self, lr: Scalar[ftype]):
        """
        For taking in errors / deltas during backward pass with learning rate.
        The 'other.layer * lr' doesn't compile, though it seems it should.
        """
        self.weight0_1 += other.weight0_1 * lr
        self.weight2_3 += other.weight2_3 * lr 
        self.weight4_5 += other.weight4_5 * lr
        self.weight5_6 += other.weight5_6 * lr

        self.bias0_1 += other.bias0_1 * lr
        self.bias2_3 += other.bias2_3 * lr 
        self.bias4_5 += other.bias4_5 * lr 
        self.bias5_6 += other.bias5_6 * lr


    @staticmethod
    def _randHelper[layout: Layout](tensor: LayoutTensor[ftype, layout, MutAnyOrigin], scale: sftype):
        var data = Span(tensor.ptr, tensor.layout.size())
        rand(data, -1.0, 1.0) # uniform distribution
        # FIXME: compile times might slow down from these LayoutTensor math ops
        tensor *= sqrt(6.0) / scale # from the paper

    def randomizeWeights(self):
        """
        For initializing for training. Biases stay at zeros.
        There might be a better way to do this (SIMD, flatten and @parameter).
        """
        Self._randHelper(self.weight0_1, LENGTH_KERNEL_SQ * (INPUT + LAYER1))
        Self._randHelper(self.weight2_3, LENGTH_KERNEL_SQ * (LAYER2 + LAYER3))
        Self._randHelper(self.weight4_5, LENGTH_KERNEL_SQ * (LAYER4 + LAYER5))
        Self._randHelper(self.weight5_6, (LAYER5 + OUTPUT))

    @staticmethod
    def _bytesHelper[filetype: DType](buffer: InlineArray[UInt8, filetype.sizeof()]) -> Scalar[filetype]:
        """
        Filetype might be able to be a runtime parameter instead.
        The "from_bytes()" method stopped working for me after an update, so
        the source code was copy / pasted here to get it working.
        Takes in some "bytes" and casts them to the desired type.
        """
        comptime f_sz = filetype.sizeof()
        var result: Scalar[filetype]

        @parameter
        if is_big_endian():
            var reversed = __type_of(buffer)(fill = 0)#(uninitialized = True)
            for b in range(f_sz):
                reversed[b] = buffer[f_sz - 1 - b] 
            result = reversed.unsafe_ptr().bitcast[Scalar[filetype]]()[]
        else:
            result = buffer.unsafe_ptr().bitcast[Scalar[filetype]]()[]

        return result

    @staticmethod
    def bytesToFType[filetype: DType, num_bytes: Int, layout: Layout]
        (bytes: InlineArray[Scalar[DType.uint8], num_bytes],
         tensor: LayoutTensor[ftype, layout, MutAnyOrigin]) -> None:
        """
        Helper function that takes in an array of bytes from a "model.dat" file
        and converts them to the correct datatype and fills the associated layer.
        """
        
        comptime f_sz = filetype.sizeof() # 4 bytes for Float32, 8 for F64, etc
        comptime num_elems = num_bytes // f_sz

        if num_elems != tensor.size():
            print("FATAL ERROR CONVERTING BYTES TO TENSOR") # TODO: better error os.abort
            print("num_elems, tensor.size(), len(bytes):", num_elems, tensor.size(), len(bytes))

        # start looking here at this commented out block
        _ = """
        elif layout.rank() == 4: # removed cases where rank < 4
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var remainder = idx % (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var j = remainder // (tensor.shape[2]() * tensor.shape[3]())
                remainder = remainder % (tensor.shape[2]() * tensor.shape[3]())
                var k = remainder // tensor.shape[3]()
                var l = remainder % tensor.shape[3]()

                var buffer = InlineArray[Scalar[DType.uint8], f_sz](fill = 0)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx]
                var value = Self._bytesHelper[filetype](buffer)
                tensor[i,j,k,l] = value.cast[ftype]()
        """

        # FIXME: comptime unrolling might slow compilation for a large LayoutTensor
        comptime for i in range(tensor.layout.size()):
            var buffer = InlineArray[Byte, f_sz](fill = 0)
            comptime for bi in range(f_sz):
                var temp_idx = i * f_sz + bi
                buffer[bi] = bytes[temp_idx]
            #var value = Self._bytesHelper[filetype](buffer)
            var value = filetype.from_bytes(buffer)
            tensor.ptr[i] = ftype(value) # ftype might not match file precision

    @staticmethod
    def fromFile[filetype: DType](filename: Path) -> Self:
        """
        Reads in a "model.dat" file and loads it into a Self.
        Note: Closures can't have parameters, yet.
        """
        comptime bytes_per_file_weight = filetype.sizeof()# sizeof[filetype]() won't work, must use filetype.sizeof()

        var model = LeNet5()

        try:
            with open(filename, "r") as model_file:

                def helper[layout: Layout](weights: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]):
                    comptime size_of_layer = layout.size()
                    comptime bytes_to_read = size_of_layer * bytes_per_file_weight
                    var bytes: List[UInt8]
                    try:
                        bytes = model_file.read_bytes(bytes_to_read)
                    except ee:
                        print("helper fromFile", ee)
                    var buffer = InlineArray[Scalar[DType.uint8], bytes_to_read](uninitialized = True)
                    for i in range(bytes_to_read):
                        buffer[i] = bytes[i] # memcpy
                    Self.bytesToFType[filetype, bytes_to_read, layout](buffer, weights)

                helper(model.weight0_1) 
                helper(model.weight2_3) 
                helper(model.weight4_5) 
                helper(model.weight5_6) 
                helper(model.bias0_1) 
                helper(model.bias2_3) 
                helper(model.bias4_5) 
                helper(model.bias5_6) 

        except e:
            print("error at reading lenet5 from file", e)
        finally:
            return model

struct Feature():
    """
    These buffers hold intermediate results. Wish it could easily be on the stack
    instead of heap, not sure about how to make that happen.
    """
    var arena: Arena

    comptime input_layout = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    var input: LayoutTensor[mut = True, ftype, Feature.input_layout, MutableAnyOrigin]

    comptime layer1_layout = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    var layer1: LayoutTensor[mut = True, ftype, Feature.layer1_layout, MutableAnyOrigin]

    comptime layer2_layout = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    var layer2: LayoutTensor[mut = True, ftype, Feature.layer2_layout, MutableAnyOrigin]

    comptime layer3_layout = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    var layer3: LayoutTensor[mut = True, ftype, Feature.layer3_layout, MutableAnyOrigin]
    
    comptime layer4_layout = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    var layer4: LayoutTensor[mut = True, ftype, Feature.layer4_layout, MutableAnyOrigin]
    
    comptime layer5_layout = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    var layer5: LayoutTensor[mut = True, ftype, Feature.layer5_layout, MutableAnyOrigin]
    
    comptime output_layout = Layout.row_major(OUTPUT)
    var output: LayoutTensor[mut = True, ftype, Feature.output_layout, MutableAnyOrigin]

    def _tensorHelper[layout: Layout](mut self) -> LayoutTensor[ftype, layout, MutAnyOrigin]:
        return LayoutTensor[ftype, layout, MutAnyOrigin](self.arena.alloc[ftype](layout.size())).fill(0.0)
    
    @staticmethod
    def _calcArenaSize() -> Int:
        var n = input_layout.size() + layer1_layout.size() + layer2_layout.size() + layer3_layout.size() + layer4_layout.size() + layer5_layout.size() + output_layout.size()
        return n * size_of[ftype]()
    
    def __init__(out self):
        """
        Needs to start as all zeros.
        """
        self.arena = Arena(Self._calcArenaSize())

        self.input = Self._tensorHelper[input_layout]()
        self.layer1 = Self._tensorHelper[layer1_layout]()
        self.layer2 = Self._tensorHelper[layer2_layout]()
        self.layer3 = Self._tensorHelper[layer3_layout]()
        self.layer4 = Self._tensorHelper[layer4_layout]()
        self.layer5 = Self._tensorHelper[layer5_layout]()
        self.output = Self._tensorHelper[output_layout]()

    def __del__(deinit self):
        print("deleting Feature()")
        self.arena.clear()

