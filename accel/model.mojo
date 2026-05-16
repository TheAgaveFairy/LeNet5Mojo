from layout import Layout, LayoutTensor

from std.gpu.host import DeviceContext, DeviceBuffer
from std.builtin.device_passable import DevicePassable
from std.reflection.reflect import reflect
from std.sys import size_of

from cpu.model import LeNet5
from accel.arena import GPUAllocator, GPUBumpArenaAllocator, GPUSystemAllocator 
from constants import (
    ftype,
    sftype,
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
    PADDED_SIZE,
)
from image import Image

struct DeviceSession[Allocator: GPUAllocator]():
    """Ties lifetimes of arena, buffers, etc all together."""
    var bufs: LeNet5GPUBuffers
    var model: LeNet5GPU
    var alloc: Self.Allocator

    def __init__(out self, ctx: DeviceContext) raises:
        self.alloc = Self.Allocator(ctx, LeNet5GPU.sizeInBytes()) # default
        self.bufs = LeNet5GPUBuffers(self.alloc)
        self.model = LeNet5GPU(self.bufs)

    def __init__(out self, mut arena: Self.Allocator) raises:
        self.alloc = arena^
        self.bufs = LeNet5GPUBuffers(self.alloc)
        self.model = LeNet5GPU(self.bufs)

    # def __del__(deinit self):
    #     pass

struct LeNet5GPUBuffers():
    """Stays on CPU — holds DeviceBuffers for host-side access (map_to_host, enqueue_copy_from, etc.).
    """
    var allocator_owns_memory: Bool # TODO: AcceptsAllocator trait idea (along with static sizeInBytes())
    var w01_storage: DeviceBuffer[ftype]
    var w23_storage: DeviceBuffer[ftype]
    var w45_storage: DeviceBuffer[ftype]
    var w56_storage: DeviceBuffer[ftype]
    var b01_storage: DeviceBuffer[ftype]
    var b23_storage: DeviceBuffer[ftype]
    var b45_storage: DeviceBuffer[ftype]
    var b56_storage: DeviceBuffer[ftype]

    def __init__(out self, mut arena: Some[GPUAllocator]) raises:
        self.allocator_owns_memory = True
        self.w01_storage = arena.alloc[ftype](comptime(LeNet5GPU.w0_1_layout.size()))
        self.w23_storage = arena.alloc[ftype](comptime(LeNet5GPU.w2_3_layout.size()))
        self.w45_storage = arena.alloc[ftype](comptime(LeNet5GPU.w4_5_layout.size()))
        self.w56_storage = arena.alloc[ftype](comptime(LeNet5GPU.w5_6_layout.size()))
        self.b01_storage = arena.alloc[ftype](comptime(LeNet5GPU.b0_1_layout.size()))
        self.b23_storage = arena.alloc[ftype](comptime(LeNet5GPU.b2_3_layout.size()))
        self.b45_storage = arena.alloc[ftype](comptime(LeNet5GPU.b4_5_layout.size()))
        self.b56_storage = arena.alloc[ftype](comptime(LeNet5GPU.b5_6_layout.size()))


    def __init__(out self, ctx: DeviceContext) raises:
        self.allocator_owns_memory = False
        self.w01_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.w0_1_layout.size()))
        self.w23_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.w2_3_layout.size()))
        self.w45_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.w4_5_layout.size()))
        self.w56_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.w5_6_layout.size()))
        self.b01_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.b0_1_layout.size()))
        self.b23_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.b2_3_layout.size()))
        self.b45_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.b4_5_layout.size()))
        self.b56_storage = ctx.enqueue_create_buffer[ftype](comptime(LeNet5GPU.b5_6_layout.size()))
        self.zero(sync_ctx = ctx)
        #ctx.synchronize() # or pass no ctx (sync_ctx = None) and do it yourself here

    @staticmethod
    def sizeInBytes() -> Int:
        return LeNet5GPU.sizeInBytes() # TODO: consolodate this pattern into an AcceptsAllocator trait or similar

    def loadCPUWeights(mut self, cpu_model: LeNet5) raises:
        self.w01_storage.enqueue_copy_from(cpu_model.weight0_1.ptr)
        self.w23_storage.enqueue_copy_from(cpu_model.weight2_3.ptr)
        self.w45_storage.enqueue_copy_from(cpu_model.weight4_5.ptr)
        self.w56_storage.enqueue_copy_from(cpu_model.weight5_6.ptr)
        self.b01_storage.enqueue_copy_from(cpu_model.bias0_1.ptr)
        self.b23_storage.enqueue_copy_from(cpu_model.bias2_3.ptr)
        self.b45_storage.enqueue_copy_from(cpu_model.bias4_5.ptr)
        self.b56_storage.enqueue_copy_from(cpu_model.bias5_6.ptr)

    def zero(mut self, *, sync_ctx: Optional[DeviceContext]) raises:
        self.w01_storage.enqueue_fill(0.0)
        self.w23_storage.enqueue_fill(0.0)
        self.w45_storage.enqueue_fill(0.0)
        self.w56_storage.enqueue_fill(0.0)
        self.b01_storage.enqueue_fill(0.0)
        self.b23_storage.enqueue_fill(0.0)
        self.b45_storage.enqueue_fill(0.0)
        self.b56_storage.enqueue_fill(0.0)
        if sync_ctx:
            sync_ctx.value().synchronize()

    # TODO:def __del__(deinit self):

struct LeNet5GPU(DevicePassable, TrivialRegisterPassable):
    """
    Same as the CPU version, but storage is on the GPU.
    LayoutTensors only — DeviceBuffers live in LeNet5GPUBuffers.
    """

    comptime device_type: AnyType = Self

    # WEIGHTS
    comptime w0_1_layout = Layout.row_major(
        INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL
    )
    var weight0_1: LayoutTensor[ftype, Self.w0_1_layout, MutAnyOrigin]

    comptime w2_3_layout = Layout.row_major(
        LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL
    )
    var weight2_3: LayoutTensor[ftype, Self.w2_3_layout, MutAnyOrigin]

    comptime w4_5_layout = Layout.row_major(
        LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL
    )
    var weight4_5: LayoutTensor[ftype, Self.w4_5_layout, MutAnyOrigin]

    comptime w5_6_layout = Layout.row_major(
        LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT
    )
    var weight5_6: LayoutTensor[ftype, Self.w5_6_layout, MutAnyOrigin]

    # BIASES
    comptime b0_1_layout = Layout.row_major(LAYER1)
    var bias0_1: LayoutTensor[ftype, Self.b0_1_layout, MutAnyOrigin]

    comptime b2_3_layout = Layout.row_major(LAYER3)
    var bias2_3: LayoutTensor[ftype, Self.b2_3_layout, MutAnyOrigin]

    comptime b4_5_layout = Layout.row_major(LAYER5)
    var bias4_5: LayoutTensor[ftype, Self.b4_5_layout, MutAnyOrigin]

    comptime b5_6_layout = Layout.row_major(OUTPUT)
    var bias5_6: LayoutTensor[ftype, Self.b5_6_layout, MutAnyOrigin]

    def __init__(out self, bufs: LeNet5GPUBuffers) raises:
        """Ensure you are initialized to all zeros..
        The storage local var copy trick allows for compliation; this is the only pattern I could get working.
        Lifetimes are handled either by DeviceSession() or manually (benchmark.compiler.keep() is useful).
        """
        var w0 = bufs.w01_storage
        var w2 = bufs.w23_storage
        var w4 = bufs.w45_storage
        var w5 = bufs.w56_storage 
        self.weight0_1 = LayoutTensor[ftype, Self.w0_1_layout, MutAnyOrigin](w0)
        self.weight2_3 = LayoutTensor[ftype, Self.w2_3_layout, MutAnyOrigin](w2)
        self.weight4_5 = LayoutTensor[ftype, Self.w4_5_layout, MutAnyOrigin](w4)
        self.weight5_6 = LayoutTensor[ftype, Self.w5_6_layout, MutAnyOrigin](w5)
        var b0 = bufs.b01_storage
        var b2 = bufs.b23_storage
        var b4 = bufs.b45_storage
        var b5 = bufs.b56_storage 
        self.bias0_1 = LayoutTensor[ftype, Self.b0_1_layout, MutAnyOrigin](b0)
        self.bias2_3 = LayoutTensor[ftype, Self.b2_3_layout, MutAnyOrigin](b2)
        self.bias4_5 = LayoutTensor[ftype, Self.b4_5_layout, MutAnyOrigin](b4)
        self.bias5_6 = LayoutTensor[ftype, Self.b5_6_layout, MutAnyOrigin](b5)

    @staticmethod
    def sizeInBytes() -> Int:
        var num_ftypes = comptime(
                Self.w0_1_layout.size() +
                Self.w2_3_layout.size() +
                Self.w4_5_layout.size() +
                Self.w5_6_layout.size() +
                Self.b0_1_layout.size() +
                Self.b2_3_layout.size() +
                Self.b4_5_layout.size() +
                Self.b5_6_layout.size()
        )
        return num_ftypes * size_of[ftype]()

    @staticmethod
    def get_type_name() -> String:
        return reflect[Self].name()

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self


struct FeatureGPUBuffers():
    """Stays on CPU — holds DeviceBuffers for host-side access (map_to_host, enqueue_copy_from, etc.).
    """

    var input_storage: DeviceBuffer[ftype]
    var layer1_storage: DeviceBuffer[ftype]
    var layer2_storage: DeviceBuffer[ftype]
    var layer3_storage: DeviceBuffer[ftype]
    var layer4_storage: DeviceBuffer[ftype]
    var layer5_storage: DeviceBuffer[ftype]
    var output_storage: DeviceBuffer[ftype]

    def __init__(out self, ctx: DeviceContext, feat: FeatureGPU) raises:
        self.input_storage = feat.input.to_device_buffer(ctx)
        self.layer1_storage = feat.layer1.to_device_buffer(ctx)
        self.layer2_storage = feat.layer2.to_device_buffer(ctx)
        self.layer3_storage = feat.layer3.to_device_buffer(ctx)
        self.layer4_storage = feat.layer4.to_device_buffer(ctx)
        self.layer5_storage = feat.layer5.to_device_buffer(ctx)
        self.output_storage = feat.output.to_device_buffer(ctx)

    def loadInput(mut self, image: Image) raises -> None:
        try:
            with self.input_storage.map_to_host() as load_me:
                for i in range(PADDED_SIZE):
                    for j in range(PADDED_SIZE):
                        load_me[i * PADDED_SIZE + j] = rebind[sftype](
                            image.pixels[i, j]
                        )
        except e:
            raise Error("loadInput FeatureGPUBuffers ERROR", e)


struct FeatureGPU(Copyable, Movable):
    """Holds intermediate results on the GPU. LayoutTensors only — DeviceBuffers live in FeatureGPUBuffers.
    """

    comptime input_layout = Layout.row_major(
        INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0
    )
    var input: LayoutTensor[ftype, FeatureGPU.input_layout, MutAnyOrigin]

    comptime layer1_layout = Layout.row_major(
        LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1
    )
    var layer1: LayoutTensor[ftype, FeatureGPU.layer1_layout, MutAnyOrigin]

    comptime layer2_layout = Layout.row_major(
        LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2
    )
    var layer2: LayoutTensor[ftype, FeatureGPU.layer2_layout, MutAnyOrigin]

    comptime layer3_layout = Layout.row_major(
        LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3
    )
    var layer3: LayoutTensor[ftype, FeatureGPU.layer3_layout, MutAnyOrigin]

    comptime layer4_layout = Layout.row_major(
        LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4
    )
    var layer4: LayoutTensor[ftype, FeatureGPU.layer4_layout, MutAnyOrigin]

    comptime layer5_layout = Layout.row_major(
        LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5
    )
    var layer5: LayoutTensor[ftype, FeatureGPU.layer5_layout, MutAnyOrigin]

    comptime output_layout = Layout.row_major(OUTPUT)
    var output: LayoutTensor[ftype, FeatureGPU.output_layout, MutAnyOrigin]

    def __init__(out self, ctx: DeviceContext) raises:
        """All buffers start zeroed."""
        var in_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.input_layout.size())
        )
        in_buf.enqueue_fill(0)
        var l1_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.layer1_layout.size())
        )
        l1_buf.enqueue_fill(0)
        var l2_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.layer2_layout.size())
        )
        l2_buf.enqueue_fill(0)
        var l3_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.layer3_layout.size())
        )
        l3_buf.enqueue_fill(0)
        var l4_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.layer4_layout.size())
        )
        l4_buf.enqueue_fill(0)
        var l5_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.layer5_layout.size())
        )
        l5_buf.enqueue_fill(0)
        var out_buf = ctx.enqueue_create_buffer[ftype](
            comptime (Self.output_layout.size())
        )
        out_buf.enqueue_fill(0)
        ctx.synchronize()
        self.input = LayoutTensor[ftype, Self.input_layout, MutAnyOrigin](
            in_buf
        )
        self.layer1 = LayoutTensor[ftype, Self.layer1_layout, MutAnyOrigin](
            l1_buf
        )
        self.layer2 = LayoutTensor[ftype, Self.layer2_layout, MutAnyOrigin](
            l2_buf
        )
        self.layer3 = LayoutTensor[ftype, Self.layer3_layout, MutAnyOrigin](
            l3_buf
        )
        self.layer4 = LayoutTensor[ftype, Self.layer4_layout, MutAnyOrigin](
            l4_buf
        )
        self.layer5 = LayoutTensor[ftype, Self.layer5_layout, MutAnyOrigin](
            l5_buf
        )
        self.output = LayoutTensor[ftype, Self.output_layout, MutAnyOrigin](
            out_buf
        )

    #
    # def __init__(out self, *, copy: Self):
    #     self.input  = copy.input
    #     self.layer1 = copy.layer1
    #     self.layer2 = copy.layer2
    #     self.layer3 = copy.layer3
    #     self.layer4 = copy.layer4
    #     self.layer5 = copy.layer5
    #     self.output = copy.output
    #
    # def __init__(out self, *, deinit take: Self):
    #     self.input  = take.input
    #     self.layer1 = take.layer1
    #     self.layer2 = take.layer2
    #     self.layer3 = take.layer3
    #     self.layer4 = take.layer4
    #     self.layer5 = take.layer5
    #     self.output = take.output
    #
