from layout import Layout, LayoutTensor

from std.gpu.host import DeviceContext, DeviceBuffer
from std.builtin.device_passable import DevicePassable
from std.reflection import get_type_name

from cpu.model import LeNet5
from constants import (
    ftype, sftype,
    LENGTH_KERNEL, LENGTH_KERNEL_SQ,
    LENGTH_FEATURE0, LENGTH_FEATURE1, LENGTH_FEATURE2,
    LENGTH_FEATURE3, LENGTH_FEATURE4, LENGTH_FEATURE5,
    INPUT, LAYER1, LAYER2, LAYER3, LAYER4, LAYER5, OUTPUT,
    PADDED_SIZE,
)
from image import Image


struct LeNet5GPUBuffers:
    """Stays on CPU — holds DeviceBuffers for host-side access (map_to_host, enqueue_copy_from, etc.).
    """

    var w01_storage: DeviceBuffer[ftype]
    var w23_storage: DeviceBuffer[ftype]
    var w45_storage: DeviceBuffer[ftype]
    var w56_storage: DeviceBuffer[ftype]
    var b01_storage: DeviceBuffer[ftype]
    var b23_storage: DeviceBuffer[ftype]
    var b45_storage: DeviceBuffer[ftype]
    var b56_storage: DeviceBuffer[ftype]

    def __init__(out self, ctx: DeviceContext, model: LeNet5GPU) raises:
        self.w01_storage = model.weight0_1.to_device_buffer(ctx)
        self.w23_storage = model.weight2_3.to_device_buffer(ctx)
        self.w45_storage = model.weight4_5.to_device_buffer(ctx)
        self.w56_storage = model.weight5_6.to_device_buffer(ctx)
        self.b01_storage = model.bias0_1.to_device_buffer(ctx)
        self.b23_storage = model.bias2_3.to_device_buffer(ctx)
        self.b45_storage = model.bias4_5.to_device_buffer(ctx)
        self.b56_storage = model.bias5_6.to_device_buffer(ctx)


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

    def __init__(out self, ctx: DeviceContext) raises:
        """Initialize to all zeros. Pass in a DeviceContext from the caller."""
        var w01 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w0_1_layout.size())
        )
        w01.enqueue_fill(0)
        var w23 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w2_3_layout.size())
        )
        w23.enqueue_fill(0)
        var w45 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w4_5_layout.size())
        )
        w45.enqueue_fill(0)
        var w56 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w5_6_layout.size())
        )
        w56.enqueue_fill(0)
        var b01 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b0_1_layout.size())
        )
        b01.enqueue_fill(0)
        var b23 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b2_3_layout.size())
        )
        b23.enqueue_fill(0)
        var b45 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b4_5_layout.size())
        )
        b45.enqueue_fill(0)
        var b56 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b5_6_layout.size())
        )
        b56.enqueue_fill(0)
        ctx.synchronize()
        self.weight0_1 = LayoutTensor[ftype, Self.w0_1_layout, MutAnyOrigin](
            w01
        )
        self.weight2_3 = LayoutTensor[ftype, Self.w2_3_layout, MutAnyOrigin](
            w23
        )
        self.weight4_5 = LayoutTensor[ftype, Self.w4_5_layout, MutAnyOrigin](
            w45
        )
        self.weight5_6 = LayoutTensor[ftype, Self.w5_6_layout, MutAnyOrigin](
            w56
        )
        self.bias0_1 = LayoutTensor[ftype, Self.b0_1_layout, MutAnyOrigin](b01)
        self.bias2_3 = LayoutTensor[ftype, Self.b2_3_layout, MutAnyOrigin](b23)
        self.bias4_5 = LayoutTensor[ftype, Self.b4_5_layout, MutAnyOrigin](b45)
        self.bias5_6 = LayoutTensor[ftype, Self.b5_6_layout, MutAnyOrigin](b56)

    def __init__(out self, ctx: DeviceContext, cpu_model: LeNet5) raises:
        """Upload weights from a CPU model."""
        var w01 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w0_1_layout.size())
        )
        w01.enqueue_fill(0)
        w01.enqueue_copy_from(cpu_model.weight0_1.ptr)
        var w23 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w2_3_layout.size())
        )
        w23.enqueue_fill(0)
        w23.enqueue_copy_from(cpu_model.weight2_3.ptr)
        var w45 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w4_5_layout.size())
        )
        w45.enqueue_fill(0)
        w45.enqueue_copy_from(cpu_model.weight4_5.ptr)
        var w56 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.w5_6_layout.size())
        )
        w56.enqueue_fill(0)
        w56.enqueue_copy_from(cpu_model.weight5_6.ptr)
        var b01 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b0_1_layout.size())
        )
        b01.enqueue_fill(0)
        b01.enqueue_copy_from(cpu_model.bias0_1.ptr)
        var b23 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b2_3_layout.size())
        )
        b23.enqueue_fill(0)
        b23.enqueue_copy_from(cpu_model.bias2_3.ptr)
        var b45 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b4_5_layout.size())
        )
        b45.enqueue_fill(0)
        b45.enqueue_copy_from(cpu_model.bias4_5.ptr)
        var b56 = ctx.enqueue_create_buffer[ftype](
            comptime (Self.b5_6_layout.size())
        )
        b56.enqueue_fill(0)
        b56.enqueue_copy_from(cpu_model.bias5_6.ptr)
        ctx.synchronize()
        self.weight0_1 = LayoutTensor[ftype, Self.w0_1_layout, MutAnyOrigin](
            w01
        )
        self.weight2_3 = LayoutTensor[ftype, Self.w2_3_layout, MutAnyOrigin](
            w23
        )
        self.weight4_5 = LayoutTensor[ftype, Self.w4_5_layout, MutAnyOrigin](
            w45
        )
        self.weight5_6 = LayoutTensor[ftype, Self.w5_6_layout, MutAnyOrigin](
            w56
        )
        self.bias0_1 = LayoutTensor[ftype, Self.b0_1_layout, MutAnyOrigin](b01)
        self.bias2_3 = LayoutTensor[ftype, Self.b2_3_layout, MutAnyOrigin](b23)
        self.bias4_5 = LayoutTensor[ftype, Self.b4_5_layout, MutAnyOrigin](b45)
        self.bias5_6 = LayoutTensor[ftype, Self.b5_6_layout, MutAnyOrigin](b56)

    @staticmethod
    def get_type_name() -> String:
        return get_type_name[Self]()

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self


struct FeatureGPUBuffers:
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

    def loadInput(mut self, image: Image) -> None:
        try:
            with self.input_storage.map_to_host() as load_me:
                for i in range(PADDED_SIZE):
                    for j in range(PADDED_SIZE):
                        load_me[i * PADDED_SIZE + j] = rebind[sftype](
                            image.pixels[i, j]
                        )
        except e:
            print("loadInput FeatureGPUBuffers ERROR", e)


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
