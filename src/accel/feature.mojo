from layout import Layout, LayoutTensor

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import size_of
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.reflection.reflect import reflect

from constants import (
    ftype,
    sftype,
    INPUT,
    LAYER1,
    LAYER2,
    LAYER3,
    LAYER4,
    LAYER5,
    OUTPUT,
    PADDED_SIZE,
    LENGTH_FEATURE0,
    LENGTH_FEATURE1,
    LENGTH_FEATURE2,
    LENGTH_FEATURE3,
    LENGTH_FEATURE4,
    LENGTH_FEATURE5,
)
from image import Image
from accel.arena import GPUBumpArenaAllocator


struct FeatureGPUBuffers(Movable):
    """CPU-side — holds DeviceBuffer handles for host ops (map_to_host, loadInput, etc.).
    Owns the arena sub-buffer refs; FeatureGPU's LayoutTensors point into these.
    Arena must outlive both this struct and any FeatureGPU built from it.
    """

    var allocator_owns_memory: Bool
    var input: DeviceBuffer[ftype]
    var layer1: DeviceBuffer[ftype]
    var layer2: DeviceBuffer[ftype]
    var layer3: DeviceBuffer[ftype]
    var layer4: DeviceBuffer[ftype]
    var layer5: DeviceBuffer[ftype]
    var output: DeviceBuffer[ftype]

    @staticmethod
    def sizeInBytes() -> Int:
        """Total bytes needed from the arena for one FeatureGPUBuffers instance.
        """
        return (
            comptime (FeatureGPU.input_layout.size())
            + comptime (FeatureGPU.layer1_layout.size())
            + comptime (FeatureGPU.layer2_layout.size())
            + comptime (FeatureGPU.layer3_layout.size())
            + comptime (FeatureGPU.layer4_layout.size())
            + comptime (FeatureGPU.layer5_layout.size())
            + comptime (FeatureGPU.output_layout.size())
        ) * size_of[sftype]()

    def __init__(out self, mut arena: GPUBumpArenaAllocator) raises:
        """Allocates all layer buffers from the arena. No GPU work — pure bookkeeping.
        Arena already zero-fills on creation; no enqueue_fill needed here.
        """
        self.allocator_owns_memory = True
        self.input = arena.alloc[ftype](
            comptime (FeatureGPU.input_layout.size())
        )
        self.layer1 = arena.alloc[ftype](
            comptime (FeatureGPU.layer1_layout.size())
        )
        self.layer2 = arena.alloc[ftype](
            comptime (FeatureGPU.layer2_layout.size())
        )
        self.layer3 = arena.alloc[ftype](
            comptime (FeatureGPU.layer3_layout.size())
        )
        self.layer4 = arena.alloc[ftype](
            comptime (FeatureGPU.layer4_layout.size())
        )
        self.layer5 = arena.alloc[ftype](
            comptime (FeatureGPU.layer5_layout.size())
        )
        self.output = arena.alloc[ftype](
            comptime (FeatureGPU.output_layout.size())
        )

    def __init__(out self, ctx: DeviceContext) raises:
        self.allocator_owns_memory = False
        self.input = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.input_layout.size())
        )
        self.input.enqueue_fill(0.0)
        self.layer1 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.layer1_layout.size())
        )
        self.layer1.enqueue_fill(0.0)
        self.layer2 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.layer2_layout.size())
        )
        self.layer2.enqueue_fill(0.0)
        self.layer3 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.layer3_layout.size())
        )
        self.layer3.enqueue_fill(0.0)
        self.layer4 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.layer4_layout.size())
        )
        self.layer4.enqueue_fill(0.0)
        self.layer5 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.layer5_layout.size())
        )
        self.layer5.enqueue_fill(0.0)
        self.output = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureGPU.output_layout.size())
        )
        self.output.enqueue_fill(0.0)

    def loadInput(mut self, image: Image) -> None:
        try:
            with self.input.map_to_host() as load_me:
                for i in range(PADDED_SIZE):
                    for j in range(PADDED_SIZE):
                        load_me[i * PADDED_SIZE + j] = rebind[sftype](
                            image.pixels[i, j]
                        )
        except e:
            print("loadInput FeatureGPUBuffers ERROR", e)


struct FeatureGPU(Copyable, DevicePassable, Movable):
    """GPU-side — LayoutTensors only, for passing to kernels.
    Constructed from a FeatureGPUBuffers; caller must keep the buffers (and arena) alive.
    No GPU ops in __init__ — just pointer wiring.
    """

    comptime device_type: AnyType = Self

    comptime input_layout = Layout.row_major(
        INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0
    )
    comptime layer1_layout = Layout.row_major(
        LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1
    )
    comptime layer2_layout = Layout.row_major(
        LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2
    )
    comptime layer3_layout = Layout.row_major(
        LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3
    )
    comptime layer4_layout = Layout.row_major(
        LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4
    )
    comptime layer5_layout = Layout.row_major(
        LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5
    )
    comptime output_layout = Layout.row_major(OUTPUT)

    var input: LayoutTensor[ftype, FeatureGPU.input_layout, MutAnyOrigin]
    var layer1: LayoutTensor[ftype, FeatureGPU.layer1_layout, MutAnyOrigin]
    var layer2: LayoutTensor[ftype, FeatureGPU.layer2_layout, MutAnyOrigin]
    var layer3: LayoutTensor[ftype, FeatureGPU.layer3_layout, MutAnyOrigin]
    var layer4: LayoutTensor[ftype, FeatureGPU.layer4_layout, MutAnyOrigin]
    var layer5: LayoutTensor[ftype, FeatureGPU.layer5_layout, MutAnyOrigin]
    var output: LayoutTensor[ftype, FeatureGPU.output_layout, MutAnyOrigin]

    @staticmethod
    def get_type_name() -> String:
        return reflect[Self].name()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    def __init__(out self, bufs: FeatureGPUBuffers):
        var b_input = bufs.input
        var b_layer1 = bufs.layer1
        var b_layer2 = bufs.layer2
        var b_layer3 = bufs.layer3
        var b_layer4 = bufs.layer4
        var b_layer5 = bufs.layer5
        var b_output = bufs.output
        self.input = LayoutTensor[ftype, Self.input_layout, MutAnyOrigin](
            b_input
        )
        self.layer1 = LayoutTensor[ftype, Self.layer1_layout, MutAnyOrigin](
            b_layer1
        )
        self.layer2 = LayoutTensor[ftype, Self.layer2_layout, MutAnyOrigin](
            b_layer2
        )
        self.layer3 = LayoutTensor[ftype, Self.layer3_layout, MutAnyOrigin](
            b_layer3
        )
        self.layer4 = LayoutTensor[ftype, Self.layer4_layout, MutAnyOrigin](
            b_layer4
        )
        self.layer5 = LayoutTensor[ftype, Self.layer5_layout, MutAnyOrigin](
            b_layer5
        )
        self.output = LayoutTensor[ftype, Self.output_layout, MutAnyOrigin](
            b_output
        )
