from layout import LayoutTensor
from cpu.arena import ArenaSizable
from origin_util import untrack

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import size_of
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.reflection.reflect import reflect

from constants import (
    ftype,
    sftype,
    PADDED_SIZE,
    FeatureLayouts,
)
from image import Image
from accel.arena import GPUBumpArenaAllocator


struct FeatureGPUBuffers(ArenaSizable, Movable):
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
            comptime (FeatureLayouts.input.size())
            + comptime (FeatureLayouts.layer1.size())
            + comptime (FeatureLayouts.layer2.size())
            + comptime (FeatureLayouts.layer3.size())
            + comptime (FeatureLayouts.layer4.size())
            + comptime (FeatureLayouts.layer5.size())
            + comptime (FeatureLayouts.output.size())
        ) * size_of[sftype]()

    def __init__(out self, mut arena: GPUBumpArenaAllocator) raises:
        """Allocates all layer buffers from the arena. No GPU work — pure bookkeeping.
        Arena already zero-fills on creation; no enqueue_fill needed here.
        """
        self.allocator_owns_memory = True
        self.input = arena.alloc[ftype](comptime (FeatureLayouts.input.size()))
        self.layer1 = arena.alloc[ftype](
            comptime (FeatureLayouts.layer1.size())
        )
        self.layer2 = arena.alloc[ftype](
            comptime (FeatureLayouts.layer2.size())
        )
        self.layer3 = arena.alloc[ftype](
            comptime (FeatureLayouts.layer3.size())
        )
        self.layer4 = arena.alloc[ftype](
            comptime (FeatureLayouts.layer4.size())
        )
        self.layer5 = arena.alloc[ftype](
            comptime (FeatureLayouts.layer5.size())
        )
        self.output = arena.alloc[ftype](
            comptime (FeatureLayouts.output.size())
        )

    def __init__(out self, ctx: DeviceContext) raises:
        """Standalone variant: allocate each layer buffer directly from `ctx` and
        zero-fill, bypassing the arena.
        """
        self.allocator_owns_memory = False
        self.input = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.input.size())
        )
        self.input.enqueue_fill(0.0)
        self.layer1 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.layer1.size())
        )
        self.layer1.enqueue_fill(0.0)
        self.layer2 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.layer2.size())
        )
        self.layer2.enqueue_fill(0.0)
        self.layer3 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.layer3.size())
        )
        self.layer3.enqueue_fill(0.0)
        self.layer4 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.layer4.size())
        )
        self.layer4.enqueue_fill(0.0)
        self.layer5 = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.layer5.size())
        )
        self.layer5.enqueue_fill(0.0)
        self.output = ctx.enqueue_create_buffer[ftype](
            comptime (FeatureLayouts.output.size())
        )
        self.output.enqueue_fill(0.0)

    # left for debugging
    @deprecated(
        "This synchronizes on every call. There are much better patterns."
    )
    def loadInput(mut self, image: Image) -> None:
        try:
            with self.input.map_to_host() as load_me:
                for i in range(PADDED_SIZE):
                    for j in range(PADDED_SIZE):
                        load_me[i * PADDED_SIZE + j] = rebind[sftype](
                            image.pixels[0, i, j]
                        )
        except e:
            print("loadInput FeatureGPUBuffers ERROR", e)


struct FeatureGPU(Copyable, DevicePassable, Movable):
    """GPU-side — LayoutTensors only, for passing to kernels.
    Constructed from a FeatureGPUBuffers; caller must keep the buffers (and arena) alive.
    No GPU ops in __init__ — just pointer wiring.
    """

    comptime device_type: AnyType = Self

    var input: LayoutTensor[ftype, FeatureLayouts.input, MutUntrackedOrigin]
    var layer1: LayoutTensor[ftype, FeatureLayouts.layer1, MutUntrackedOrigin]
    var layer2: LayoutTensor[ftype, FeatureLayouts.layer2, MutUntrackedOrigin]
    var layer3: LayoutTensor[ftype, FeatureLayouts.layer3, MutUntrackedOrigin]
    var layer4: LayoutTensor[ftype, FeatureLayouts.layer4, MutUntrackedOrigin]
    var layer5: LayoutTensor[ftype, FeatureLayouts.layer5, MutUntrackedOrigin]
    var output: LayoutTensor[ftype, FeatureLayouts.output, MutUntrackedOrigin]

    @staticmethod
    def get_type_name() -> String:
        return reflect[Self].name()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    def __init__(out self, bufs: FeatureGPUBuffers):
        """Wire LayoutTensor views over `bufs`' device buffers — no allocation or GPU work.
        """
        var b_input = bufs.input
        var b_layer1 = bufs.layer1
        var b_layer2 = bufs.layer2
        var b_layer3 = bufs.layer3
        var b_layer4 = bufs.layer4
        var b_layer5 = bufs.layer5
        var b_output = bufs.output
        self.input = untrack(LayoutTensor[ftype, FeatureLayouts.input](b_input))
        self.layer1 = untrack(
            LayoutTensor[ftype, FeatureLayouts.layer1](b_layer1)
        )
        self.layer2 = untrack(
            LayoutTensor[ftype, FeatureLayouts.layer2](b_layer2)
        )
        self.layer3 = untrack(
            LayoutTensor[ftype, FeatureLayouts.layer3](b_layer3)
        )
        self.layer4 = untrack(
            LayoutTensor[ftype, FeatureLayouts.layer4](b_layer4)
        )
        self.layer5 = untrack(
            LayoutTensor[ftype, FeatureLayouts.layer5](b_layer5)
        )
        self.output = untrack(
            LayoutTensor[ftype, FeatureLayouts.output](b_output)
        )
