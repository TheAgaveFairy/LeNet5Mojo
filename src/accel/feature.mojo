"""GPU batched (SoA) activation buffers — one [bs, C, H, W] buffer per layer."""

from cpu.arena import ArenaSizable

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import size_of

from constants import (
    ftype,
    sftype,
    GPU_STREAM_BATCH_SIZE,
    BatchedFeatureLayouts,
)
from accel.arena import GPUBumpArenaAllocator


struct FeatureGPUBuffers[bs: Int = GPU_STREAM_BATCH_SIZE](
    ArenaSizable, Movable
):
    """CPU-side — holds one batched DeviceBuffer per live layer for host ops
    (map_to_host etc.). Layouts come from `BatchedFeatureLayouts[bs]`; kernel-side
    LayoutTensor views are built at launch (see StreamSlot.doWork). Arena must
    outlive this struct and any views over its buffers.
    """

    comptime layouts = BatchedFeatureLayouts[Self.bs]

    var allocator_owns_memory: Bool
    var input: DeviceBuffer[ftype]
    var layer2: DeviceBuffer[ftype]
    var layer3: DeviceBuffer[ftype]
    var layer4: DeviceBuffer[ftype]
    var layer5: DeviceBuffer[ftype]

    @staticmethod
    def sizeInBytes() -> Int:
        """Total bytes needed from the arena for one FeatureGPUBuffers instance.
        """
        return (
            comptime (Self.layouts.input.size())
            + comptime (Self.layouts.layer2.size())
            + comptime (Self.layouts.layer3.size())
            + comptime (Self.layouts.layer4.size())
            + comptime (Self.layouts.layer5.size())
        ) * size_of[sftype]()

    def __init__(out self, mut arena: GPUBumpArenaAllocator) raises:
        """Allocates all layer buffers from the arena. No GPU work — pure bookkeeping.
        Arena already zero-fills on creation; no enqueue_fill needed here.
        """
        self.allocator_owns_memory = True
        self.input = arena.alloc[ftype](comptime (Self.layouts.input.size()))
        self.layer2 = arena.alloc[ftype](comptime (Self.layouts.layer2.size()))
        self.layer3 = arena.alloc[ftype](comptime (Self.layouts.layer3.size()))
        self.layer4 = arena.alloc[ftype](comptime (Self.layouts.layer4.size()))
        self.layer5 = arena.alloc[ftype](comptime (Self.layouts.layer5.size()))
