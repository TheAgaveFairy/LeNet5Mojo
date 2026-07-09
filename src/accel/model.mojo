"""GPU model storage: `LeNet5GPU`, its device buffers, and `DeviceSession`."""

from layout import LayoutTensor

from std.gpu.host import DeviceContext, DeviceBuffer
from std.sys import size_of

from cpu.model import LeNet5
from cpu.arena import ArenaSizable
from accel.arena import GPUAllocator, GPUBumpArenaAllocator, GPUSystemAllocator
from origin_util import untrack
from constants import (
    ftype,
    sftype,
    LAYER4,
    LAYER5,
    LENGTH_KERNEL,
    WeightLayouts,
    BiasLayouts,
)


struct DeviceSession[Allocator: GPUAllocator]():
    """Ties lifetimes of arena, buffers, etc all together."""

    var bufs: LeNet5GPUBuffers
    var model: LeNet5GPU
    var alloc: Self.Allocator

    def __init__(out self, ctx: DeviceContext) raises:
        """Build the allocator, buffers, and model together from `ctx`."""
        self.alloc = Self.Allocator(ctx, LeNet5GPU.sizeInBytes())  # default
        self.bufs = LeNet5GPUBuffers(self.alloc)
        self.model = LeNet5GPU(self.bufs)

    def __init__(out self, var arena: Self.Allocator) raises:
        """Adopt a pre-built allocator, deriving buffers and model from it."""
        self.bufs = LeNet5GPUBuffers(arena)
        self.model = LeNet5GPU(self.bufs)
        self.alloc = arena^


struct LeNet5GPUBuffers(ArenaSizable):
    """Stays on CPU — holds DeviceBuffers for host-side access (map_to_host, enqueue_copy_from, etc.).
    """

    var allocator_owns_memory: Bool
    var w01_storage: DeviceBuffer[ftype]
    var w23_storage: DeviceBuffer[ftype]
    var w45_storage: DeviceBuffer[ftype]
    var w45g_storage: DeviceBuffer[ftype]  # w45 transposed for the GEMM path
    var w56_storage: DeviceBuffer[ftype]
    var b01_storage: DeviceBuffer[ftype]
    var b23_storage: DeviceBuffer[ftype]
    var b45_storage: DeviceBuffer[ftype]
    var b56_storage: DeviceBuffer[ftype]

    def __init__(out self, mut arena: Some[GPUAllocator]) raises:
        """Sub-allocate every weight and bias buffer from `arena`."""
        self.allocator_owns_memory = True
        self.w01_storage = arena.alloc[ftype](
            comptime (WeightLayouts.w01.size())
        )
        self.w23_storage = arena.alloc[ftype](
            comptime (WeightLayouts.w23.size())
        )
        self.w45_storage = arena.alloc[ftype](
            comptime (WeightLayouts.w45.size())
        )
        self.w45g_storage = arena.alloc[ftype](
            comptime (WeightLayouts.w45g.size())
        )
        self.w56_storage = arena.alloc[ftype](
            comptime (WeightLayouts.w56.size())
        )
        self.b01_storage = arena.alloc[ftype](comptime (BiasLayouts.b01.size()))
        self.b23_storage = arena.alloc[ftype](comptime (BiasLayouts.b23.size()))
        self.b45_storage = arena.alloc[ftype](comptime (BiasLayouts.b45.size()))
        self.b56_storage = arena.alloc[ftype](comptime (BiasLayouts.b56.size()))

    def __init__(out self, ctx: DeviceContext) raises:
        """Standalone variant: allocate every buffer directly from `ctx`, then zero them.
        """
        self.allocator_owns_memory = False
        self.w01_storage = ctx.enqueue_create_buffer[ftype](
            comptime (WeightLayouts.w01.size())
        )
        self.w23_storage = ctx.enqueue_create_buffer[ftype](
            comptime (WeightLayouts.w23.size())
        )
        self.w45_storage = ctx.enqueue_create_buffer[ftype](
            comptime (WeightLayouts.w45.size())
        )
        self.w45g_storage = ctx.enqueue_create_buffer[ftype](
            comptime (WeightLayouts.w45g.size())
        )
        self.w56_storage = ctx.enqueue_create_buffer[ftype](
            comptime (WeightLayouts.w56.size())
        )
        self.b01_storage = ctx.enqueue_create_buffer[ftype](
            comptime (BiasLayouts.b01.size())
        )
        self.b23_storage = ctx.enqueue_create_buffer[ftype](
            comptime (BiasLayouts.b23.size())
        )
        self.b45_storage = ctx.enqueue_create_buffer[ftype](
            comptime (BiasLayouts.b45.size())
        )
        self.b56_storage = ctx.enqueue_create_buffer[ftype](
            comptime (BiasLayouts.b56.size())
        )
        self.zero(sync_ctx=ctx)
        # ctx.synchronize() # or pass no ctx (sync_ctx = None) and do it yourself here

    @staticmethod
    def sizeInBytes() -> Int:
        return LeNet5GPU.sizeInBytes()

    def loadCPUWeights(mut self, cpu_model: LeNet5) raises:
        """Copy a trained CPU model's weights and biases up into the device buffers.
        """
        self.w01_storage.enqueue_copy_from(cpu_model.weight0_1.ptr)
        self.w23_storage.enqueue_copy_from(cpu_model.weight2_3.ptr)
        self.w45_storage.enqueue_copy_from(cpu_model.weight4_5.ptr)
        self.w56_storage.enqueue_copy_from(cpu_model.weight5_6.ptr)
        # w45 rearranged once, (ic, oc, kw, kh) -> GEMM b(K=ic*kw*kh, N=oc);
        # written through map_to_host (syncs on exit — no staging lifetime)
        with self.w45g_storage.map_to_host() as w45g_host:
            for ic in range(LAYER4):
                for oc in range(LAYER5):
                    for kw in range(LENGTH_KERNEL):
                        for kh in range(LENGTH_KERNEL):
                            var k = (
                                ic * LENGTH_KERNEL + kw
                            ) * LENGTH_KERNEL + kh
                            w45g_host[
                                k * LAYER5 + oc
                            ] = cpu_model.weight4_5.ptr[
                                ((ic * LAYER5 + oc) * LENGTH_KERNEL + kw)
                                * LENGTH_KERNEL
                                + kh
                            ]
        self.b01_storage.enqueue_copy_from(cpu_model.bias0_1.ptr)
        self.b23_storage.enqueue_copy_from(cpu_model.bias2_3.ptr)
        self.b45_storage.enqueue_copy_from(cpu_model.bias4_5.ptr)
        self.b56_storage.enqueue_copy_from(cpu_model.bias5_6.ptr)

    def zero(mut self, *, sync_ctx: Optional[DeviceContext]) raises:
        """Zero every buffer; blocks until done when `sync_ctx` is given."""
        self.w01_storage.enqueue_fill(0.0)
        self.w23_storage.enqueue_fill(0.0)
        self.w45_storage.enqueue_fill(0.0)
        self.w45g_storage.enqueue_fill(0.0)
        self.w56_storage.enqueue_fill(0.0)
        self.b01_storage.enqueue_fill(0.0)
        self.b23_storage.enqueue_fill(0.0)
        self.b45_storage.enqueue_fill(0.0)
        self.b56_storage.enqueue_fill(0.0)
        if sync_ctx:
            sync_ctx.value().synchronize()


struct LeNet5GPU(ArenaSizable):
    """
    Same as the CPU version, but storage is on the GPU.
    LayoutTensors only — DeviceBuffers live in LeNet5GPUBuffers.
    Host-side aggregate only: kernels take individual weight/bias tensors
    (already device-passable), so no DevicePassable machinery here.
    """

    # WEIGHTS
    var weight0_1: LayoutTensor[ftype, WeightLayouts.w01, MutUntrackedOrigin]
    var weight2_3: LayoutTensor[ftype, WeightLayouts.w23, MutUntrackedOrigin]
    var weight4_5: LayoutTensor[ftype, WeightLayouts.w45, MutUntrackedOrigin]
    var weight4_5_gemm: LayoutTensor[
        ftype, WeightLayouts.w45g, MutUntrackedOrigin
    ]
    var weight5_6: LayoutTensor[ftype, WeightLayouts.w56, MutUntrackedOrigin]

    # BIASES
    var bias0_1: LayoutTensor[ftype, BiasLayouts.b01, MutUntrackedOrigin]
    var bias2_3: LayoutTensor[ftype, BiasLayouts.b23, MutUntrackedOrigin]
    var bias4_5: LayoutTensor[ftype, BiasLayouts.b45, MutUntrackedOrigin]
    var bias5_6: LayoutTensor[ftype, BiasLayouts.b56, MutUntrackedOrigin]

    def __init__(out self, bufs: LeNet5GPUBuffers) raises:
        """Ensure you are initialized to all zeros..
        The storage local var copy trick allows for compliation; this is the only pattern I could get working.
        Lifetimes are handled either by DeviceSession() or manually (benchmark.compiler.keep() is useful).
        """
        var w0 = bufs.w01_storage
        var w2 = bufs.w23_storage
        var w4 = bufs.w45_storage
        var w4g = bufs.w45g_storage
        var w5 = bufs.w56_storage
        self.weight0_1 = untrack(LayoutTensor[ftype, WeightLayouts.w01](w0))
        self.weight2_3 = untrack(LayoutTensor[ftype, WeightLayouts.w23](w2))
        self.weight4_5 = untrack(LayoutTensor[ftype, WeightLayouts.w45](w4))
        self.weight4_5_gemm = untrack(
            LayoutTensor[ftype, WeightLayouts.w45g](w4g)
        )
        self.weight5_6 = untrack(LayoutTensor[ftype, WeightLayouts.w56](w5))
        var b0 = bufs.b01_storage
        var b2 = bufs.b23_storage
        var b4 = bufs.b45_storage
        var b5 = bufs.b56_storage
        self.bias0_1 = untrack(LayoutTensor[ftype, BiasLayouts.b01](b0))
        self.bias2_3 = untrack(LayoutTensor[ftype, BiasLayouts.b23](b2))
        self.bias4_5 = untrack(LayoutTensor[ftype, BiasLayouts.b45](b4))
        self.bias5_6 = untrack(LayoutTensor[ftype, BiasLayouts.b56](b5))

    @staticmethod
    def sizeInBytes() -> Int:
        var num_ftypes = comptime (
            WeightLayouts.w01.size()
            + WeightLayouts.w23.size()
            + WeightLayouts.w45.size()
            + WeightLayouts.w45g.size()
            + WeightLayouts.w56.size()
            + BiasLayouts.b01.size()
            + BiasLayouts.b23.size()
            + BiasLayouts.b45.size()
            + BiasLayouts.b56.size()
        )
        return num_ftypes * size_of[ftype]()
