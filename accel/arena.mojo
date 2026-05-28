from layout import Layout, LayoutTensor

from std.gpu import thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.sys.info import size_of, align_of
from std.sys import has_accelerator
from std.testing import assert_equal
from std.os import abort


comptime ftype = DType.float32
comptime sftype = Scalar[ftype]

comptime ROWS = 4
comptime COLS = 10
comptime layout = Layout.row_major(ROWS, COLS)
comptime TILE_SIZE = ROWS * COLS


trait GPUAllocator(ImplicitlyDestructible, Movable):
    def __init__(out self, ctx: DeviceContext, capacity_bytes: Int) raises:
        ...

    def alloc[dtype: DType](mut self, count: Int) raises -> DeviceBuffer[dtype]:
        ...

    def free_all(mut self):
        # Bump: reset offset only. System: release all tracked buffers.
        ...

    def zero(mut self) raises:
        # Fill all allocated memory with 0 — do not change offset/tracking.
        ...

    def wipe(mut self) raises:
        # zero() then free_all().
        ...


struct GPUBumpArenaAllocator(GPUAllocator):
    """Host-side GPU bump allocator backed by one byte-level DeviceBuffer.
    Tracks offset in bytes — alloc[dtype](count) handles alignment padding
    and returns a typed sub-buffer via create_sub_buffer.
    """

    var buffer: DeviceBuffer[DType.uint8]
    var capacity: Int  # bytes
    var offset: Int  # bytes

    def __init__(out self, ctx: DeviceContext, capacity_bytes: Int) raises:
        self.buffer = ctx.enqueue_create_buffer[DType.uint8](capacity_bytes)
        self.buffer.enqueue_fill(UInt8(0))
        self.capacity = capacity_bytes
        self.offset = 0

    def __init__(out self, *, deinit take: Self):
        self.buffer = take.buffer^
        self.capacity = take.capacity
        self.offset = take.offset

    def alloc[dtype: DType](mut self, count: Int) raises -> DeviceBuffer[dtype]:
        var elem_size = size_of[Scalar[dtype]]()
        var alignment = align_of[Scalar[dtype]]()
        var aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1)
        var total_bytes = count * elem_size
        if aligned_offset + total_bytes > self.capacity:
            abort("GPUArena out of memory!")
        var sub = self.buffer.create_sub_buffer[dtype](
            aligned_offset // elem_size, count
        )
        self.offset = aligned_offset + total_bytes
        return sub

    def free_all(mut self):
        """Reset bump offset — GPU memory stays allocated."""
        self.offset = 0

    def zero(mut self) raises:
        """Fill slab with zeros — offset unchanged."""
        self.buffer.enqueue_fill(UInt8(0))

    def wipe(mut self) raises:
        """Zero slab then reset offset."""
        self.buffer.enqueue_fill(UInt8(0))
        self.offset = 0

    def base_address(self) -> Int:
        return Int(self.buffer.unsafe_ptr())


struct GPUSystemAllocator(GPUAllocator):
    """
    GPU system allocator. Each alloc() creates an independent DeviceBuffer.
    Tracks raw byte-level buffers; typed views are sub-buffers of those.
    free_all() destroys all tracked buffers — any held sub-buffer views become invalid.
    """

    var _ctx: DeviceContext
    var _allocations: List[DeviceBuffer[DType.uint8]]

    def __init__(out self, ctx: DeviceContext, capacity_bytes: Int) raises:
        self._ctx = ctx
        self._allocations = List[DeviceBuffer[DType.uint8]]()

    def __init__(out self, *, deinit take: Self):
        self._ctx = take._ctx
        self._allocations = take._allocations^

    def alloc[dtype: DType](mut self, count: Int) raises -> DeviceBuffer[dtype]:
        var elem_size = size_of[Scalar[dtype]]()
        var raw = self._ctx.enqueue_create_buffer[DType.uint8](
            count * elem_size
        )
        var view = raw.create_sub_buffer[dtype](0, count)
        self._allocations.append(raw^)
        return view

    def free_all(mut self):
        """Destroy all tracked DeviceBuffers, releasing GPU memory."""
        self._allocations.clear()

    def zero(mut self) raises:
        """Fill all tracked buffers with 0."""
        for i in range(len(self._allocations)):
            self._allocations[i].enqueue_fill(UInt8(0))

    def wipe(mut self) raises:
        """Zero all tracked buffers then release them."""
        self.zero()
        self.free_all()


# Tests


def work(
    input: LayoutTensor[mut=False, ftype, layout, ImmutAnyOrigin],
    output: LayoutTensor[mut=True, ftype, layout, MutAnyOrigin],
):
    var tid = thread_idx.x
    comptime for x in range(COLS):
        output[tid, x] += input[tid, x]


def test_gpu_arena_offsets(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_offsets ---")
    var arena = GPUBumpArenaAllocator(ctx, 40 * size_of[sftype]())
    print("Arena base addr:", arena.base_address())

    var sub0 = arena.alloc[ftype](5)
    var sub1 = arena.alloc[ftype](7)
    var sub2 = arena.alloc[ftype](3)
    ctx.synchronize()

    var addr0 = Int(sub0.unsafe_ptr())
    var addr1 = Int(sub1.unsafe_ptr())
    var addr2 = Int(sub2.unsafe_ptr())
    var elem_bytes = size_of[sftype]()

    print("sub0 @ addr:", addr0)
    print(
        "sub1 @ addr:",
        addr1,
        " delta:",
        addr1 - addr0,
        " (expected",
        5 * elem_bytes,
        "bytes)",
    )
    print(
        "sub2 @ addr:",
        addr2,
        " delta:",
        addr2 - addr1,
        " (expected",
        7 * elem_bytes,
        "bytes)",
    )

    assert_equal(addr1 - addr0, 5 * elem_bytes)
    assert_equal(addr2 - addr1, 7 * elem_bytes)
    print("PASS")


def test_gpu_arena_free_all(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_free_all ---")
    var arena = GPUBumpArenaAllocator(ctx, 20 * size_of[sftype]())

    var sub0 = arena.alloc[ftype](10)
    var addr0 = Int(sub0.unsafe_ptr())
    print("Before free_all: alloc addr =", addr0)

    arena.free_all()

    var sub1 = arena.alloc[ftype](10)
    var addr1 = Int(sub1.unsafe_ptr())
    print("After free_all:  alloc addr =", addr1)

    assert_equal(addr0, addr1)
    print("PASS")


def test_gpu_arena_wipe(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_wipe ---")
    var arena = GPUBumpArenaAllocator(ctx, 20 * size_of[sftype]())
    var sub = arena.alloc[ftype](10)

    sub.enqueue_fill(69.0)
    ctx.synchronize()

    with sub.map_to_host() as host:
        print("Before wipe: host[0] =", host[0])
        assert_equal(host[0], sftype(69.0))

    arena.wipe()
    ctx.synchronize()

    with sub.map_to_host() as host:
        print("After wipe:  host[0] =", host[0])
        assert_equal(host[0], sftype(0.0))
    print("PASS")


def test_gpu_arena_zero(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_zero ---")
    var arena = GPUBumpArenaAllocator(ctx, 20 * size_of[sftype]())
    var sub0 = arena.alloc[ftype](5)
    sub0.enqueue_fill(42.0)
    ctx.synchronize()

    var offset_before = arena.offset
    arena.zero()
    ctx.synchronize()

    assert_equal(arena.offset, offset_before)  # offset unchanged
    with sub0.map_to_host() as host:
        print("After zero: host[0] =", host[0])
        assert_equal(host[0], sftype(0.0))

    # can still alloc from same offset (not reset)
    var sub1 = arena.alloc[ftype](5)
    assert_equal(
        Int(sub1.unsafe_ptr()), Int(sub0.unsafe_ptr()) + 5 * size_of[sftype]()
    )
    print("PASS")


def test_gpu_arena_mixed_dtype(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_mixed_dtype ---")
    # 5*f32(20B) + pad-to-8 + 4*f64(32B) + 3*f32(12B) = ~68 bytes
    var arena = GPUBumpArenaAllocator(ctx, 256)
    print("Arena base addr:", arena.base_address())

    var f32_a = arena.alloc[DType.float32](5)
    var f64_b = arena.alloc[DType.float64](4)
    var f32_c = arena.alloc[DType.float32](3)

    var addr_a = Int(f32_a.unsafe_ptr())
    var addr_b = Int(f64_b.unsafe_ptr())
    var addr_c = Int(f32_c.unsafe_ptr())

    print("f32[5] @ addr:", addr_a)
    print(
        "f64[4] @ addr:",
        addr_b,
        " delta from f32:",
        addr_b - addr_a,
        "(padded to 8-align)",
    )
    print(
        "f32[3] @ addr:",
        addr_c,
        " delta from f64:",
        addr_c - addr_b,
        " (expected",
        4 * size_of[Scalar[DType.float64]](),
        ")",
    )

    assert_equal(addr_b % 8, 0)
    assert_equal(addr_c - addr_b, 4 * size_of[Scalar[DType.float64]]())

    f32_a.enqueue_fill(1.5)
    f64_b.enqueue_fill(Scalar[DType.float64](2.5))
    f32_c.enqueue_fill(3.5)
    ctx.synchronize()

    with f32_a.map_to_host() as h:
        print("f32_a[0] =", h[0], " (expected 1.5)")
        assert_equal(h[0], Scalar[DType.float32](1.5))
    with f64_b.map_to_host() as h:
        print("f64_b[0] =", h[0], " (expected 2.5)")
        assert_equal(h[0], Scalar[DType.float64](2.5))
    with f32_c.map_to_host() as h:
        print("f32_c[0] =", h[0], " (expected 3.5)")
        assert_equal(h[0], Scalar[DType.float32](3.5))
    print("PASS")


def test_gpu_arena_work_kernel(ctx: DeviceContext) raises:
    print("\n--- test_gpu_arena_work_kernel ---")
    var arena = GPUBumpArenaAllocator(ctx, TILE_SIZE * size_of[sftype]() * 4)
    print("Arena base addr:", arena.base_address())

    var in_buf = arena.alloc[ftype](TILE_SIZE)
    var out_buf = arena.alloc[ftype](TILE_SIZE)
    print("Input  device addr:", Int(in_buf.unsafe_ptr()))
    print("Output device addr:", Int(out_buf.unsafe_ptr()))
    print(
        "Offset between:    ",
        Int(out_buf.unsafe_ptr()) - Int(in_buf.unsafe_ptr()),
        "bytes (expected",
        TILE_SIZE * size_of[sftype](),
        ")",
    )

    in_buf.enqueue_fill(1.0)
    out_buf.enqueue_fill(2.0)
    ctx.synchronize()

    var input_t = LayoutTensor[ftype, layout, MutAnyOrigin](in_buf)
    var output_t = LayoutTensor[ftype, layout, MutAnyOrigin](out_buf)

    ctx.enqueue_function[work](input_t, output_t, grid_dim=1, block_dim=ROWS)
    ctx.synchronize()

    with out_buf.map_to_host() as host:
        var result = LayoutTensor[ftype, layout, MutAnyOrigin](host)
        print("Output (all should be 3.0):")
        print(result)
        comptime for r in range(ROWS):
            comptime for c in range(COLS):
                assert_equal(result[r, c], sftype(3.0))
    print("PASS")


def test_gpu_system_alloc_basic(mut ctx: DeviceContext) raises:
    print("\n--- test_gpu_system_alloc_basic ---")
    var sa = GPUSystemAllocator(ctx, 0)
    var buf = sa.alloc[ftype](8)
    buf.enqueue_fill(7.0)
    ctx.synchronize()
    with buf.map_to_host() as host:
        print("host[0] =", host[0])
        assert_equal(host[0], sftype(7.0))
    assert_equal(len(sa._allocations), 1)
    print("PASS")


def test_gpu_system_multi_dtype(ctx: DeviceContext) raises:
    print("\n--- test_gpu_system_multi_dtype ---")
    # var ctx_cpy = ctx
    var sa = GPUSystemAllocator(ctx, 0)
    var f32_buf = sa.alloc[DType.float32](4)
    var f64_buf = sa.alloc[DType.float64](4)
    var u8_buf = sa.alloc[DType.uint8](16)
    assert_equal(len(sa._allocations), 3)
    f32_buf.enqueue_fill(Scalar[DType.float32](1.5))
    f64_buf.enqueue_fill(Scalar[DType.float64](2.5))
    u8_buf.enqueue_fill(UInt8(255))
    ctx.synchronize()
    with f32_buf.map_to_host() as h:
        assert_equal(h[0], Scalar[DType.float32](1.5))
    with f64_buf.map_to_host() as h:
        assert_equal(h[0], Scalar[DType.float64](2.5))
    with u8_buf.map_to_host() as h:
        assert_equal(h[0], UInt8(255))
    print("PASS")


def test_gpu_system_free_all(ctx: DeviceContext) raises:
    print("\n--- test_gpu_system_free_all ---")
    var sa = GPUSystemAllocator(DeviceContext(), 0)
    _ = sa.alloc[ftype](4)
    _ = sa.alloc[ftype](4)
    assert_equal(len(sa._allocations), 2)
    sa.free_all()
    assert_equal(len(sa._allocations), 0)
    _ = sa.alloc[ftype](4)
    assert_equal(len(sa._allocations), 1)
    print("PASS")


def test_gpu_system_zero(ctx: DeviceContext) raises:
    print("\n--- test_gpu_system_zero ---")
    var sa = GPUSystemAllocator(DeviceContext(), 0)
    var buf = sa.alloc[ftype](4)
    buf.enqueue_fill(99.0)
    ctx.synchronize()
    sa.zero()
    ctx.synchronize()
    assert_equal(len(sa._allocations), 1)  # zero doesn't free
    with buf.map_to_host() as host:
        print("After zero: host[0] =", host[0])
        assert_equal(host[0], sftype(0.0))
    print("PASS")


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    with DeviceContext() as ctx:
        test_gpu_arena_offsets(ctx)
        test_gpu_arena_free_all(ctx)
        test_gpu_arena_wipe(ctx)
        test_gpu_arena_zero(ctx)
        test_gpu_arena_mixed_dtype(ctx)
        test_gpu_arena_work_kernel(ctx)
        test_gpu_system_alloc_basic(ctx)
        test_gpu_system_multi_dtype(ctx)
        test_gpu_system_free_all(ctx)
        test_gpu_system_zero(ctx)
    print("\nAll GPU arena tests passed!")
