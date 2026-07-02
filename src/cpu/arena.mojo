from layout import Layout, LayoutTensor

# from attention import Weights, ModelParams
from constants import ftype, sftype
from origin_util import untrack

# for allocation arena
from std.memory import memset_zero
from std.sys import stderr
from std.sys.info import size_of, align_of
from std.testing import assert_equal, TestSuite
from std.os import abort

comptime sitype = Scalar[DType.int32]  # for testing


trait ArenaSizable:
    """Structs that know their own byte footprint for arena allocation."""

    @staticmethod
    def sizeInBytes() -> Int:
        ...


trait CPUAllocator(ImplicitlyDeletable, Movable):
    """Uniform allocator interface — mirrors GPU `GPUAllocator` (System ignores capacity).
    """

    def __init__(out self, capacity_bytes: Int):
        # Bump: allocate the slab. System: ignore (sizes per alloc()).
        ...

    def alloc[
        T: AnyType
    ](mut self, count: Int) -> UnsafePointer[T, MutUntrackedOrigin]:
        ...

    def free_all(mut self):
        # Bump arena: resets offset so slab can be reused (no system free).
        # System allocator: calls free() on every tracked pointer.
        ...

    def zero(mut self):
        # Fill all allocated memory with 0 — offset/tracking unchanged.
        ...

    def wipe(mut self):
        # Bump: zero the slab + reset offset. System: zero then free_all.
        ...


struct CPUBumpArenaAllocator(CPUAllocator):
    # TODO: return Spans?
    """
    Simple bump allocator. Minor design help from Claude 4.5.
    """
    var buffer: UnsafePointer[UInt8, MutUntrackedOrigin]
    var capacity: Int
    var offset: Int

    def __init__(out self, capacity_bytes: Int):
        self.buffer = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
            alloc[UInt8](capacity_bytes)
        )
        self.capacity = capacity_bytes
        self.offset = 0

    def __del__(deinit self):
        self.buffer.free()

    def alloc[
        T: AnyType
    ](mut self, count: Int = 1) -> UnsafePointer[
        T, MutUntrackedOrigin
    ]:  # , origin_of(self)]:
        """Allocate space for `count` items of type T."""
        var size = size_of[T]() * count
        var alignment = align_of[T]()

        var aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1)

        if aligned_offset + size > self.capacity:
            # Could auto-grow here, or just panic
            abort("Arena out of memory! Aborting!")
            # raise Error("Arena out of memory!")

        var ptr = (self.buffer + aligned_offset).bitcast[T]()
        self.offset = aligned_offset + size

        # print("allocating", String(count), get_type_name[T](), "begin", Int(ptr), "->", Int(ptr + count))
        return ptr  # .unsafe_origin_cast[origin_of(self)]()

    def free_all(mut self):
        """Reset bump offset — slab stays alive, memory is reusable."""
        self.offset = 0

    def wipe(mut self):
        """Zero the slab and reset the offset. Arena-specific; not in trait."""
        memset_zero(self.buffer, self.capacity)
        self.offset = 0

    def zero(mut self):
        """Zero the whole slab; offset and live allocations are untouched."""
        memset_zero(self.buffer, self.capacity)


struct CPUSystemAllocator(CPUAllocator):
    """
    Pass-through to the system allocator. Tracks all allocations so
    free_all() / __del__ can bulk-free them. Mirrors the arena API so
    callsites are identical regardless of allocator choice.
    """

    var _allocations: List[UnsafePointer[UInt8, MutUntrackedOrigin]]
    var _sizes: List[Int]  # byte size of each tracked alloc — needed by zero()

    def __init__(out self, capacity_bytes: Int = 0):
        # capacity_bytes ignored — the system allocator sizes per alloc(). The
        # param exists so it's a drop-in for the bump arena in generic code.
        self._allocations = List[UnsafePointer[UInt8, MutUntrackedOrigin]]()
        self._sizes = List[Int]()

    def __del__(deinit self):
        self.free_all()

    def alloc[
        T: AnyType
    ](mut self, count: Int = 1) -> UnsafePointer[T, MutUntrackedOrigin]:
        """Allocate `count` items of `T`, tracking the pointer for later bulk free.
        """
        var ptr = alloc[T](count)
        self._allocations.append(
            rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
                ptr.bitcast[UInt8]()
            )
        )
        self._sizes.append(count * size_of[T]())
        return rebind[UnsafePointer[T, MutUntrackedOrigin]](ptr)

    def free_all(mut self):
        """Call free() on every tracked pointer."""
        for i in range(len(self._allocations)):
            self._allocations[i].free()
        self._allocations.clear()
        self._sizes.clear()

    def zero(mut self):
        """Zero every tracked allocation; keeps them allocated/tracked."""
        for i in range(len(self._allocations)):
            memset_zero(self._allocations[i], self._sizes[i])

    def wipe(mut self):
        """Zero then release all tracked allocations."""
        self.zero()
        self.free_all()


def main() raises:
    """Tests here. Some reflection examples to start if you `uncomment`."""
    _ = """
    printTypeInfo[ftype]()
    printTypeInfo[itype]()
    print("Some reflection tests...")
    comptime test_weights = TestWeights()
    comptime T = type_of(test_weights)
    printFields[T]()
    print("Arena time...")
    printFields[ModelParams]()
    test_nested_arena()
    """

    var suite = TestSuite()
    suite.test[test_allocator_offsets]()
    # suite.test[test_allocation_failure]() # now panics/aborts instead of raises
    suite.test[test_allocator_wipe]()
    suite.test[test_allocator_free_all]()
    suite.test[test_system_alloc_basic]()
    suite.test[test_system_alloc_multi_type]()
    suite.test[test_system_free_all_clears]()
    suite.test[test_system_alloc_write_read]()
    suite^.run()


def test_allocator_offsets() raises:
    var size_in_bytes = 12 * size_of[sftype]() + size_of[sitype]() * 3
    var c_arena = CPUBumpArenaAllocator(size_in_bytes)
    try:
        var p0 = c_arena.alloc[sftype](5)
        var p1 = c_arena.alloc[sftype](7)
        # DIFFERENT TYPE BEING ALLOCATED
        var p2 = c_arena.alloc[sitype](3)
        var end = p2 + 3  # pointer arithmetic scales by element size

        var size0 = Int(p1) - Int(p0)
        var size1 = Int(p2) - Int(p1)
        var size2 = Int(end) - Int(p2)

        assert_equal(size0, 20)  # 5 float32
        assert_equal(size1, 28)
        assert_equal(size2, 12)
    except e:
        print(e)
        assert_equal(0, -1)


@deprecated("alloc now aborts instead of raises")
def test_allocation_failure() raises:
    # var arena = CPUBumpArenaAllocator(5)
    # try:
    #    var ptr = arena.alloc[sftype](10)
    # except e:
    #    _ = e
    #    assert_equal(0, 0)
    assert_equal(0, 0)


def test_allocator_wipe() raises:
    var arena = CPUBumpArenaAllocator(128)
    var ptr = arena.alloc[sitype](10)
    for i in range(10):
        ptr[i] = 69
    arena.wipe()
    for i in range(10):
        assert_equal(ptr[i], 0)


def test_allocator_free_all() raises:
    var arena = CPUBumpArenaAllocator(128)
    var ptr0 = arena.alloc[UInt8](128)
    arena.free_all()
    var ptr1 = arena.alloc[UInt8](128)
    assert_equal(ptr0, ptr1)


def test_system_alloc_basic() raises:
    var sa = CPUSystemAllocator()
    var ptr = sa.alloc[sitype](4)
    assert_equal(len(sa._allocations), 1)
    ptr[0] = 42
    assert_equal(ptr[0], 42)


def test_system_alloc_multi_type() raises:
    var sa = CPUSystemAllocator()
    var p0 = sa.alloc[sftype](5)
    var p1 = sa.alloc[sitype](3)
    var p2 = sa.alloc[UInt8](16)
    assert_equal(len(sa._allocations), 3)
    p0[0] = 1.5
    p1[0] = 99
    p2[0] = 255
    assert_equal(p0[0], 1.5)
    assert_equal(p1[0], 99)
    assert_equal(p2[0], 255)


def test_system_free_all_clears() raises:
    var sa = CPUSystemAllocator()
    _ = sa.alloc[sitype](8)
    _ = sa.alloc[sftype](8)
    assert_equal(len(sa._allocations), 2)
    sa.free_all()
    assert_equal(len(sa._allocations), 0)
    # can allocate again after free_all
    _ = sa.alloc[sitype](4)
    assert_equal(len(sa._allocations), 1)


def test_system_alloc_write_read() raises:
    var sa = CPUSystemAllocator()
    var ptr = sa.alloc[sftype](10)
    for i in range(10):
        ptr[i] = Float32(i) * 0.5
    for i in range(10):
        assert_equal(ptr[i], Float32(i) * 0.5)


def test_nested_arena():
    var tc = TestContainer()  # allocates Arena itself
    var tw_arena = CPUBumpArenaAllocator(7 * size_of[sftype]())
    var tw = TestWeights(tw_arena)

    print(tw.a.ptr, tc.a.ptr, tc.sub_weights.a.ptr)


struct TestWeights:
    comptime layout = Layout.row_major(7)
    var a: LayoutTensor[ftype, Self.layout, MutUntrackedOrigin]

    def __init__(out self, mut arena: CPUBumpArenaAllocator):
        self.a = untrack(
            LayoutTensor[ftype, Self.layout](
                arena.alloc[sftype](comptime (self.layout.size()))
            ).fill(3.0)
        )

    @staticmethod
    def sizeInBytes() -> Int:
        return comptime (Self.layout.size()) * size_of[ftype]()

    @staticmethod
    def initRandom(
        out self: Self, mut arena: CPUBumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        _ = self.a.fill(1.0)

    def freeMemory(mut self):
        self.a.ptr.free()


struct TestContainer:
    var arena: CPUBumpArenaAllocator

    comptime layout = Layout.row_major(5)
    var a: LayoutTensor[ftype, Self.layout, MutUntrackedOrigin]
    var sub_weights: TestWeights

    def __init__(out self):
        self.arena = type_of(self.arena)(
            self.sizeInBytes() + TestWeights.sizeInBytes()
        )
        self.a = untrack(
            LayoutTensor[ftype, Self.layout](
                self.arena.alloc[sftype](comptime (self.layout.size()))
            ).fill(1.0)
        )
        self.sub_weights = TestWeights.initRandom(self.arena)

    @staticmethod
    def sizeInBytes() -> Int:
        return comptime (Self.layout.size()) * size_of[sftype]()

    def __del__(deinit self):
        pass
        # self.a.ptr.free()
        # self.sub_weights.a.ptr.free()
