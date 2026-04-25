# for reflection
from layout import Layout, LayoutTensor
from std.reflection import (
    struct_field_types,
    struct_field_names,
    struct_field_count,
    get_type_name,
)

# from attention import Weights, ModelParams
from constants import ftype, sftype

# for allocation arena
from std.memory import memset_zero
from std.sys import stderr
from std.sys.info import size_of, align_of
from std.testing import assert_equal, TestSuite
from std.os import abort


trait CPUAllocator(Copyable, ImplicitlyCopyable):
    def __init__(out self, capacity_bytes: Int):
        ...

    def alloc[
        T: AnyType
    ](mut self, count: Int) -> UnsafePointer[T, MutAnyOrigin]:
        ...

    def reset(mut self):
        ...

    def clear(mut self):
        ...


struct CPUBumpArenaAllocator(CPUAllocator):
    #comptime __copyinit__is_trivial = True
    #comptime __moveinit__is_trivial = True
    # TODO: return Spans?
    """
    Simple bump allocator. Minor design help from Claude 4.5.
    """
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var capacity: Int
    var offset: Int

    def __init__(out self, capacity_bytes: Int):
        self.buffer = alloc[UInt8](capacity_bytes)
        self.capacity = capacity_bytes
        self.offset = 0

    def __init__(
        out self, capacity_bytes: Int, extra_space_factor: Float64
    ) raises:
        if extra_space_factor < 0.0:
            print("ARENA INIT ERROR! extra_space_factor < 0.0", file=stderr)
            raise Error("Arena init error. extra_space_factor < 0.0")

        var expanded_size = capacity_bytes + Int(
            Float64(capacity_bytes) * extra_space_factor
        )
        return Self(expanded_size)

        # self.buffer = alloc[UInt8](expanded_size)
        # self.capacity = capacity_bytes
        # self.offset = 0

    def __del__(deinit self):
        # TODO: I think we can reinstate the self-clearing
        # print("BumpArenaAllocator __del__()"
        self.buffer.free()
        pass

    def alloc[
        T: AnyType
    ](mut self, count: Int = 1) -> UnsafePointer[T, MutAnyOrigin]:
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
        return ptr

    def reset(mut self):
        """Free all allocations at once by resetting the offset."""
        self.offset = 0
        # Memory is still there, just reusable

    def clear(mut self):
        """Reset and zero out memory."""
        memset_zero(self.buffer, self.capacity)
        self.offset = 0


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
    suite.test[test_allocator_clear]()
    suite.test[test_allocator_reset]()
    suite^.run()


def printFields[T: AnyType]():
    """Testing new reflection features."""
    print(get_type_name[T](), "has fields:")
    comptime f_types = struct_field_types[T]()
    comptime f_names = struct_field_names[T]()

    # @parameter
    comptime for i in range(struct_field_count[T]()):
        print("\t", materialize[f_names[i]](), ":", get_type_name[f_types[i]]())


def printTypeInfo[T: DType]():
    """Prints type name, size, and alignment."""
    comptime thing = "{}:\n\tsize: {}, align {}".format(
        T, size_of[Scalar[T]](), align_of[Scalar[T]]()
    )
    print(thing)


def test_allocator_offsets() raises:
    var size_in_bytes = 12 * size_of[sftype]() + size_of[sitype]() * 3
    var c_arena = CPUBumpArenaAllocator(size_in_bytes)
    try:
        var p0 = c_arena.alloc[sftype](5)
        var p1 = c_arena.alloc[sftype](7)
        # DIFFERENT TYPE BEING ALLOCATED
        var p2 = c_arena.alloc[sitype](3)
        var end = p2 + 3 * size_of[sitype]()

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
    # var arena = BumpArenaAllocator(5)
    # try:
    #    var ptr = arena.alloc[sftype](10)
    # except e:
    #    _ = e
    #    assert_equal(0, 0)
    assert_equal(0, 0)


def test_allocator_clear() raises:
    var arena = CPUBumpArenaAllocator(128)
    var ptr = arena.alloc[sitype](10)
    for i in range(10):
        ptr[i] = 69
    arena.clear()
    for i in range(10):
        assert_equal(ptr[i], 0)


def test_allocator_reset() raises:
    var arena = CPUBumpArenaAllocator(128)
    var ptr0 = arena.alloc[UInt8](128)
    arena.reset()
    var ptr1 = arena.alloc[UInt8](128)
    assert_equal(ptr0, ptr1)


def test_nested_arena():
    var tc = TestContainer()  # allocates Arena itself
    var tw_arena = CPUBumpArenaAllocator(7 * size_of[sftype]())
    var tw = TestWeights(tw_arena)

    print(tw.a.ptr, tc.a.ptr, tc.sub_weights.a.ptr)


struct TestWeights(Weights):
    var arena: CPUBumpArenaAllocator

    comptime layout = Layout.row_major(7)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]

    def __init__(out self, arena: BumpArenaAllocator):
        self.arena = arena
        self.a = type_of(self.a)(
            self.arena.alloc[sftype](comptime (self.layout.size()))
        ).fill(3.0)

    @staticmethod
    def sizeInBytes() -> Int:
        return comptime (Self.layout.size()) * size_of[ftype]()

    @staticmethod
    def initRandom(
        out self: Self, arena: CPUBumpArenaAllocator, std: Float64 = 0.02
    ):
        self = Self(arena)
        _ = self.a.fill(1.0)

    def freeMemory(mut self):
        self.a.ptr.free()


struct TestContainer:
    var arena: CPUBumpArenaAllocator

    comptime layout = Layout.row_major(5)
    var a: LayoutTensor[ftype, Self.layout, MutAnyOrigin]
    var sub_weights: TestWeights

    def __init__(out self):
        self.arena = type_of(self.arena)(
            self.sizeInBytes() + TestWeights.sizeInBytes()
        )
        self.a = type_of(self.a)(
            self.arena.alloc[sftype](comptime (self.layout.size()))
        ).fill(1.0)
        self.sub_weights = TestWeights.initRandom(self.arena)

    @staticmethod
    def sizeInBytes() -> Int:
        return comptime (Self.layout.size()) * size_of[sftype]()

    def __del__(deinit self):
        pass
        # self.a.ptr.free()
        # self.sub_weights.a.ptr.free()
