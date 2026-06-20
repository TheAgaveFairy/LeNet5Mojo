from layout import Layout, LayoutTensor
from std.math import sqrt
from std.memory import memcpy
from std.sys import stderr, simd_width_of, size_of
from std.algorithm.functional import vectorize

from constants import IMAGE_SIZE, PADDED_SIZE, PADDING, ftype, sftype, nelts
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as Arena


struct Image(ImplicitlyCopyable):
    """
    We need the images normalized and possibly padded from raw UInt8.
    """

    comptime PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
    comptime PixelStorage = InlineArray[ # TODO: remove this
        UInt8, Self.PixelLayout.size()
    ]  # raw bytes
    comptime PixelTensor = LayoutTensor[
        DType.uint8, Self.PixelLayout, MutAnyOrigin
    ]  # raw pixels

    comptime DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
    comptime DataTensor = LayoutTensor[
        ftype, Self.DataLayout, MutAnyOrigin
    ]  # normalized into ftype and padded

    var pixels: Self.PixelTensor
    var label: UInt8  # digits [0, 9] MNIST, could store as "Int"

    # TODO: test and use
    def __init__(out self, hosted_pixels: Span[UInt8, _], label: UInt8) raises:
        if len(hosted_pixels) != layout_size:
            raise Error(t"Span[Byte] for Image has unexpected len: {len(raw)}.")
        if label > 9:
            raise Error(t"Error with image label: {label}.")
        self.label = label
        self.pixels = Self.PixelTensor(hosted_pixels)

    def __init__(
        out self, raw: List[Byte], label: UInt8, mut arena: Arena
    ) raises:
        comptime layout_size = Self.PixelLayout.size()
        if len(raw) != layout_size:
            raise Error(t"List[Byte] for Image has unexpected len: {len(raw)}.")
        if label > 9:
            raise Error(t"Error with image label: {label}.")
        self.label = label
        self.pixels = Self.PixelTensor(arena.alloc[UInt8](layout_size)) 
        memcpy(src=raw.unsafe_ptr(), dest=self.pixels.ptr, count=layout_size)

    def __init__(
        out self, raw: Self.PixelStorage, label: UInt8, mut arena: Arena
    ):  # Some[CPUAllocator]):
        comptime layout_size = Self.PixelLayout.size()

        if label > 9:
            print("Error with image label:", label, file=stderr)  # could raise
        self.label = label
        self.pixels = Self.PixelTensor(arena.alloc[UInt8](layout_size))
        memcpy(src=raw.unsafe_ptr(), dest=self.pixels.ptr, count=layout_size)

        # no longer normalizing at init because we don't know the end device and that's a separate task

    def normalized[padded: Bool = True](self: Self, tensor: Self.DataTensor):
        var sum: UInt64 = 0
        var std_sum: UInt64 = 0

        comptime N = Self.PixelTensor.layout.size()

        var temp_buffer = InlineArray[UInt64, N](uninitialized=True)
        comptime for i in range(N):
            temp_buffer[i] = UInt64(self.pixels.ptr[i])

        def sum_closure[width: Int](i: Int) {mut}:
            var nums = temp_buffer.unsafe_ptr().load[width=width](i)
            sum += nums.reduce_add()
            std_sum += (nums * nums).reduce_add()

        vectorize[nelts](N, sum_closure)

        var mean = Float64(sum) / Float64(N)
        var temp = Float64(std_sum) / Float64(N) - (mean * mean)
        var std = sqrt(temp)

        comptime off = PADDING if padded else 0
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                var curr = Float64(Int(self.pixels.ptr[idx]))
                tensor[r + off, c + off] = ((curr - mean) / std).cast[ftype]()

    @deprecated("Use non-static self.normalized(output_tensor).")
    @staticmethod
    def _normalize[
        padded: Bool = True
    ](raw: Self.PixelStorage, tensor: Self.DataTensor):
        var sum: UInt64 = 0
        var std_sum: UInt64 = 0

        comptime N = raw.size

        var temp_buffer = InlineArray[UInt64, raw.size](uninitialized=True)
        comptime for i in range(raw.size):
            temp_buffer[i] = UInt64(raw[i])

        def sum_closure[width: Int](i: Int) {mut}:
            var nums = temp_buffer.unsafe_ptr().load[width=width](i)
            sum += nums.reduce_add()
            std_sum += (nums * nums).reduce_add()

        vectorize[nelts](N, sum_closure)

        var mean = Float64(sum) / Float64(N)
        var temp = Float64(std_sum) / Float64(N) - (mean * mean)
        var std = sqrt(temp)

        comptime off = PADDING if padded else 0
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                var curr = Float64(Int(raw[idx]))
                tensor[r + off, c + off] = ((curr - mean) / std).cast[ftype]()
