from layout import Layout, LayoutTensor
from std.math import sqrt
from std.memory import memcpy
from std.sys import stderr, simd_width_of, size_of
from std.algorithm.functional import vectorize

from constants import (
    INPUT,
    IMAGE_SIZE,
    PADDING,
    ftype,
    sftype,
    nelts,
    FeatureLayouts,
)
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as Arena
from origin_util import untrack


struct Image(ImplicitlyCopyable):
    """A raw `UInt8` MNIST image plus its label. Normalization and padding into the
    network's `ftype` input are deferred to `normalized` — the target device isn't
    known at load time.
    """

    # explicit channel dim [C, H, W] — matches the [C, H, W] feature/weight
    # convention. C = INPUT (1 today); .size() is unchanged so all byte math holds.
    comptime PixelLayout = Layout.row_major(INPUT, IMAGE_SIZE, IMAGE_SIZE)
    comptime PixelStorage = InlineArray[  # TODO: remove this
        UInt8, Self.PixelLayout.size()
    ]  # raw bytes
    comptime PixelTensor = LayoutTensor[
        DType.uint8, Self.PixelLayout, MutUntrackedOrigin
    ]  # raw pixels

    # the padded, normalized image IS the network's feature input — same [1,32,32]
    # shape, so alias it rather than re-deriving (PADDED_SIZE == LENGTH_FEATURE0).
    comptime DataLayout = FeatureLayouts.input
    comptime DataTensor = LayoutTensor[
        ftype, Self.DataLayout, MutAnyOrigin
    ]  # normalized into ftype and padded

    var pixels: Self.PixelTensor
    var label: UInt8  # digits [0, 9] MNIST, could store as "Int"

    # TODO: test and use
    def __init__[
        o: Origin
    ](out self, hosted_pixels: Span[UInt8, o], label: UInt8) raises:
        """View an existing host span as pixels — no copy; caller owns the bytes.
        """
        comptime layout_size = Self.PixelLayout.size()
        if len(hosted_pixels) != layout_size:
            raise Error(
                t"Span[Byte] for Image has unexpected len:"
                t" {len(hosted_pixels)}."
            )
        if label > 9:
            raise Error(t"Error with image label: {label}.")
        self.label = label
        self.pixels = untrack(
            LayoutTensor[DType.uint8, Self.PixelLayout](hosted_pixels)
        )

    def __init__(
        out self, raw: List[Byte], label: UInt8, mut arena: Arena
    ) raises:
        """Copy `raw` into arena-owned storage; raises on wrong length or bad label.
        """
        comptime layout_size = Self.PixelLayout.size()
        if len(raw) != layout_size:
            raise Error(t"List[Byte] for Image has unexpected len: {len(raw)}.")
        if label > 9:
            raise Error(t"Error with image label: {label}.")
        self.label = label
        self.pixels = untrack(
            LayoutTensor[DType.uint8, Self.PixelLayout](
                arena.alloc[UInt8](layout_size)
            )
        )
        memcpy(src=raw.unsafe_ptr(), dest=self.pixels.ptr, count=layout_size)

    def __init__(
        out self, raw: Self.PixelStorage, label: UInt8, mut arena: Arena
    ):  # Some[CPUAllocator]):
        """Copy a fixed-size `PixelStorage` into arena-owned storage (non-raising).
        """
        comptime layout_size = Self.PixelLayout.size()

        if label > 9:
            print("Error with image label:", label, file=stderr)  # could raise
        self.label = label
        self.pixels = untrack(
            LayoutTensor[DType.uint8, Self.PixelLayout](
                arena.alloc[UInt8](layout_size)
            )
        )
        memcpy(src=raw.unsafe_ptr(), dest=self.pixels.ptr, count=layout_size)

        # no longer normalizing at init because we don't know the end device and that's a separate task

    def normalized[padded: Bool = True](self: Self, tensor: Self.DataTensor):
        """Standardize the pixels (per-image zero mean, unit std) into `tensor`.
        With `padded`, writes into the interior at a `PADDING` offset, leaving the
        border zero.
        """
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
                tensor[0, r + off, c + off] = ((curr - mean) / std).cast[
                    ftype
                ]()

    @deprecated("Use non-static self.normalized(output_tensor).")
    @staticmethod
    def _normalize[
        padded: Bool = True
    ](raw: Self.PixelStorage, tensor: Self.DataTensor):
        """Static predecessor of `normalized`, taking raw pixels instead of `self`.
        """
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
                tensor[0, r + off, c + off] = ((curr - mean) / std).cast[
                    ftype
                ]()
