import std.os as os
from std.pathlib import Path
from std.memory import memcpy
from std.sys import size_of, stderr

from image import Image
from cpu.arena import CPUAllocator, CPUBumpArenaAllocator as Arena
from constants import ftype, sftype


# TODO: rename MNISTBatch — when used as a full-dataset view (getTrainBatch(0, 60000)) the name is
# misleading. Consider MNISTDataView or MNISTData, with a .slice(i, batch_size) -> MNISTBatch method
# to make the call site at the loop level explicit.
# TODO: add num_images() -> Int method = len(raw_labels), so callers don't need to know which span defines count.
# TODO: (future) if CPU hot path also moves off List[Image] AoS, both CPU and GPU can share these
# SoA spans directly. Image would become a lightweight view rather than an owned arena-backed struct.
@fieldwise_init
struct MNISTBatch[
    # is_mutable: Bool,
    # //,
    # origin: Origin[mut = is_mutable],
](Sized, TrivialRegisterPassable):
    var raw_pixels: Span[UInt8, ImmutAnyOrigin]
    var raw_labels: Span[UInt8, ImmutAnyOrigin]

    # TODO: remove or fix — this custom init is dead code. It memcpys Image.pixels but the count
    # was wrong (DataTensor size vs PixelLayout size) and nothing calls this path anymore.
    def __init__(
        out self,
        images: List[Image],
        pixel_buffer: Span[UInt8, MutAnyOrigin],
        label_buffer: Span[UInt8, MutAnyOrigin],
    ) raises:  # CANNOT take Some[CPUAllocator] - memory must be guaranteed to be contiguous. could also accept a mut Span, UnsafePointer, etc
        # comptime o = origin_of(a, b)
        comptime size = Image.PixelLayout.size()  # 784 raw bytes, NOT DataTensor (1024)
        var num_images = len(images)
        if pixel_buffer == label_buffer:
            raise Error("Buffers cannot be the same for MNISTBatch.")
        if len(pixel_buffer) < size * num_images:
            raise Error("Pixel buffer not large enough.")
        if len(label_buffer) < num_images:
            raise Error("Label buffer not large enough.")

        var offset = 0
        var pptr = pixel_buffer.unsafe_ptr()
        for i in range(num_images):
            var img = images[i]
            memcpy(src=img.pixels.ptr, dest=pptr + offset, count=size)
            label_buffer.unsafe_get(
                i
            ) = img.label  # we already bounds checked above
            offset += size
        self.raw_pixels = pixel_buffer
        self.raw_labels = label_buffer

    def __len__(self) -> Int:
        return len(self.raw_labels)


struct MNISTDataRepository:
    comptime COUNT_TRAIN = 60000
    comptime COUNT_TEST = 10000

    var _train_pixels_arena: Arena
    var _test_pixels_arena: Arena
    var _train_labels_arena: Arena
    var _test_labels_arena: Arena
    var test_data: List[
        Image
    ]  # Image is a label + a (fat LayoutTensor) pointer into the arena for the pixels
    var train_data: List[Image]

    var data_dir: Path
    var train_image_file: Path
    var train_label_file: Path
    var test_image_file: Path
    var test_label_file: Path

    def __init__(out self, data_dir: String = "data"):
        # setup paths
        self.data_dir = Path(data_dir)
        self.train_image_file = Path(data_dir + "/train-images-idx3-ubyte")
        self.train_label_file = Path(data_dir + "/train-labels-idx1-ubyte")
        self.test_image_file = Path(data_dir + "/t10k-images-idx3-ubyte")
        self.test_label_file = Path(data_dir + "/t10k-labels-idx1-ubyte")

        # setup storage and do the loading
        comptime image_size_in_bytes = Image.PixelLayout.size()  # 784, not DataTensor (1024)
        self._train_pixels_arena = Arena(image_size_in_bytes * Self.COUNT_TRAIN)
        self._train_labels_arena = Arena(size_of[UInt8]() * Self.COUNT_TRAIN)
        self._test_pixels_arena = Arena(image_size_in_bytes * Self.COUNT_TEST)
        self._test_labels_arena = Arena(size_of[UInt8]() * Self.COUNT_TEST)
        self.test_data = List[Image](capacity=Self.COUNT_TEST)
        self.train_data = List[Image](capacity=Self.COUNT_TRAIN)
        try:
            self._readTrainData()
            self._readTestData()
        except e:
            print(e, file=stderr)

    def _readTrainData(mut self) raises:
        """
        The span for the data we send in we'll fill with normalized images.
        """
        try:
            var data_file = open(self.train_image_file, "r")
            var label_file = open(self.train_label_file, "r")

            _ = data_file.seek(
                16, os.SEEK_SET
            )  # data has a magic header # 2049
            _ = label_file.seek(8, os.SEEK_SET)  # labels too # 2051

            comptime size = Image.PixelLayout.size()
            for c in range(Self.COUNT_TRAIN):
                var data_list = data_file.read_bytes(
                    size  # IMAGE_SIZE * IMAGE_SIZE
                )  # -> List[Byte]

                var temp = label_file.read_bytes(1)
                var data_label: UInt8 = temp[0]

                var img = Image(data_list, data_label, self._train_pixels_arena)
                self._train_labels_arena.buffer[c] = data_label
                self.train_data.append(img^)

            data_file.close()
            label_file.close()
        except e:
            raise Error(t"Error with input MNIST train binary files: {e}.")

    def _readTestData(mut self) raises:
        """
        The span for the data we send in we'll fill with normalized images.
        """
        try:
            var data_file = open(self.test_image_file, "r")
            var label_file = open(self.test_label_file, "r")

            _ = data_file.seek(
                16, os.SEEK_SET
            )  # data has a magic header # 2049
            _ = label_file.seek(8, os.SEEK_SET)  # labels too # 2051

            comptime size = Image.PixelLayout.size()
            for c in range(Self.COUNT_TEST):
                var data_list = data_file.read_bytes(
                    size  # IMAGE_SIZE * IMAGE_SIZE
                )  # -> List[Byte]

                var temp = label_file.read_bytes(1)
                var data_label: UInt8 = temp[0]

                var img = Image(data_list, data_label, self._test_pixels_arena)
                self.test_data.append(img^)
                self._test_labels_arena.buffer[c] = data_label

            data_file.close()
            label_file.close()
        except e:
            raise Error(t"Error with input MNIST test binary files: {e}.")

    # TODO: consider combining into one method for test and train and use arg to pick
    def getTrainBatch(self, start: Int, end: Int) -> MNISTBatch:
        """Get a view / span / slice from the arena directly as raw Bytes."""
        if end <= start or end > Self.COUNT_TRAIN:
            print(
                t"getTrainBatch error: invalid slice {start}:{end}).",
                file=stderr,
            )
        comptime image_size_in_bytes = Image.PixelLayout.size()  # 784 bytes/image
        var pixels_ptr_start = self._train_pixels_arena.buffer + (
            start * image_size_in_bytes
        )
        var count_bytes = (end - start) * image_size_in_bytes
        var pixels_span = Span(ptr=pixels_ptr_start, length=count_bytes)

        var labels_ptr_start = self._train_labels_arena.buffer + start
        var labels_span = Span(ptr=labels_ptr_start, length=(end - start))
        var batch = MNISTBatch(pixels_span, labels_span)
        return batch

    def getTestBatch(self, start: Int, end: Int) -> MNISTBatch:
        """Get a view / span / slice from the arena directly as raw Bytes."""
        if end <= start or end > Self.COUNT_TEST:
            print(
                t"getTestBatch error: invalid slice {start}:{end}).",
                file=stderr,
            )
        comptime image_size_in_bytes = Image.PixelLayout.size()  # 784 bytes/image
        var pixels_ptr_start = self._test_pixels_arena.buffer + (
            start * image_size_in_bytes
        )
        var count_bytes = (end - start) * image_size_in_bytes
        var pixels_span = Span(ptr=pixels_ptr_start, length=count_bytes)

        var labels_ptr_start = self._test_labels_arena.buffer + start
        var labels_span = Span(ptr=labels_ptr_start, length=(end - start))
        var batch = MNISTBatch(pixels_span, labels_span)
        return batch

    @deprecated("Use separate readTrainData and readTestData")
    def _readData(mut self, test_or_train: String) raises:
        """
        The span for the data we send in we'll fill with normalized images.
        """
        var data_filename = (
            self.test_image_file if test_or_train
            == "test" else self.train_image_file
        )
        var label_filename = (
            self.test_label_file if test_or_train
            == "test" else self.train_label_file
        )
        var cap = (
            Self.COUNT_TEST if test_or_train == "test" else Self.COUNT_TRAIN
        )

        try:
            var data_file = open(data_filename, "r")
            var label_file = open(label_filename, "r")

            _ = data_file.seek(16, os.SEEK_SET)  # data has an unknown header
            _ = label_file.seek(8, os.SEEK_SET)  # labels too

            comptime buffer_size = Image.PixelLayout.size()  # IMAGE_SIZE * IMAGE_SIZE
            var image_buffer = Image.PixelStorage(
                uninitialized=True
            )  # TODO: skip this InlineArray intermediate and send List[Byte]

            for c in range(cap):
                var data_list = data_file.read_bytes(
                    buffer_size
                )  # -> List[Byte]

                var temp = label_file.read_bytes(1)
                var data_label: UInt8 = temp[0]

                memcpy(
                    src=data_list.unsafe_ptr(),
                    dest=image_buffer.unsafe_ptr(),
                    count=materialize[Image.PixelLayout.size()](),
                )
                var img = Image(
                    image_buffer, data_label, self._train_pixels_arena
                )  # FIXME: this arena won't work
                if test_or_train == "test":
                    self.test_data.append(img^)
                else:
                    self.train_data.append(img^)

            data_file.close()
            label_file.close()
        except e:
            print("Error with input binary files:", e, file=stderr)

    # Shuffle helpers

    comptime seed_default = 69

    @staticmethod
    def _shuffleData(mut data: List[Image], seed: Int = Self.seed_default):
        """
        Not needed, but I / Claude wrote it just to play around and learn.
        """
        var count = len(data)
        if count < 1:
            return
        var rng_state = seed
        # some Claude 4 work
        for i in range(count - 1, 0, -1):
            rng_state = (rng_state * 1664525 + 1013904223) % 2147483647
            var j = Int(rng_state) % (i + 1)

            var temp = data[i]
            data[i] = data[j]
            data[j] = temp

    def shuffle(mut self, seed: Int = Self.seed_default):
        Self._shuffleData(self.train_data, seed)
        # Self.shuffleData(self.test_data, seed) # not sure why you'd want to
