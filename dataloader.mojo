import std.os as os
from std.pathlib import Path
from std.memory import memcpy
from std.sys import size_of, stderr

from image import Image
from arena import Allocator, BumpArenaAllocator as Arena
from lenet import ftype, sftype

struct MNISTDataRepository():
    comptime COUNT_TRAIN =     60000
    comptime COUNT_TEST =      10000

    var pixel_arena: Arena
    var test_data: List[Image] # Image is a label + a (fat LayoutTensor) pointer into the arena for the pixels
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
        comptime image_size_in_bytes = size_of[ftype]() * Image.DataTensor.layout.size()
        self.pixel_arena = Arena(image_size_in_bytes * (Self.COUNT_TRAIN + Self.COUNT_TEST))

        self.test_data = List[Image](capacity = Self.COUNT_TEST)
        self.train_data = List[Image](capacity = Self.COUNT_TRAIN)
        try:
            self._readData("test")
            self._readData("train")
        except e:
            print(e, file = stderr)

    def _readData(mut self, test_or_train: String) raises:
        """
        The span for the data we send in we'll fill with normalized images.
        """
        var data_filename = self.test_image_file if test_or_train == "test" else self.train_image_file
        var label_filename = self.test_label_file if test_or_train == "test" else self.train_label_file
        var cap = Self.COUNT_TEST if test_or_train == "test" else Self.COUNT_TRAIN

        try:
            var data_file = open(data_filename, "r")
            var label_file = open(label_filename, "r")
        
            _ = data_file.seek(16, os.SEEK_SET)    # data has an unknown header
            _ = label_file.seek(8, os.SEEK_SET)  # labels too

            comptime buffer_size = Image.PixelLayout.size() #IMAGE_SIZE * IMAGE_SIZE
            var image_buffer = Image.PixelStorage(uninitialized=True) # TODO: skip this InlineArray intermediate and send List[Byte]
            
            for c in range(cap):
                var data_list = data_file.read_bytes(buffer_size) # -> List[Byte]
                
                var temp = label_file.read_bytes(1)
                var data_label: UInt8 = temp[0]

                memcpy(src = data_list.unsafe_ptr(), dest = image_buffer.unsafe_ptr(), count = materialize[Image.PixelLayout.size()]())
                var img = Image(image_buffer, data_label, self.pixel_arena)
                if test_or_train == "test":
                    self.test_data.append(img^)
                else:
                    self.train_data.append(img^)

            data_file.close()
            label_file.close()
        except e:
            print("Error with input binary files:", e, file = stderr)

    @staticmethod
    def shuffleData(mut data: Span[Image, MutAnyOrigin], seed: Int = 69):
        """
        Not needed, but I / Claude wrote it just to play around and learn.
        """
        var count = len(data)
        if count < 1:
            return
        var rng_state = seed
        #some Claude 4 work
        for i in range(count - 1, 0, -1):
            rng_state = (rng_state * 1664525 + 1013904223) % 2147483647
            var j = Int(rng_state) % (i + 1)

            var temp = data[i]
            data[i] = data[j]
            data[j] = temp
