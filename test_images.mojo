from std.python import Python, PythonObject
from std.random import random_ui64

from dataloader import MNISTDataRepository
from lenet import PADDED_SIZE


# CLAUDE CODE VIBE CODED FILE 4.7 OPUS I THINK?
def main() raises:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    # NOTE: MNISTDataRepository loads all 70k images on init — takes a few seconds
    print("Loading MNIST data...")
    var repo = MNISTDataRepository()
    print("Done. Displaying images.")

    # A handful of random test-set indices to spot-check
    comptime n = 5
    var indices: List[Int] = []
    for _ in range(n):
        indices.append(Int(random_ui64(0, MNISTDataRepository.COUNT_TEST - 1)))

    var figsize = Python.tuple(n * 3, 4)
    var fig = plt.figure(figsize=figsize)

    for i in range(n):
        var idx = indices[i]
        var img = repo.test_data[idx]

        # Build a flat Python list from the 32x32 normalized float tensor, then reshape
        var flat: PythonObject = []
        for r in range(PADDED_SIZE):
            for c in range(PADDED_SIZE):
                flat.append(img.pixels[r, c])

        var np_img = np.array(flat).reshape(PADDED_SIZE, PADDED_SIZE)

        var ax = fig.add_subplot(1, n, i + 1)
        ax.imshow(np_img, cmap="gray")
        ax.set_title("Label: " + String(img.label))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("test_images.png")
    print("Saved to test_images.png")
    plt.show()
