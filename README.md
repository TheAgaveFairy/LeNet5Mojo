# LeNet5Mojo: LeNet-5 from Scratch in Mojo🔥

A high-performance LeNet-5 Convolutional Neural Network built entirely from scratch in [Mojo🔥](https://www.modular.com/mojo) with custom CPU and GPU kernels — featuring compile-time swappable activation functions, custom arena allocators, and the Modular/MAX ecosystem throughout.

## Project Motivation

- **Learn CNNs from first principles** — every component hand-rolled: forward pass, backpropagation, weight updates, cross-entropy loss, softmax
- **Explore the Modular/MAX ecosystem** — systems programming for AI, using `layout` for compile-time tensor shapes and `std.gpu` for GPU kernel dispatch
- **Build custom GPU kernels** without CUDA C, PTX, or external ML libraries
- **Custom memory management** — bump arena allocators on both CPU and GPU, pre-allocating one slab per run and sub-allocating into it
- **Compile-time activation functions** — switch between ReLU, GELU, GELUTanh, GELUFast, Sigmoid, and Tanh with a compiler flag; no runtime branching

## Performance Highlights

Streaming MNIST inference on an RTX 3070, full 10,000-image test set:

- **~1.17M images/sec** peak GPU throughput — faster than PyTorch (eager, tuned, and `torch.compile`) and ONNX Runtime CUDA, and within ~1.5% of JAX/XLA
- **~7× faster than MAX Engine** on this model — hand-written Mojo kernels beat Modular's own graph compiler at small-CNN inference
- Even **single-stream** (no concurrency overlap), the custom kernels hit **~912k img/s** — still ahead of PyTorch eager
- **Matched accuracy** — 96.48% (9648/10000), identical to the PyTorch / JAX / ONNX reference models
- **Custom GPU kernels** written in pure Mojo (no CUDA C, cuDNN, or BLAS), supporting NVidia, AMD, and Apple

*Only ONNX-RT + TensorRT and the compute-only ("data resident in VRAM") variants lead; see [Performance Comparison](#performance-comparison) for the full table and methodology.*

## Architecture

Modified LeNet-5 on MNIST:

```
Input (1×32×32, zero-padded from 28×28)
  → Conv1 (1→6 ch, 5×5) → Act* → MaxPool (2×2)
  → Conv2 (6→16 ch, 5×5) → Act* → MaxPool (2×2)
  → Conv3 (16→120 ch, 5×5) → Act*
  → FC (120→10)
  → Logits
```

*`Act` is compile-time selectable — ReLU by default.*

The 84-unit penultimate layer and some skip connections from the original LeCun et al. paper are intentionally omitted, for consistency with a [prior C/CUDA implementation](https://github.com/TheAgaveFairy/LeNet-5).

## Project Structure

```
├── main.mojo              # Entry point: CPU train + test, GPU inference, logging
├── constants.mojo         # Architecture dims, dtype, compile-time activation selection
├── activation_fn.mojo     # ActivationFunction trait + ReLU, GELU, GELUTanh, GELUFast, Sigmoid, Tanh
├── dataloader.mojo        # MNISTDataRepository, MNISTBatch (SoA arena view)
├── image.mojo             # Image struct: raw UInt8 pixels + per-image normalization
├── resultlogger.mojo      # CSV logging for training epochs and inference results
├── cpu/
│   ├── arena.mojo         # CPUBumpArenaAllocator, CPUSystemAllocator + unit tests
│   ├── model.mojo         # LeNet5 and Feature structs (weights + intermediates)
│   └── ops.mojo           # Forward, backward, parallel training, parallel testing
├── accel/
│   ├── arena.mojo         # GPUBumpArenaAllocator, GPUSystemAllocator + unit tests
│   ├── feature.mojo       # FeatureGPU / FeatureGPUBuffers (per-image intermediate buffers)
│   ├── model.mojo         # LeNet5GPU, LeNet5GPUBuffers, DeviceSession
│   └── ops.mojo           # All GPU kernels and the batched inference pipeline
├── data/
│   └── *-ubyte            # MNIST binary files (standard IDX format)
└── models/
    └── model*.dat         # Pre-trained model weights
```

## Technical Implementation

### The Modular / MAX Ecosystem

This project uses Mojo's first-party ecosystem throughout — not just the core language:

- **`layout`** — compile-time tensor shape descriptions (`Layout`, `LayoutTensor`, `row_major`) for zero-overhead dimension tracking across all layers, on both CPU and GPU
- **`std.gpu`** — `DeviceContext`, `DeviceBuffer`, kernel launch via `enqueue_function`, `barrier`, thread indexing (`global_idx`, `block_idx`, `thread_idx`)
- **`std.algorithm`** — `vectorize` for SIMD-width loops, `parallelize` for multi-threaded training and testing

All operations are hand-rolled in Mojo. No PyTorch, TensorFlow, JAX, or BLAS.

### Compile-Time Activation Functions

The activation function is selected at **compile time** via a `-D` flag — zero runtime cost, no virtual dispatch:

```bash
mojo main.mojo               # ReLU (default)
mojo -D GELU main.mojo       # Exact GELU (erf-based)
mojo -D GELUTanh main.mojo   # GELU tanh approximation
mojo -D GELUFast main.mojo   # Fast GELU (sigmoid-based, ~Swish)
mojo -D Sigmoid main.mojo    # Sigmoid
mojo -D Tanh main.mojo       # Tanh
```

Each activation implements `forward`, `backward`, `simdForward`, and `simdBackward`. CPU operations use the layout-level SIMD-vectorized versions; GPU kernels call `simdForward` directly per-element in shared memory.

**The right learning rate is activation-dependent** — the same `-D ALPHA=N` (learning rate `N/1000`) that trains one activation can stall another. For example, ReLU alone swings from **8726/10000** at `ALPHA=50` to **9721/10000** at `ALPHA=1000` (quick single-run trains); GELUFast peaks in a different range. ⚠️ **Sigmoid collapses to ~10–18%** under the default short schedule regardless of `ALPHA` — presumably vanishing gradients — a reminder that some activations need a different setup entirely, not just a retuned learning rate. See [`docs/activation_tuning.md`](docs/activation_tuning.md) for suggested per-activation defaults.

### Arena Allocators

Both CPU and GPU use custom bump arena allocators to avoid per-tensor allocation overhead:

- **`CPUBumpArenaAllocator`** — one pre-allocated heap arena; model weights and all intermediate `Feature` buffers sub-allocate from it. `wipe()` zeroes and resets; no individual frees during training.
- **`GPUBumpArenaAllocator`** — same pattern on GPU: one `DeviceBuffer[uint8]` backing slab, typed sub-buffers via `create_sub_buffer` with alignment padding. The full batch of `FeatureGPUBuffers` for an inference run comes from a single arena.
- **`CPUSystemAllocator` / `GPUSystemAllocator`** — drop-in alternatives that call the system allocator per-request; same interface, useful for profiling or one-off allocations.
- **`[CPU/Device]Session`** — ties arena, weight buffers, and `LeNet5[GPU]` view lifetimes together; no manual lifetime juggling at the call site.

### GPU Kernels

All GPU computation is written in plain Mojo `def` functions:

| Kernel | What it does |
|--------|-------------|
| `normalizeInputsKernel` | Fused H2D: per-image mean/std computed in shared memory, normalize into padded feature buffer |
| `conv1FusedKernel` | Conv (1→6 ch) with bias + activation; weights cached in shared memory |
| `conv2FusedKernel` | Conv (6→16 ch); channel divisions to fit thread block resource limits |
| `conv3FusedKernel` | Conv (16→120 ch); reduction across 16×5×5 inputs per output channel |
| `maxPool1Kernel` / `maxPool2Kernel` | 2×2 max pooling |
| `matMulFusedKernel` | FC layer (120→10) as a parallel tree reduction in shared memory |

Kernels use `LayoutTensor` for type-safe indexing and `comptime for` for inner-loop unrolling at compile time.

### CPU Training

- **Multi-threaded** via `parallelize` — forward + backward per sample runs in parallel within each batch; weight deltas accumulated after
- **SIMD-vectorized** weight accumulation, cross-entropy loss, and all activation functions via `vectorize`
- **Numerically stable** softmax (max-subtraction trick) and cross-entropy loss

### Model Serialization

`LeNet5.saveToFile` / `loadFromFile` write weights as raw binary with big-endian byte-swapping. `loadFromFile[filetype]` supports loading weights saved in a different float precision than the current runtime model (e.g. load `float64` weights into a `float32` model).

## Getting Started

### Prerequisites

- [Pixi](https://prefix.dev/) package manager (manages Mojo + dependencies)
- NVIDIA or AMD or Apple GPU (optional — CPU-only works without one)
- MNIST dataset files in `data/` (standard IDX format, download from [Yann LeCun's site](http://yann.lecun.com/exdb/mnist/))

### Installation

```bash
git clone <repo-url>
cd LeNet5Mojo
pixi shell
```

### Running

```bash
# Train on CPU, then run GPU inference (ReLU, alpha=0.5, batch_size=50)
mojo main.mojo

# Compile-time options
mojo -D GELU -D ALPHA=300 main.mojo      # GELU activation, alpha=0.3
mojo -D BATCH_SIZE=100 main.mojo         # GPU inference batch size 100
mojo -D DISPLAY main.mojo               # Show training progress bars

# Build an optimized binary
pixi run build && ./main

# Run arena unit tests
pixi run test-cpu-arena
pixi run test-gpu-arena

# Format all source
pixi run formatall
```

## Performance Comparison

Benchmarked against PyTorch, JAX, ONNX Runtime (incl. TensorRT), MAX Engine, and NumPy on the full **10,000-image MNIST test set**. Throughput is the median over the set — **higher is better**.

### GPU — streaming inference (peak throughput)

| Framework | Backend | Throughput (img/s) | Latency (ms) | Accuracy |
|-----------|---------|-------------------:|-------------:|:--------:|
| ONNX-RT + TensorRT | GPU | 1,428,778 | 7.0 | 96.49% |
| JAX | XLA | 1,187,601 | 8.4 | 96.48% |
| **LeNet5Mojo** | **GPU, 5 streams** | **1,169,481** | **8.0** | **96.48%** |
| PyTorch (`torch.compile`) | GPU | 1,032,441 | 9.7 | 96.48% |
| **LeNet5Mojo** | **GPU, single stream** | **911,578** | **10.0** | **96.48%** |
| ONNX Runtime | GPU | 888,744 | 11.3 | 96.48% |
| PyTorch (eager) | GPU | 851,311 | 11.7 | 96.48% |
| MAX Engine | GPU | 161,045 | 57.2 | 96.5%* |

*All implementations reach the same 96.48% accuracy; \* MAX Engine's large-batch runs drop the remainder images.*

**Methodology / fairness notes:**
- **Streaming scenario** (the apples-to-apples default): images are copied host→device *inside* the timed loop. LeNet5Mojo uploads raw `uint8` and normalizes on-GPU — ~4× less PCIe traffic than the libraries' pre-normalized `fp32`.
- *5 streams* = ping-pong overlap of the H2D copy with compute; *single stream* matches the single-stream libraries for a stricter comparison (and still beats PyTorch eager).
- **Compute-only** ("dataset already resident in VRAM") numbers — where JAX-resident (1.50M img/s) and PyTorch-resident (1.09M) pull ahead — are tracked separately. A resident mode for LeNet5Mojo is in progress; until then it isn't in the headline table.

### CPU

The CPU path (`parallelize` + SIMD, hand-rolled) reaches **~27k img/s** — honest about it: this trails vendor-tuned CPU backends like ONNX Runtime / MLAS (~474k img/s) and PyTorch (~166k). CPU was not the optimization focus; the GPU kernels are where the work went.

*AMD Ryzen 7600X / NVidia RTX 3070 8GB. `fp32`, full 10k test set, median of repeated runs.*

## Current Limitations

- GPU **training** not implemented — inference only
- Batch size must divide the dataset evenly; remainder images are currently dropped (default `bs=50` divides both 10k and 60k). Arbitrary batch sizes need tail-padding (planned).
- Stream count (`--num-streams`) is hand-tuned per batch size to hit peak — no auto-default yet
- CPU inference throughput trails vendor-optimized CPU backends (ONNX Runtime / MLAS)
- Only `fp32` / `fp64` paths exercised; `fp16` / `bf16` untested
- `conv1` kernel hardcodes single-channel input (fine for MNIST, breaks if extended)

## Planned Improvements

- **Compute-only / resident benchmark mode** — preload the full test set to VRAM for a pure kernel-vs-kernel comparison against XLA / cuDNN / TensorRT (removes Mojo's raw-`uint8` transfer edge)
- **Tail-padding** so any batch size covers the full test set (pad the last partial batch, mask padded slots when tallying)
- **Auto-heuristic for stream count** derived from the batch size, so good numbers come out-of-the-box without a grid search
- **conv3 Tier B** — tiled GEMM that fills the GPU in a single launch, reducing reliance on high stream counts for occupancy
- **`fp16` / `bf16` dtype paths** + dtype-parity notes vs PyTorch/JAX (which default to TF32 matmuls on Ampere)
- Pre-normalized `fp32` input mode to mirror how the libraries upload images

## Contributing

Educational project — suggestions and discussion on Mojo patterns or optimization techniques are welcome.

## Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) and the [MAX platform](https://www.modular.com/max) by [Modular](https://www.modular.com/)
- MNIST dataset by Yann LeCun et al.
- Inspired by the original [LeNet-5 paper](http://yann.lecun.com/exdb/lenet/) by Y. LeCun et al.
- Prior C/CUDA implementation: [TheAgaveFairy/LeNet-5](https://github.com/TheAgaveFairy/LeNet-5)
