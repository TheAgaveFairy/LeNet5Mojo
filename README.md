# LeNet5Mojo: LeNet-5 from Scratch in Mojo🔥

A high-performance LeNet-5 Convolutional Neural Network built entirely from scratch in [Mojo🔥](https://www.modular.com/mojo) with custom CPU and GPU kernels — featuring compile-time swappable activation functions, custom arena allocators, and the Modular/MAX ecosystem throughout.

## Project Motivation

- **Learn CNNs from first principles** — every component hand-rolled: forward pass, backpropagation, weight updates, cross-entropy loss, softmax
- **Explore the Modular/MAX ecosystem** — systems programming for AI, using `layout` for compile-time tensor shapes and `std.gpu` for GPU kernel dispatch
- **Build custom GPU kernels** without CUDA C, PTX, or external ML libraries
- **Custom memory management** — bump arena allocators on both CPU and GPU, pre-allocating one slab per run and sub-allocating into it
- **Compile-time activation functions** — switch between ReLU, GELU, GELUTanh, GELUFast, Sigmoid, and Tanh with a compiler flag; no runtime branching

## Performance Highlights

- **BENCHMARKS PENDING** within an order of magnitude of major ML libraries
- **Matched accuracy** within ±0.5% of reference implementations
- **Custom GPU kernels** written in plain Mojo — no CUDA C

*A full benchmark suite against JAX, MAX, PyTorch, and ONNX Runtime is in progress.*

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
- **`std.benchmark`** — lifetime management utilities

No PyTorch, TensorFlow, JAX, or BLAS. All operations are hand-rolled in Mojo.

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

### Arena Allocators

Both CPU and GPU use custom bump arena allocators to avoid per-tensor allocation overhead:

- **`CPUBumpArenaAllocator`** — one pre-allocated heap slab; model weights and all intermediate `Feature` buffers sub-allocate from it. `wipe()` zeroes and resets; no individual frees during training.
- **`GPUBumpArenaAllocator`** — same pattern on GPU: one `DeviceBuffer[uint8]` backing slab, typed sub-buffers via `create_sub_buffer` with alignment padding. The full batch of `FeatureGPUBuffers` for an inference run comes from a single arena.
- **`CPUSystemAllocator` / `GPUSystemAllocator`** — drop-in alternatives that call the system allocator per-request; same interface, useful for profiling or one-off allocations.
- **`DeviceSession`** — ties arena, weight buffers, and `LeNet5GPU` view lifetimes together; no manual lifetime juggling at the call site.

### GPU Kernels

All GPU computation is written in plain Mojo `def` functions — no CUDA C syntax:

| Kernel | What it does |
|--------|-------------|
| `normalizeInputsKernel` | Fused H2D: per-image mean/std computed in shared memory, normalize into padded feature buffer |
| `conv1FusedKernel` | Conv (1→6 ch) with bias + activation; weights cached in shared memory |
| `conv2FusedKernel` | Conv (6→16 ch); channel divisions to fit thread block resource limits |
| `conv3FusedKernel` | Conv (16→120 ch); reduction across 16×5×5 inputs per output channel |
| `maxPool1Kernel` / `maxPool2Kernel` | 2×2 max pooling with shared-memory staging |
| `matMulFusedKernel` | FC layer (120→10) as a parallel tree reduction in shared memory |
| `gatherOutputsKernel` | Scatter per-image logits into a flat output buffer for host argmax |

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

> **Note:** A comprehensive benchmark suite against JAX, MAX, PyTorch, and ONNX Runtime is in progress. The table below will be expanded significantly.

| Implementation | Platform | Time (ms) | Notes |
|---------------|----------|-----------|-------|
| LeNet5Mojo | GPU | 2069 | Custom Mojo kernels |
| LeNet5Mojo | CPU | 12381 | Multi-threaded + SIMD |
| PyTorch | GPU | 2150 | ~4% slower than LeNet5Mojo |
| PyTorch | CPU | 2485 | Reference |
| C/CUDA | CPU | 4241 | Stack-allocated model |

*AMD Ryzen 7600X / NVidia RTX 3070 8GB. `-O3`, batch size 50, 60,000 images, average of 10 runs.*

## Current Limitations

- GPU **training** not implemented — inference only
- GPU batch size bounded by VRAM; default 50, tested to ~100 on RTX 3070 8GB
- Tail batch silently dropped when dataset size isn't divisible by batch size
- GPU inference result logger not yet wired up

## Planned Improvements

- Ping-pong H2D streaming — overlap copy and compute across batches
- Wire up GPU inference result logging to CSV
- Comprehensive benchmark suite (JAX, MAX, PyTorch, ONNX Runtime)
- `CPUSession` struct to bind arena + model lifetimes together
- `MNISTBatch.slice()` for explicit batch-level slicing

## Contributing

Educational project — suggestions and discussion on Mojo patterns or optimization techniques are welcome.

## Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) and the [MAX platform](https://www.modular.com/max) by [Modular](https://www.modular.com/)
- MNIST dataset by Yann LeCun et al.
- Inspired by the original [LeNet-5 paper](http://yann.lecun.com/exdb/lenet/) by Y. LeCun et al.
- Prior C/CUDA implementation: [TheAgaveFairy/LeNet-5](https://github.com/TheAgaveFairy/LeNet-5)
