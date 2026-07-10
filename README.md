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

- **~1.47M images/sec** peak GPU throughput — the **fastest** implementation measured, edging out ONNX-RT + TensorRT (~1.42M) and clear of JAX/XLA (~1.19M), PyTorch, and ONNX Runtime CUDA
- Even **single-stream** (no copy/compute overlap), the custom kernels hit **~1.33M img/s** — behind only their own multi-stream mode and TensorRT, and ahead of *every* other framework
- **~1.9× the best hand-tuned MAX Engine config** (and ~12× its default graph) — hand-written Mojo kernels beat Modular's own compiler at small-CNN inference
- **Matched accuracy** — 96.86% (9686/10000), identical to the PyTorch and JAX reference models
- **Custom GPU kernels** written in pure Mojo (no CUDA C, cuDNN, or BLAS), supporting NVidia, AMD, and Apple

Two optimizations drove the latest jump — **fusing conv1 + activation + pool1 into one kernel** (layer1 never touches global memory) and **recasting conv3 as a batched GEMM**. See [Key GPU Optimizations](#key-gpu-optimizations) for the details.

*Only the compute-only ("data resident in VRAM") variants remain ahead; see [Performance Comparison](#performance-comparison) for the full table and methodology.*

## Architecture

Modified LeNet-5 on MNIST:

```
Input (1×32×32, zero-padded from 28×28)
  → Conv1 (1→6 ch, 5×5) → Act* → MaxPool (2×2)   ┐ fused into one GPU kernel
  → Conv2 (6→16 ch, 5×5) → Act* → MaxPool (2×2)
  → Conv3 (16→120 ch, 5×5) → Act*                  ┐ a 5×5→1×1 full reduction, run as a GEMM
  → FC (120→10)                                     ┘ also a GEMM
  → Logits
```

*`Act` is compile-time selectable — ReLU by default.*

On the GPU path, Conv1→Act→Pool1 is a single fused kernel (the layer1 intermediate never materializes in global memory), and Conv3's 5×5 window collapses its 5×5 input to a single pixel — a full 16×5×5 → 120 reduction — so it and the FC layer are both dispatched as tiled GEMMs. See [Key GPU Optimizations](#key-gpu-optimizations).

The 84-unit penultimate layer and some skip connections from the original LeCun et al. paper are intentionally omitted, for consistency with a [prior C/CUDA implementation](https://github.com/TheAgaveFairy/LeNet-5).

## Project Structure

```
├── main.mojo              # Entry point: CPU train + test, GPU inference, logging
├── constants.mojo         # Architecture dims, dtype, compile-time activation selection
├── activation_fn.mojo     # ActivationFunction trait + ReLU, GELU, GELUTanh, GELUFast, Sigmoid, Tanh
├── dataloader.mojo        # MNISTDataRepository, MNISTDataView (SoA arena view)
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
mojo src/main.mojo               # ReLU (default)
mojo -D GELU src/main.mojo       # Exact GELU (erf-based)
mojo -D GELUTanh src/main.mojo   # GELU tanh approximation
mojo -D GELUFast src/main.mojo   # Fast GELU (sigmoid-based, ~Swish)
mojo -D Sigmoid src/main.mojo    # Sigmoid
mojo -D Tanh src/main.mojo       # Tanh
```

Each activation implements `simdForward`, and `simdBackward`. CPU operations use the layout-level `forward` or `backward` calls whose default trait implementations wrap the SIMD-vectorized functions; GPU kernels call `simdForward` directly.

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
| `conv1PoolFusedKernel` | **Conv (1→6 ch) + bias + activation + 2×2 maxpool, fused** — one block per image, one thread per *pooled* output pixel; layer1 never hits global memory |
| `conv2FusedKernel` | Conv (6→16 ch); channel divisions to fit thread block resource limits |
| `maxPool2Kernel` | 2×2 max pooling after conv2 |
| `gemmFusedKernel` | Tiled shared-memory GEMM `C = A·B + bias` with optional activation epilogue; drives **conv3** (`Conv3GemmKernel`) and the **FC layer** (`FCGemmKernel`) |
| `argMaxKernel` | Argmax over the final logits (gather fused away) |

Kernels use `LayoutTensor` for type-safe indexing and `comptime for` for inner-loop unrolling at compile time. The GEMM kernel was developed and benchmarked standalone in `tests/gemm.mojo` before wiring in.

### Key GPU Optimizations

Two rewrites account for most of the recent throughput gain from ~1.17M to ~1.47M img/s.

#### Conv1 + activation + pool1, fused into one kernel

The original pipeline ran conv1 and its 2×2 maxpool as two separate launches, with the full `1→6 ch, 28×28` layer1 feature map written to and re-read from global memory in between. `conv1PoolFusedKernel` collapses conv, bias, activation, and pool into a **single launch that never materializes layer1**:

- **One block per image, one thread per *pooled* output pixel** (a 14×14 block). The 150 weight floats, the padded 32×32 input, and the biases are staged once into shared memory (196 threads stride-load the 1024 input pixels).
- Each thread then computes the **four** conv outputs of its 2×2 pool window directly in registers (6 out-channels × 4 pixels × 25 MACs against shared memory), applies the activation, and writes only the **max** to global memory.
- Pooling is applied **post-activation** to stay bit-exact with the CPU reference — the activation isn't guaranteed monotonic (e.g. GELU), so max-then-activate would diverge.

Net effect: the layer1 buffer is dead on the GPU path, one launch and one global-memory round-trip disappear, and each output pixel's inputs are read from shared memory instead of DRAM.

#### Conv3 as a GEMM

Conv3 is a `16→120 ch, 5×5` convolution over a 5×5 input feature map — the window exactly covers the input, so the `5×5` spatial output collapses to `1×1`. Every output channel is therefore a **full dot product over all 16×5×5 = 400 inputs**, i.e. conv3 is a matrix multiply in disguise.

- The batched (SoA) feature buffers are dense row-major, so the layer4 tensor `(bs, 16, 5, 5)` **reshapes for free** to the GEMM operand `A = (bs, 400)` — no data movement, just a view.
- Weights are stored transposed (`w45g`, `[ic, kw, kh, oc]`) as the GEMM operand `B = (400, 120)`, giving coalesced reads; bias and the activation are **fused into the GEMM epilogue**.
- The result is a tiled shared-memory GEMM with the batch as the `M` dimension — replacing the old per-output-channel reduction kernel and yielding a **~1.83× speedup** on conv3. The identical `gemmFusedKernel` also drives the FC layer (with the activation epilogue disabled, since the final logits are raw).

Recasting conv3 as a GEMM also feeds larger, better-shaped launches to the GPU, reducing the reliance on high stream counts for occupancy.

### CPU Training

- **Multi-threaded** via `parallelize` — forward + backward per sample runs in parallel within each batch; weight deltas accumulated after
- **SIMD-vectorized** weight accumulation, cross-entropy loss, and all activation functions via `vectorize`
- **Numerically stable** softmax (max-subtraction trick) and cross-entropy loss

### Model Serialization

`LeNet5.saveToFile` / `loadFromFile` write weights as raw binary. `loadFromFile[filetype]` supports loading weights saved in a different float precision than the current runtime model (e.g. load `float64` weights into a `float32` model). Big-endian support is partial — the write side byte-swaps but the load side doesn't yet, so the round-trip isn't wired up (hosts are little-endian in practice).

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
# Train on CPU, then run GPU inference (ReLU, alpha=0.5)
mojo src/main.mojo

# Compile-time options
mojo -D GELU -D ALPHA=300 src/main.mojo             # GELU activation, alpha=0.3
mojo -D GPU_STREAM_BATCH_SIZE=100 src/main.mojo     # GPU per-stream batch size
mojo -D DISPLAY src/main.mojo                       # Show training progress bars

# Build an optimized binary
pixi run buildmain && ./src/main

# Run arena unit tests
mojo -I src src/cpu/arena.mojo
mojo -I src src/accel/arena.mojo

# Format all source
pixi run formatall
```

## Performance Comparison

Benchmarked against PyTorch, JAX, ONNX Runtime (incl. TensorRT), MAX Engine, and NumPy on the full **10,000-image MNIST test set**. Throughput is the median over the set — **higher is better**.

### GPU — streaming inference (peak throughput)

Each row is that framework's best throughput across the batch-size sweep.

| Framework | Backend | Throughput (img/s) | Accuracy | Config |
|-----------|---------|-------------------:|:--------:|--------|
| **LeNet5Mojo** | **GPU, 12 streams** | **1,472,445** | **96.86%** | bs=125×12 |
| ONNX-RT + TensorRT | GPU | 1,417,848 | 96.48% | bs=2048 |
| **LeNet5Mojo** | **GPU, single stream** | **1,325,875** | **96.86%** | bs=2000 |
| JAX | XLA | 1,187,210 | 96.86% | bs=2048 |
| PyTorch (eager) | CUDA | 791,256 | 96.86% | bs=2048 |
| MAX Engine (FC-RS + resident pool) | GPU | 789,986 | 96.86%* | bs=2048 |
| ONNX Runtime | GPU | 782,434 | 96.48% | bs=2048 |
| MAX Engine (default graph) | GPU | 123,919 | 96.86%* | bs=2048 |

*LeNet5Mojo, PyTorch, and JAX all reach 96.86% (9686/10000); the ONNX-exported model lands at 96.48%. \* MAX Engine's large-batch runs drop the remainder images (accuracy over a smaller denominator).*

### GPU — single-image latency (batch size 1)

Peak throughput is a *saturation* metric (large batch, pipeline full). Single-request **latency** is the orthogonal axis: one image, `batch_size=1`, nothing to hide launch overhead behind. Time per image = 1000 / (bs=1 throughput):

| Framework | Backend | Latency per image |
|-----------|---------|------------------:|
| **LeNet5Mojo** (single stream) | GPU | **~49 µs** |
| ONNX-RT + TensorRT | GPU | ~61 µs |
| ONNX Runtime | GPU | ~126 µs |
| PyTorch (eager) | CUDA | ~183 µs |
| JAX | XLA | ~370 µs |
| MAX Engine (FC-RS + resident pool) | GPU | ~614 µs |
| MAX Engine (default graph) | GPU | ~1,553 µs |

LeNet5Mojo also has the **lowest single-image latency** in the set — the same lean, launch-overhead-light pipeline that wins throughput also wins responsiveness.

**Methodology / fairness notes:**
- **Streaming scenario** (the apples-to-apples default): images are copied host→device *inside* the timed loop. LeNet5Mojo uploads raw `uint8` and normalizes on-GPU — ~4× less PCIe traffic than the libraries' pre-normalized `fp32`.
- *12 streams* = ping-pong overlap of the H2D copy with compute; *single stream* matches the single-stream libraries for a stricter comparison — and, at ~1.33M img/s, still beats every framework except TensorRT.
- Even the plain single-stream run tops PyTorch, ONNX Runtime, and JAX; the fused conv1+pool1 and GEMM-based conv3/FC are what push the multi-stream peak past TensorRT.
- **Compute-only** ("dataset already resident in VRAM") numbers are tracked separately and remain the only configs ahead of LeNet5Mojo's streaming peak. A resident mode for LeNet5Mojo is in progress; until then it isn't in the headline table.

### CPU

The CPU path (`parallelize` + SIMD, hand-rolled) reaches **~28k img/s** — honest about it: this trails vendor-tuned CPU backends like ONNX Runtime / MLAS (~436k img/s peak), MAX Engine (~185k), and PyTorch (~125k). CPU was not the optimization focus; the GPU kernels are where the work went.

*AMD Ryzen 7600X / NVidia RTX 3070 8GB. `fp32`, full 10k test set, median of repeated runs.*

## Current Limitations

- GPU **training** not implemented — inference only
- Stream count (`--num-streams`) is hand-tuned per batch size to hit peak — no auto-default yet
- CPU inference throughput trails vendor-optimized CPU backends (ONNX Runtime / MLAS)
- Only `fp32` / `fp64` paths exercised; `fp16` / `bf16` untested
- `conv1PoolFusedKernel` hardcodes single-channel input (fine for MNIST, guarded by a `comptime assert` — breaks at compile time if extended to multi-channel)

## Planned Improvements

- **Compute-only / resident benchmark mode** — preload the full test set to VRAM for a pure kernel-vs-kernel comparison against XLA / cuDNN / TensorRT (removes Mojo's raw-`uint8` transfer edge); the resident configs are the last ones still ahead of the streaming peak
- **Auto-heuristic for stream count** derived from the batch size, so good numbers come out-of-the-box without a grid search
- **conv2 as a batched GEMM** — the natural extension of the conv3 GEMM work now that the SoA feature layout makes the operand views dense
- **`fp16` / `bf16` dtype paths** + dtype-parity notes vs PyTorch/JAX (which default to TF32 matmuls on Ampere)
- Pre-normalized `fp32` input mode to mirror how the libraries upload images

*Recently landed: tail-padding (any batch size now covers the full test set — remainder images are padded and masked, no longer dropped), the fused conv1+pool1 kernel, and conv3/FC recast as GEMMs.*

## Contributing

Educational project — suggestions and discussion on Mojo patterns or optimization techniques are welcome.

## Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) and the [MAX platform](https://www.modular.com/max) by [Modular](https://www.modular.com/)
- MNIST dataset by Yann LeCun et al.
- Inspired by the original [LeNet-5 paper](http://yann.lecun.com/exdb/lenet/) by Y. LeCun et al.
- Prior C/CUDA implementation: [TheAgaveFairy/LeNet-5](https://github.com/TheAgaveFairy/LeNet-5)
