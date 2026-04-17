# MojoCNN: LeNet-5 Implementation from Scratch

A high-performance implementation of the LeNet-5 Convolutional Neural Network built entirely from scratch in Mojo🔥 with custom CPU and GPU kernels, achieving competitive performance with established frameworks.

## Project Motivation

This project was undertaken as a deep learning exercise to:
- **Learn CNNs from first principles** by implementing every component from scratch
- **Explore Mojo**, a cutting-edge systems programming language designed for AI workloads
- **Build custom GPU kernels** without relying on existing ML libraries or frameworks
- **Achieve competitive performance** through low-level optimization and manual memory management

## Performance Highlights

- **6x speedup** GPU vs CPU inference (AMD Ryzen 7600X 32GB DDR5, NVidia RTX 3070 8GB)
- **4% faster** than PyTorch GPU inference (batch size 50)
- **Maintained accuracy** within ±0.5% of reference implementations
- **Custom GPU kernels** written entirely in Mojo

## Architecture

This implementation features a modified LeNet-5 architecture:
- Convolutional layers with custom kernels
- Max pooling operations
- Fully connected layers
- ReLU activation function
- MNIST dataset integration

*Note: The traditional 84-unit penultimate layer is omitted for direct comparison with a [previous](https://github.com/TheAgaveFairy/LeNet-5) C/CUDA implementation. Additionally, some skipped connections are omitted after the first pooling. Other differences might be found from the original paper, but this version is still a very common implementation.*

## Project Structure

```
├── main.mojo              # CPU training/testing, GPU testing, logging integration
├── lenet.mojo             # CPU model implementation and operations
├── lenetgpu.mojo          # GPU model implementation and kernels
├── image.mojo             # Image struct for MNIST data representation
├── dataloader.mojo        # MNIST data loading utilities
├── helpers.mojo           # Progress bar and activation functions
├── resultlogger.mojo      # Result logging to file functionality
├── deviceinfo.mojo        # GPU device information utilities
├── pytorch/               # PyTorch reference implementation
│   └── pytorch.py         # Implementation
│   └── *results.txt       # Benchmarking results
│   └── runner.sh          # For quick benchmarking multiple times
├── data/
│   └── *-ubyte           # MNIST dataset files
├── models/
│   └── model*.dat        # Pre-trained model weights
└── results/              # Logging output directory
    └── *.csv             # Performance benchmarking results
```

## Technical Implementation

### Custom Components Built from Scratch
- **Memory Management**: Manual allocation using UnsafePointers
- **Matrix Operations**: Custom implementations without external BLAS libraries  
- **GPU Kernels**: Hand-written kernels in Mojo for all operations
- **Data Pipeline**: Custom MNIST loader with proper header handling
- **Forward Pass**: Complete inference pipeline optimized for both CPU and GPU

### Key Features
- Zero external ML library dependencies
- Custom GPU memory management and kernel execution
- Batch processing support (tested up to batch size 75)
- Cross-platform compatibility (CPU/GPU)
- Custom logging for training and testing

## Getting Started

### Prerequisites
- Mojo 25.5.0.dev2025072405
- Mojo Supported GPU (NVidia, AMD. Apple support soon!)
- Pixi package manager

### Installation & Usage

```bash
# Install dependencies
pixi shell

# CPU training and inference
mojo main.mojo

# GPU inference only
mojo lenetgpu.mojo

# Build executable
mojo build main.mojo
mojo build lenetgpu.mojo
```

## Performance Comparison

| Implementation | Platform | Time in ms | Notes |
|---------------|----------|----------------|--------|
| MojoCNN | GPU | 2069 | Custom kernels |
| MojoCNN | CPU | 12381 | Baseline |
| PyTorch | GPU | 2150 | 4% slower than MojoCNN |
| PyTorch | CPU | 2485 | For scaling reference |
| C/CUDA | CPU | 4241 | Stack-allocated model |

*All benchmarks conducted with -O3 optimization and batch size 50 on 60,000 images. Times are averages of 10 runs.*

## Current Limitations & Future Work

### Known Limitations
- GPU training not implemented (inference only)
- Missing 84-unit penultimate layer from standard LeNet-5 and skip connections
- Batch size limited to ~75 due to memory constraints
- Some potential memory leaks in edge cases

### Planned Improvements
- [ ] GPU training implementation
- [ ] Complete LeNet-5 architecture with all features
- [ ] Memory optimization (stack allocation where possible)
- [ ] Kernel tiling and streaming optimizations
- [ ] SIMD vectorization for CPU operations
- [ ] Full model serialization/deserialization
- [ ] Comprehensive profiling and benchmarking suite
- [ ] Dynamic batch size support

## Contributing

This is primarily an educational project, but suggestions and discussions about optimization techniques or Mojo best practices are welcome!

## Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) by Modular
- MNIST dataset from Yann LeCun's database
- Inspired by the original LeNet-5 paper by Y. LeCun et al.
