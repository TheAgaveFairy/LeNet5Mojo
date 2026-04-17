#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import argparse


class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture for MNIST classification
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,
                               padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.AvgPool2d(
            kernel_size=2, stride=2)      # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)           # 14x14 -> 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)     # 10x10 -> 5x5

        # Classifier layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)
        self.fc3 = nn.Linear(120, num_classes)

    def forward(self, x):
        # Feature extraction
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        # Flatten for fully connected layers
        x = x.view(-1, 16 * 5 * 5)

        # Classification
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


def load_mnist_data(num_samples=10000, batch_size=64):
    """
    Load MNIST dataset (combines train and test if more than 10k samples needed)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    # Load test dataset (10,000 samples)
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    dataset = test_dataset

    # If we need more than 10k samples, also load training data
    if num_samples > len(test_dataset):
        print(f"Requested {num_samples} samples, but test set only has {
              len(test_dataset)}")
        print("Loading training set as well...")

        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        # Combine datasets
        dataset = torch.utils.data.ConcatDataset([test_dataset, train_dataset])
        print(f"Combined dataset size: {len(dataset)} samples")

    # Limit dataset size if we have more than requested
    if num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)

    print(f"Using {len(dataset)} samples for benchmark")

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader


"""
def load_mnist_data(num_samples=10000, batch_size=64):
    """
# Load MNIST test dataset
"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Limit dataset size if requested
    if num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader
"""


def initialize_model():
    """
    Create and initialize LeNet-5 model with random weights
    """
    model = LeNet5(num_classes=10)

    # Initialize weights (optional, for consistency)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model


def benchmark_inference(model, data_loader, device, device_name, warmup_batches=5):
    """
    Benchmark inference speed on given device
    """
    model.to(device)
    model.eval()

    total_samples = 0
    total_batches = len(data_loader)

    print(f"\nBenchmarking on {device_name}...")
    print(f"Total batches: {total_batches}")

    # Warmup phase
    print("Warming up...")
    with torch.no_grad():
        warmup_count = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            if warmup_count >= warmup_batches:
                break
            data = data.to(device, non_blocking=True)
            _ = model(data)
            warmup_count += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Actual benchmark
    print("Starting benchmark...")
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data = data.to(device, non_blocking=True)

            # Forward pass
            outputs = model(data)

            total_samples += data.size(0)

            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{total_batches} batches")

    # Ensure all GPU operations are complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    return inference_time, total_samples


def main():
    parser = argparse.ArgumentParser(
        description='LeNet-5 CPU vs GPU Benchmark')
    parser.add_argument('--samples', type=int, default=60000,
                        help='Number of MNIST samples to process (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Only run CPU benchmark')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Only run GPU benchmark')

    args = parser.parse_args()

    print("=" * 60)
    print("LeNet-5 Inference Benchmark: CPU vs GPU")
    print("=" * 60)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available - GPU benchmark will be skipped")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of samples: {args.samples:,}")
    print(f"Batch size: {args.batch_size}")

    # Load data
    print("\nLoading MNIST data...")
    test_loader = load_mnist_data(args.samples, args.batch_size)

    # Initialize model
    model = initialize_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = {}

    # CPU Benchmark
    if not args.gpu_only:
        cpu_time, cpu_samples = benchmark_inference(
            model, test_loader, torch.device('cpu'), 'CPU'
        )
        results['CPU'] = {
            'time': cpu_time,
            'samples': cpu_samples,
            'fps': cpu_samples / cpu_time
        }

    # GPU Benchmark
    if cuda_available and not args.cpu_only:
        gpu_time, gpu_samples = benchmark_inference(
            model, test_loader, torch.device('cuda'), 'GPU'
        )
        results['GPU'] = {
            'time': gpu_time,
            'samples': gpu_samples,
            'fps': gpu_samples / gpu_time
        }

    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for device_name, result in results.items():
        print(f"\n{device_name} Performance:")
        print(f"  Total time: {result['time']:.3f} seconds")
        print(f"  Total time: {result['time'] * 1_000_000_000} ns")
        print(f"  Samples processed: {result['samples']:,}")
        print(f"  Throughput: {result['fps']:.1f} samples/second")
        print(f"  Latency per sample: {
              (result['time'] / result['samples']) * 1000:.3f} ms")

    # Compare if both devices were tested
    if 'CPU' in results and 'GPU' in results:
        speedup = results['GPU']['fps'] / results['CPU']['fps']
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

        if speedup > 1:
            print(f"GPU is {speedup:.1f}x faster")
        else:
            print(f"CPU is {1/speedup:.1f}x faster")

    print("\n" + "=" * 60)

    with open("pytorchresults.txt", "a") as res:
        for device in ['CPU', 'GPU']:
            line = f"PyTorch {
                device}, " + str(results[device]['time'] * 1_000_000_000) + f", Batch Size: {args.batch_size}\n"
            res.write(line)


if __name__ == "__main__":
    main()
