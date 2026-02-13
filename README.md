# CUDA-Implementation-of-the-Bitonic-Sort-Algorithm

This repository contains baseline and optimized CUDA implementations of the Bitonic Sort algorithm, together with a detailed performance analysis and comparison against NVIDIA Thrust library sorting algorithms (Radix Sort and Merge Sort).

The project focuses on identifying performance bottlenecks in GPU-based Bitonic Sort and improving memory efficiency, kernel execution behavior, and overall throughput.

## Project Overview

Bitonic Sort is a comparison-based parallel sorting algorithm with time complexity:

O(n log² n)

Although well-suited for parallel architectures, its performance on large datasets is often limited by global memory access patterns and kernel launch overhead.

This project includes:

- Baseline GPU Bitonic Sort implementation
- Optimized Bitonic Sort implementation
- Performance comparison with:
  - Thrust Radix Sort
  - Thrust Merge Sort
- Profiling using NVIDIA Nsight Compute

## Hardware and Software Environment

Experiments were conducted using:

- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 12.4
- NVIDIA Driver: 550.54.15
- Profiling Tool: NVIDIA Nsight Compute
- Development Environment: Google Colab


## Implementations

### Baseline Bitonic Sort
- Multiple kernel launches per merge stage
- Heavy global memory usage
- Memory-bound behavior
- High kernel invocation overhead

### Optimized Bitonic Sort
- Reduced number of global merge kernel launches
- Multi-stage merging in shared memory
- Improved occupancy
- Higher DRAM throughput utilization
- Reduced redundant instructions

## Performance Analysis

Profiling metrics collected:

- DRAM Throughput
- Global Memory Transactions
- SM Busy
- Achieved Occupancy
- Warp Efficiency
- Kernel Launch Count

Results show that while optimization significantly improves Bitonic Sort performance, it remains slower than highly optimized Thrust Radix and Merge Sort implementations due to its inherent O(n log² n) complexity and memory access characteristics.
