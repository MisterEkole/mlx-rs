# API Coverage

This document tracks the implementation status of MLX features in mlx-rs. It's regularly updated as new functionality is added.

**Last Updated:** February 2026

## Implementation Status Overview

- ✅ **Implemented** - Feature is complete and tested
- 🚧 **In Progress** - Currently being developed
- 📋 **Planned** - Scheduled for future implementation
- ❌ **Not Planned** - Not currently on the roadmap

---

## Core Array Operations

### Array Creation & Manipulation
| Feature | Status | Notes |
|---------|--------|-------|
| Array creation from slices | ✅ | Fully supported |
| Array from scalar values | ✅ | |
| zeros, ones, full | ✅ | |
| arange, linspace | ✅ | |
| eye, identity | ✅ | |
| Array indexing | ✅ | Basic and advanced indexing |
| Array slicing | ✅ | Multi-dimensional slicing |
| reshape | ✅ | |
| transpose | ✅ | |
| flatten | ✅ | |
| squeeze, expand_dims | ✅ | |
| concatenate, stack | ✅ | |
| split | 📋 | Planned |
| pad | 📋 | Planned |
| repeat, tile | 📋 | Planned |

### Element-wise Operations
| Feature | Status | Notes |
|---------|--------|-------|
| Addition, subtraction | ✅ | |
| Multiplication, division | ✅ | |
| Power operations | ✅ | |
| abs, negative, sign | ✅ | |
| exp, log, log2, log10 | ✅ | |
| sqrt, square | ✅ | |
| Trigonometric (sin, cos, tan, etc.) | ✅ | |
| Inverse trig (arcsin, arccos, etc.) | ✅ | |
| Hyperbolic functions | 📋 | Planned |
| ceil, floor, round | ✅ | |
| Comparison operators | ✅ | |
| Logical operators | ✅ | |
| where (conditional selection) | ✅ | |

### Reduction Operations
| Feature | Status | Notes |
|---------|--------|-------|
| sum | ✅ | With axis support |
| mean | ✅ | |
| var, std | 📋 | Planned |
| min, max | ✅ | |
| argmin, argmax | 📋 | Planned |
| all, any | 📋 | Planned |
| logsumexp | 📋 | Planned |
| cumsum, cumprod | 📋 | Planned |

### Broadcasting & Shape Operations
| Feature | Status | Notes |
|---------|--------|-------|
| Automatic broadcasting | ✅ | |
| broadcast_to | ✅ | |
| broadcast_arrays | 📋 | Planned |
| swapaxes, moveaxis | 📋 | Planned |

### Matrix Operations
| Feature | Status | Notes |
|---------|--------|-------|
| Matrix multiplication (matmul) | ✅ | Optimized for Apple Silicon |
| Dot product | ✅ | |
| Batch matrix multiplication | ✅ | |
| Outer product | 📋 | Planned |

---

## Linear Algebra (`mlx.linalg`)

| Feature | Status | Notes |
|---------|--------|-------|
| inv (matrix inverse) | ✅  | |
| norm (vector/matrix norms) | ✅  | |
| svd (Singular Value Decomposition) | ✅  |  |
| eig, eigh (Eigenvalues) | ✅ |  |
| qr (QR Decomposition) | ✅  | |
| cholesky | ✅ | Planned |
| solve (linear systems) | ✅ | |
| solve_triangular | ✅  | |
| det, slogdet (Determinant) | ✅  | |
| pinv (Pseudo-inverse) | ✅  | |

---

## FFT Operations (`mlx.fft`)

| Feature | Status | Notes |
|---------|--------|-------|
| fft, ifft (1D) | ✅ | |
| rfft, irfft (Real FFT) | ✅ | |
| fft2, ifft2 (2D) | ✅  | Planned |
| fftn, ifftn (N-dimensional) |✅ | |
| fftshift, ifftshift | ✅ | |
| fftfreq, rfftfreq | ✅ | |

---

## Random Number Generation (`mlx.random`)

| Feature | Status | Notes |
|---------|--------|-------|
| key, split (PRNG key management) | 📋 | Planned |
| uniform | 📋 | Planned |
| normal | 📋 | Planned |
| bernoulli | 📋 | Planned |
| categorical | 📋 | Planned |
| randint | 📋 | Planned |
| permutation, shuffle | 📋 | Planned |
| multivariate_normal | 📋 | Planned |
| truncated_normal | 📋 | Planned |

---

## Neural Networks (`mlx.nn`)

### Layers
| Feature | Status | Notes |
|---------|--------|-------|
| Linear (Dense) | ✅ | Fully featured |
| Conv1d | ✅ | |
| Conv2d | ✅ | |
| Conv3d | 📋 | |
| ConvTranspose1d | ✅  |  |
| ConvTranspose2d | ✅  |  |
| Embedding | ✅ | |
| Dropout | ✅ | |
| BatchNorm | 📋 | |
| LayerNorm | ✅| |
| GroupNorm | 📋 | |
| InstanceNorm | 📋 | |
| RMSNorm | ✅ | |

### Recurrent Layers
| Feature | Status | Notes |
|---------|--------|-------|
| RNN | ✅ | |
| LSTM | ✅ | |
| GRU | ✅ | |
| Bidirectional wrappers | ✅ ||

### Pooling Layers
| Feature | Status | Notes |
|---------|--------|-------|
| MaxPool1d | ✅| |
| MaxPool2d | ✅| |
| AvgPool1d |✅| |
| AvgPool2d | ✅| |
| AdaptiveAvgPool | 📋 | Planned |
| AdaptiveMaxPool | 📋 | Planned |

### Activation Functions
| Feature | Status | Notes |
|---------|--------|-------|
| ReLU | ✅ | |
| GELU | ✅ | |
| SiLU (Swish) | 📋| |
| Sigmoid | ✅ | |
| Tanh | ✅ | |
| Softmax | ✅ | |
| LogSoftmax | ✅ ||
| LeakyReLU | ✅ | |
| ELU | 📋|Planned |
| PReLU | 📋 | Planned |
| Mish | 📋 | Planned |

### Attention Mechanisms
| Feature | Status | Notes |
|---------|--------|-------|
| MultiHeadAttention | ✅ | Planned|
| Scaled Dot-Product Attention |✅ | Planned|
| Cross Attention | ✅ | |
| Rotary Position Embeddings (RoPE) | 📋 | Planned |
| Flash Attention | 📋 | Planned |

### Transformer Components
| Feature | Status | Notes |
|---------|--------|-------|
| TransformerEncoder | ✅ | |
| TransformerDecoder | ✅| |
| TransformerEncoderLayer | ✅| |
| TransformerDecoderLayer | ✅ | |

### Loss Functions
| Feature | Status | Notes |
|---------|--------|-------|
| MSE Loss | ✅ | |
| Cross Entropy Loss | ✅ ||
| Binary Cross Entropy | ✅| |
| L1 Loss |✅| |
| Smooth L1 Loss |✅| |
| KL Divergence | ✅| |
| Cosine Embedding Loss |✅| |

---

## Optimizers (`mlx.optimizers`)

| Feature | Status | Notes |
|---------|--------|-------|
| SGD | ✅ | With momentum support |
| Adam | ✅ | |
| AdamW | ✅||
| AdaGrad |✅| |
| RMSprop | ✅| |
| Lion | ✅| |
| Adafactor | ✅| |

### Learning Rate Schedulers
| Feature | Status | Notes |
|---------|--------|-------|
| StepLR | 📋 | Planned |
| ExponentialLR | 📋 | Planned |
| CosineAnnealingLR | 📋 | Planned |
| ReduceLROnPlateau | 📋 | Planned |
| OneCycleLR | 📋 | Planned |
| Warmup schedules | 📋 | Planned |

---

## Automatic Differentiation

| Feature | Status | Notes |
|---------|--------|-------|
| grad | ✅ | Compute gradients |
| value_and_grad | ✅ | Value and gradient together |
| vjp (Vector-Jacobian Product) | 📋 | Planned |
| jvp (Jacobian-Vector Product) | 📋 | Planned |
| jacobian | 📋 | Planned |
| hessian | 📋 | Planned |
| stop_gradient | 📋 | Planned |
| Custom gradient functions | 📋 | Planned |

---

## Function Transformations

| Feature | Status | Notes |
|---------|--------|-------|
| vmap (Vectorization) |  ✅  |  |
| compile (JIT Compilation) |  ✅ | |
| checkpoint (Gradient Checkpointing) |  ✅  | |

---

## Quantization

| Feature | Status | Notes |
|---------|--------|-------|
| 4-bit quantization | 📋| Planned|
| 8-bit quantization | 📋 | Planned|
| quantize, dequantize | 📋| Planned |
| QuantizedLinear | 📋 | Planned |
| QuantizedEmbedding | 📋 | Planned |
| Quantized Attention | 📋 | Planned |
| Dynamic quantization | 📋 | Planned |
| Static quantization | 📋 | Planned |

---

## File I/O

### Serialization Formats
| Feature | Status | Notes |
|---------|--------|-------|
| NumPy format (save/load) |  ✅  | |
| Safetensors format |  ✅ | |
| GGUF format | 📋 | Planned (llama.cpp compat) |
| Pickle format | 📋 | Planned |

### Model Management
| Feature | Status | Notes |
|---------|--------|-------|
| Save model weights | ✅ | |
| Load model weights |  ✅ |  |
| Checkpoint management | 📋 | Planned |
| Partial loading | 📋 | Planned |
| Model sharding | 📋 | Planned |

---

## Distributed Computing

| Feature | Status | Notes |
|---------|--------|-------|
| distributed.init |  ✅ |  |
| all_reduce | ✅ | |
| all_gather |  ✅ |  |
| all_sum |  ✅ | |
| broadcast | ✅ | |
| Multi-GPU support | 📋 | |
| Data parallelism | 📋 | |
| Model parallelism | 📋 | |

---

## Stream Management

| Feature | Status | Notes |
|---------|--------|-------|
| Stream creation | ✅ | |
| Stream synchronization | ✅ | |
| Default stream management | ✅ | |
| Stream context managers | ✅ | |
| Async operations | 📋| |

---

## Utilities

| Feature | Status | Notes |
|---------|--------|-------|
| eval (Force evaluation) | ✅ | |
| Device management (cpu/gpu) | ✅ | |
| Memory pool management | ✅ | Planned |
| depends (Operation dependencies) | 📋 | Planned |
| tree_map, tree_flatten | 📋 | Planned |
| Profiling utilities | 📋 | Planned |

---

## Platform Support

| Platform | Status | Notes |
|---------|--------|-------|
| macOS M1 | ✅ | Fully supported |
| macOS M2 | ✅ | Fully supported |
| macOS M3 | ✅ | Fully supported |
| macOS M4 | ✅ | Fully supported |
| Intel Macs | ❌ | Not supported by MLX |
| Linux | ❌ | Not supported by MLX |
| Windows | ❌ | Not supported by MLX |

---

## Contributing to API Coverage

If you'd like to contribute to implementing any of these features:

1. Check the status in this document
2. Open an issue on GitHub to discuss the implementation
3. Reference the [MLX C API documentation](https://github.com/ml-explore/mlx-c)
4. Follow the contribution guidelines in CONTRIBUTING.md
5. Update this document when features are completed

---

## Changelog

### February 2026
- ✅ Completed core array operations
- ✅ Implemented neural network module
- ✅ Added automatic differentiation (grad, value_and_grad)
- ✅ Implemented SGD and Adam optimizers
