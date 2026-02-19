# mlx-rs

Rust bindings for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning on Apple silicon.

## Overview

MLX is an array framework for machine learning research on Apple silicon. `mlx-rs` provides safe, idiomatic Rust bindings to MLX via the [mlx-c](https://github.com/ml-explore/mlx-c) C API.

### Features

- **Safe Rust API**: High-level, ergonomic interface with automatic memory management
- **Zero-cost abstractions**: Thin wrapper over mlx-c with minimal overhead
- **Type safety**: Strongly-typed arrays and operations
- **Lazy evaluation**: Leverage MLX's lazy computation model
- **Unified memory**: Arrays accessible from both CPU and GPU without copies

## Project Structure

This repository contains two crates:

- **`mlx-sys`**: Low-level FFI bindings to mlx-c (unsafe)
- **`mlx`**: High-level safe Rust API (recommended for users)

## 🧪 Installation & Setup

**Note:** These instructions are for the `release` branch, which uses dynamic binding generation.

### Prerequisites

1. **macOS with Apple Silicon** (M1, M2, M3, or later)
2. **Xcode Command Line Tools**: `xcode-select --install`
3. **CMake**: `brew install cmake`

### Step 1: Build the MLX-C Engine

`mlx-rs` relies on the C-wrapper to interface with the C++ core. You must build this locally first:

```bash
# Clone the patched mlx-c (includes quantize wrapper functions)
git clone https://github.com/MisterEkole/mlx-c.git
cd mlx-c


**Why a fork?** The official mlx-c passes `mlx_optional_int` structs by value,
> which causes an [ABI mismatch](docs/Rust_C_ABI_Mismatch_Quantization.md) when called
> from Rust on ARM64. Our fork adds thin C wrapper functions that accept plain
> `int` parameters.


# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### Step 2: Setup mlx-rs

```bash
# Clone the repository and switch to the release branch
git clone -b release https://github.com/MisterEkole/mlx-rs.git
cd mlx-rs

# Set the critical environment variable (Replace with your actual path from Step 1)
export MLX_C_PATH=/path/to/your/mlx-c

# Run the pre-flight check script to verify libraries are found
chmod +x check_env.sh
./check_env.sh
```

**CRITICAL:** You must build the mlx-sys crate first. This triggers the build script to generate a fresh `bindings.rs` file tailored to your local machine.

```bash
# Build the sys crate to generate bindings.rs
cargo build -p mlx-sys
```

### Step 3: Run Examples

Once the environment is validated and bindings are generated, you can run the provided test cases:

```bash
# Test basic array operations
cargo run --example basic_ops

# Test a Convolutional Neural Network training loop
cargo run --example cnn
```


## Development Status

**⚠️ Early Development**: This project is in early development. APIs may change. Advanced features are being added regularly.

**📊 For detailed project development and API coverage status, see [API_COVERAGE.md](API_COVERAGE.md)**

## Contributing

Contributions are welcome! This project follows the same spirit as MLX - designed by ML researchers for ML researchers.

## Architecture

```
┌─────────────────────────────────────┐
│  Your Rust Application              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  mlx crate (Safe Rust API)          │
│  - Array, Dtype types                │
│  - Memory management via Drop        │
│  - Error handling                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  mlx-sys (Raw FFI Bindings)         │
│  - Generated via bindgen             │
│  - Unsafe C function declarations    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  mlx-c (C API)                       │
│  - C wrapper around MLX C++          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MLX (C++ Core)                      │
│  - Metal shaders                     │
│  - Array operations                  │
│  - Neural network primitives         │
└─────────────────────────────────────┘
```

## Why Rust + MLX?

- **Safety**: Rust's ownership system prevents common bugs
- **Performance**: Zero-cost abstractions compile to efficient code
- **Ecosystem**: Integrate with Rust's rich crate ecosystem
- **Ergonomics**: Idiomatic Rust API following language conventions
- **Apple Silicon**: Native performance on M-series chips

## License

MIT License - see LICENSE file for details.

This project is not officially affiliated with Apple. MLX is created by Apple's machine learning research team.

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX C GitHub](https://github.com/ml-explore/mlx-c)
- [MLX Swift](https://github.com/ml-explore/mlx-swift) (similar approach for Swift)

## Acknowledgments

- Apple ML Research team for creating MLX
- The mlx-c contributors for providing the C API bridge
- The Rust community for excellent FFI tooling