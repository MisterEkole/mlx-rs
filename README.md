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

## Installation

### Prerequisites

1. **macOS with Apple Silicon** (M1, M2, M3, or later)
2. **MLX C library** installed

#### Installing MLX C

```bash
# Clone mlx-c
git clone https://github.com/ml-explore/mlx-c.git
cd mlx-c

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Install (optional, or note the path for MLX_C_PATH)
sudo make install
```

### Using mlx-rs in Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
mlx = { path = "path/to/mlx-rs/mlx" }
```

### Building from Source

```bash
# Clone this repository
git clone https://github.com/yourusername/mlx-rs.git
cd mlx-rs

# Set MLX_C_PATH to your mlx-c installation
export MLX_C_PATH=/path/to/mlx-c/build

# Build with mlx-c bindings
cargo build --features mlx-c-bindings

# Or build with placeholder types (for development without mlx-c)
cargo build
```

## Usage Example

```rust
use mlx::{Array, Dtype, Result};

fn main() -> Result<()> {
    // Create arrays
    let a = Array::from_slice(&[1.0, 2.0, 3.0], &[3], Dtype::Float32)?;
    let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3], Dtype::Float32)?;
    
    // Perform operations (lazy evaluation)
    let sum = a.add(&b)?;
    let product = a.multiply(&b)?;
    
    // Force evaluation
    sum.eval();
    product.eval();
    
    // Print results
    println!("Sum: {:?}", sum);
    println!("Product: {:?}", product);
    
    Ok(())
}
```

## API Coverage

This is an early version. Currently implemented:

- [x] Basic array creation from slices
- [x] Element-wise addition and multiplication
- [x] Array evaluation
- [ ] Array indexing and slicing
- [ ] Matrix operations
- [ ] Neural network layers
- [ ] Automatic differentiation
- [ ] Custom kernels
- [ ] Stream management
- [ ] File I/O

## Development Status

**⚠️ Early Development**: This project is in early development. APIs may change.

### Current Limitations

1. Only basic operations are implemented
2. Requires mlx-c to be built and installed separately
3. macOS/Apple Silicon only (following MLX's platform support)
4. Not all MLX features are exposed yet

### Roadmap

- [ ] Complete array operations API
- [ ] Neural network module (mlx.nn equivalent)
- [ ] Optimizer implementations
- [ ] Automatic differentiation (grad, value_and_grad)
- [ ] Examples: LLM inference, training, etc.
- [ ] Documentation and tutorials
- [ ] Benchmarks against mlx-python


## Contributing

Contributions are welcome! This project follows the same spirit as MLX - designed by ML researchers for ML researchers.

<!-- ### Areas Needing Help

1. Wrapping more mlx-c functions
2. Writing examples and documentation
3. Testing on different macOS/Apple Silicon configurations
4. Performance benchmarking -->

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
