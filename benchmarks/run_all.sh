#!/bin/bash
# run_benchmarks.sh — Run all three benchmarks and compare.
#
# Usage:
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh
#
# Prerequisites:
#   - Rust project built in release mode
#   - pip install mlx torch

set -e

RESULTS_DIR="bench_results"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════╗"
echo "║       Apple Silicon ML Benchmark — 3-Way Compare     ║"
echo "║       mlx-rs (Rust) vs MLX (Python) vs PyTorch MPS   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "macOS:   $(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
echo "Rust:    $(rustc --version 2>/dev/null || echo 'not found')"
echo "Python:  $(python3 --version 2>/dev/null || echo 'not found')"
echo ""

# ── 1. Rust mlx-rs ────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/3] Running mlx-rs (Rust) benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo run --release --example bench_mlx_rs2 2>&1 | tee "$RESULTS_DIR/rust.txt"
echo ""

# ── 2. Python MLX ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/3] Running Python MLX benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bench_mlx.py 2>&1 | tee "$RESULTS_DIR/python_mlx.txt"
echo ""

# ── 3. PyTorch MPS ────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/3] Running PyTorch MPS benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 bench_torch.py 2>&1 | tee "$RESULTS_DIR/pytorch.txt"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results saved to $RESULTS_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"