#!/usr/bin/env python3
"""
bench_mlx.py — Python MLX benchmark suite.

Mirrors the Rust mlx-rs benchmark (examples/benchmark.rs) so you can
compare language-level overhead on the same MLX C++ backend.

    pip install mlx
    python bench_mlx.py
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bench(name: str, warmup: int, iters: int, fn):
    """Time a function over `iters` iterations after `warmup` warmups."""
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1000  # ms

    per_iter = elapsed / iters
    print(f"  {name:<40} {elapsed:>8.2f} ms total | {per_iter:>8.3f} ms/iter  ({iters} iters)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Raw Matmul Throughput
# ═══════════════════════════════════════════════════════════════════════════

def bench_matmul():
    print("\n══ MATMUL THROUGHPUT ══")
    for size in [256, 512, 1024, 2048, 4096]:
        a = mx.random.uniform(shape=(size, size))
        b = mx.random.uniform(shape=(size, size))

        def run():
            c = a @ b
            mx.eval(c)

        bench(f"matmul {size}x{size}", 3, 50, run)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Element-wise Operations
# ═══════════════════════════════════════════════════════════════════════════

def bench_elementwise():
    print("\n══ ELEMENT-WISE OPS (1M elements) ══")
    n = 1_000_000
    a = mx.random.uniform(shape=(n,))
    b = mx.random.uniform(shape=(n,))

    bench("add",      3, 200, lambda: mx.eval(a + b))
    bench("multiply", 3, 200, lambda: mx.eval(a * b))
    bench("exp",      3, 200, lambda: mx.eval(mx.exp(a)))
    bench("sin",      3, 200, lambda: mx.eval(mx.sin(a)))
    bench("sqrt",     3, 200, lambda: mx.eval(mx.sqrt(mx.abs(a))))


# ═══════════════════════════════════════════════════════════════════════════
# 3. MLP Training Step
# ═══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


def bench_mlp_training():
    print("\n══ MLP TRAINING STEP ══")

    configs = [
        (32, 128, 256, 10),
        (64, 512, 1024, 100),
        (128, 784, 2048, 10),
    ]

    for batch, input_dim, hidden, output in configs:
        model = MLP(input_dim, hidden, output)
        mx.eval(model.parameters())
        optimizer = optim.Adam(learning_rate=1e-3)

        x = mx.random.uniform(shape=(batch, input_dim))
        labels = mx.random.randint(0, output, shape=(batch,))

        def loss_fn(model, x, labels):
            logits = model(x)
            return mx.mean(nn.losses.cross_entropy(logits, labels))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step():
            loss, grads = loss_and_grad(model, x, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        label = f"MLP fwd+bwd+update B={batch} [{input_dim}->{hidden}->{hidden}->{output}]"
        bench(label, 3, 100, step)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CNN Training Step
# ═══════════════════════════════════════════════════════════════════════════

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(1568, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def bench_cnn_training():
    print("\n══ CNN TRAINING STEP (MNIST-like) ══")
    batch = 32
    model = CNN()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=1e-3)

    x = mx.random.uniform(shape=(batch, 28, 28, 1))
    labels = mx.random.randint(0, 10, shape=(batch,))

    def loss_fn(model, x, labels):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, labels))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step():
        loss, grads = loss_and_grad(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    bench("CNN fwd+bwd+update B=32 [28x28x1 -> 10]", 3, 100, step)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Transformer Encoder Forward
# ═══════════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.0):
        super().__init__()
        self.layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x,mask=None)
        return x


def bench_transformer():
    print("\n══ TRANSFORMER ENCODER FORWARD ══")

    configs = [
        (64, 4, 2, 32, 16),
        (128, 4, 4, 64, 16),
        (256, 8, 6, 128, 8),
    ]

    for d_model, n_heads, n_layers, seq_len, batch in configs:
        d_ff = d_model * 4
        encoder = TransformerEncoder(d_model, n_heads, n_layers, d_ff)
        mx.eval(encoder.parameters())

        x = mx.random.uniform(shape=(batch, seq_len, d_model))

        def run():
            out = encoder(x)
            mx.eval(out)

        label = f"Encoder fwd B={batch} S={seq_len} d={d_model} h={n_heads} L={n_layers}"
        bench(label, 3, 50, run)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║          Python MLX Benchmark Suite              ║")
    print("║          Device: Apple Silicon GPU               ║")
    print("╚══════════════════════════════════════════════════╝")

    mx.set_default_device(mx.gpu)

    bench_matmul()
    bench_elementwise()
    bench_mlp_training()
    bench_cnn_training()
    bench_transformer()

    print("\n══ DONE ══")