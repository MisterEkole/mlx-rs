#!/usr/bin/env python3
"""
bench_torch.py — PyTorch MPS benchmark suite.

Same tasks as benchmark.rs and bench_mlx.py, but using
PyTorch's MPS backend on Apple Silicon.

    pip install torch
    python bench_torch.py
"""

import time
import torch
import torch.nn as tnn
import torch.optim as toptim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def sync():
    """Ensure MPS ops are complete before timing."""
    if device.type == "mps":
        torch.mps.synchronize()

def bench(name: str, warmup: int, iters: int, fn):
    for _ in range(warmup):
        fn()
        sync()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
        sync()
    elapsed = (time.perf_counter() - start) * 1000

    per_iter = elapsed / iters
    print(f"  {name:<40} {elapsed:>8.2f} ms total | {per_iter:>8.3f} ms/iter  ({iters} iters)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Raw Matmul Throughput
# ═══════════════════════════════════════════════════════════════════════════

def bench_matmul():
    print("\n══ MATMUL THROUGHPUT ══")
    for size in [256, 512, 1024, 2048, 4096]:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        def run():
            c = a @ b
            # Force sync
            _ = c.shape

        bench(f"matmul {size}x{size}", 3, 50, run)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Element-wise Operations
# ═══════════════════════════════════════════════════════════════════════════

def bench_elementwise():
    print("\n══ ELEMENT-WISE OPS (1M elements) ══")
    n = 1_000_000
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)

    bench("add",      3, 200, lambda: a + b)
    bench("multiply", 3, 200, lambda: a * b)
    bench("exp",      3, 200, lambda: torch.exp(a))
    bench("sin",      3, 200, lambda: torch.sin(a))
    bench("sqrt",     3, 200, lambda: torch.sqrt(torch.abs(a)))


# ═══════════════════════════════════════════════════════════════════════════
# 3. MLP Training Step
# ═══════════════════════════════════════════════════════════════════════════

class MLP(tnn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = tnn.Sequential(
            tnn.Linear(input_dim, hidden_dim),
            tnn.ReLU(),
            tnn.Linear(hidden_dim, hidden_dim),
            tnn.ReLU(),
            tnn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def bench_mlp_training():
    print("\n══ MLP TRAINING STEP ══")

    configs = [
        (32, 128, 256, 10),
        (64, 512, 1024, 100),
        (128, 784, 2048, 10),
    ]

    for batch, input_dim, hidden, output in configs:
        model = MLP(input_dim, hidden, output).to(device)
        optimizer = toptim.Adam(model.parameters(), lr=1e-3)
        criterion = tnn.CrossEntropyLoss()

        x = torch.randn(batch, input_dim, device=device)
        labels = torch.randint(0, output, (batch,), device=device)

        def step():
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        label = f"MLP fwd+bwd+update B={batch} [{input_dim}->{hidden}->{hidden}->{output}]"
        bench(label, 3, 100, step)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CNN Training Step
# ═══════════════════════════════════════════════════════════════════════════

class CNN(tnn.Module):
    def __init__(self):
        super().__init__()
        self.net = tnn.Sequential(
            # PyTorch expects NCHW, so input is (B, 1, 28, 28)
            tnn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            tnn.ReLU(),
            tnn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            tnn.ReLU(),
            tnn.Flatten(),
            tnn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def bench_cnn_training():
    print("\n══ CNN TRAINING STEP (MNIST-like) ══")
    batch = 32
    model = CNN().to(device)
    optimizer = toptim.Adam(model.parameters(), lr=1e-3)
    criterion = tnn.CrossEntropyLoss()

    # PyTorch uses NCHW layout
    x = torch.randn(batch, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (batch,), device=device)

    def step():
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    bench("CNN fwd+bwd+update B=32 [28x28x1 -> 10]", 3, 100, step)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Transformer Encoder Forward
# ═══════════════════════════════════════════════════════════════════════════

def bench_transformer():
    print("\n══ TRANSFORMER ENCODER FORWARD ══")

    configs = [
        (64, 4, 2, 32, 16),
        (128, 4, 4, 64, 16),
        (256, 8, 6, 128, 8),
    ]

    for d_model, n_heads, n_layers, seq_len, batch in configs:
        d_ff = d_model * 4
        encoder_layer = tnn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True,
        )
        encoder = tnn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)
        encoder.eval()

        x = torch.randn(batch, seq_len, d_model, device=device)

        def run():
            with torch.no_grad():
                out = encoder(x)
                _ = out.shape

        label = f"Encoder fwd B={batch} S={seq_len} d={d_model} h={n_heads} L={n_layers}"
        bench(label, 3, 50, run)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print(f"║          PyTorch MPS Benchmark Suite             ║")
    print(f"║          Device: {device}                            ║")
    print(f"║          PyTorch: {torch.__version__:<32}║")
    print("╚══════════════════════════════════════════════════╝")

    bench_matmul()
    bench_elementwise()
    bench_mlp_training()
    bench_cnn_training()
    bench_transformer()

    print("\n══ DONE ══")