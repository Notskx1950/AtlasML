# scripts/benchmark_edge.py

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Model definition ---
class TinyMLP(nn.Module):
    # Using a simple MLP for demonstration; in practice, this could be any model you want to benchmark.
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    digits = load_digits()

    x = digits.data.astype("float32")
    y = digits.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )

def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
) -> nn.Module:
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

@torch.no_grad()
def accuracy_torch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(x)
    preds = logits.argmax(dim=1)
    return accuracy_score(y.numpy(), preds.numpy())

def percentile(values: list[float], pct: float) -> float:
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * pct)
    return sorted_values[index]


@torch.no_grad()
def benchmark_torch_latency(
    model: nn.Module,
    sample: torch.Tensor,
    runs: int,
    warmup: int,
) -> tuple[float, float]:
    model.eval()

    for _ in range(warmup):
        _ = model(sample)

    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(sample)
        latencies.append((time.perf_counter() - start) * 1000)

    return percentile(latencies, 0.50), percentile(latencies, 0.95)

def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def save_torch_state_dict(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    return file_size_mb(path)

def quantize_dynamic(model: nn.Module) -> nn.Module:
    model.eval()
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

def export_torchscript(model: nn.Module, example: torch.Tensor, path: Path) -> torch.jit.ScriptModule:
    model.eval()
    traced = torch.jit.trace(model, example)
    traced.save(str(path))
    return traced

def export_onnx(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

def accuracy_onnx(session: ort.InferenceSession, x: torch.Tensor, y: torch.Tensor) -> float:
    logits = session.run(None, {"input": x.numpy()})[0]
    preds = logits.argmax(axis=1)
    return accuracy_score(y.numpy(), preds)


def benchmark_onnx_latency(
    session: ort.InferenceSession,
    sample: torch.Tensor,
    runs: int,
    warmup: int,
) -> tuple[float, float]:
    input_dict = {"input": sample.numpy()}

    for _ in range(warmup):
        _ = session.run(None, input_dict)

    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(None, input_dict)
        latencies.append((time.perf_counter() - start) * 1000)

    return percentile(latencies, 0.50), percentile(latencies, 0.95)

@dataclass
class EdgeBenchmarkRow:
    model: str
    format: str
    size_mb: float
    accuracy: float
    p50_ms: float
    p95_ms: float


def print_markdown(rows: list[EdgeBenchmarkRow]) -> None:
    print("| Model | Format | Size MB | Accuracy | P50 Latency ms | P95 Latency ms |")
    print("|---|---|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row.model} "
            f"| {row.format} "
            f"| {row.size_mb:.3f} "
            f"| {row.accuracy:.4f} "
            f"| {row.p50_ms:.3f} "
            f"| {row.p95_ms:.3f} |"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out-dir", default="artifacts/edge_benchmark")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_data()

    model = TinyMLP()
    model = train_model(model, x_train, y_train, epochs=args.epochs)

    sample = x_test[: args.batch_size]

    rows: list[EdgeBenchmarkRow] = []

    # FP32 PyTorch
    fp32_path = out_dir / "tinymlp_fp32.pt"
    fp32_size = save_torch_state_dict(model, fp32_path)
    fp32_acc = accuracy_torch(model, x_test, y_test)
    fp32_p50, fp32_p95 = benchmark_torch_latency(model, sample, args.runs, args.warmup)

    rows.append(
        EdgeBenchmarkRow(
            model="TinyMLP",
            format="FP32 PyTorch",
            size_mb=fp32_size,
            accuracy=fp32_acc,
            p50_ms=fp32_p50,
            p95_ms=fp32_p95,
        )
    )

    # INT8 dynamic quantized
    quantized = quantize_dynamic(model)
    int8_path = out_dir / "tinymlp_int8_dynamic.pt"
    int8_size = save_torch_state_dict(quantized, int8_path)
    int8_acc = accuracy_torch(quantized, x_test, y_test)
    int8_p50, int8_p95 = benchmark_torch_latency(quantized, sample, args.runs, args.warmup)

    rows.append(
        EdgeBenchmarkRow(
            model="TinyMLP",
            format="INT8 Dynamic Quantized",
            size_mb=int8_size,
            accuracy=int8_acc,
            p50_ms=int8_p50,
            p95_ms=int8_p95,
        )
    )

    # TorchScript
    ts_path = out_dir / "tinymlp_torchscript.pt"
    scripted = export_torchscript(model, sample, ts_path)
    ts_size = file_size_mb(ts_path)
    ts_acc = accuracy_torch(scripted, x_test, y_test)
    ts_p50, ts_p95 = benchmark_torch_latency(scripted, sample, args.runs, args.warmup)

    rows.append(
        EdgeBenchmarkRow(
            model="TinyMLP",
            format="TorchScript",
            size_mb=ts_size,
            accuracy=ts_acc,
            p50_ms=ts_p50,
            p95_ms=ts_p95,
        )
    )

    # ONNX Runtime
    onnx_path = out_dir / "tinymlp.onnx"
    export_onnx(model, sample, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_size = file_size_mb(onnx_path)
    onnx_acc = accuracy_onnx(session, x_test, y_test)
    onnx_p50, onnx_p95 = benchmark_onnx_latency(session, sample, args.runs, args.warmup)

    rows.append(
        EdgeBenchmarkRow(
            model="TinyMLP",
            format="ONNX Runtime",
            size_mb=onnx_size,
            accuracy=onnx_acc,
            p50_ms=onnx_p50,
            p95_ms=onnx_p95,
        )
    )

    print_markdown(rows)


if __name__ == "__main__":
    main()