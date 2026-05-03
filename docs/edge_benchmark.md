# AtlasML Edge Benchmark

## Goal

AtlasML Edge Benchmark evaluates small-model deployment formats for on-device inference tradeoffs.

The benchmark compares:

- FP32 PyTorch
- INT8 dynamic quantization
- TorchScript export
- ONNX Runtime export

## Dataset

This benchmark uses the sklearn digits dataset, a small 8x8 image classification dataset.

## Model

The initial model is a TinyMLP with Linear + ReLU layers.

Dynamic quantization is applied to Linear layers.

## Results

| Model | Format | Size MB | Accuracy | P50 Latency ms | P95 Latency ms |
|---|---|---:|---:|---:|---:|
| TinyMLP | FP32 PyTorch | 0.069 | 0.9089 | 0.023 | 0.023 |
| TinyMLP | INT8 Dynamic Quantized | 0.022 | 0.9067 | 0.193 | 0.330 |
| TinyMLP | TorchScript | 0.077 | 0.9089 | 0.016 | 0.018 |
| TinyMLP | ONNX Runtime | 0.002 | 0.9089 | 0.009 | 0.009 |

## Interpretation

This benchmark demonstrates common edge AI tradeoffs:

- FP32 PyTorch is the baseline.
- INT8 dynamic quantization can reduce model size and may improve CPU inference latency.
- TorchScript provides a serialized PyTorch execution format.
- ONNX Runtime provides a portable inference path.

## Limitations

This is a CPU-based local benchmark. It does not represent Qualcomm DSP/NPU performance.
The goal is to demonstrate model optimization workflow and tradeoff analysis.