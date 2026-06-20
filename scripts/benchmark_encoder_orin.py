#!/usr/bin/env python
"""Benchmark DINOv3 ViT-B/16 on Jetson Orin NX — B1.11 encoder gate.

Loads the frozen DinoV3Encoder, exports to TensorRT FP16 via torch2trt or
torch.compile with TensorRT backend, and measures median/p99 latency at batch=1
over 100 forward passes.

Decision rule:
  - ViT-B/16 TRT FP16 < 20ms  => keep D=768
  - ViT-B/16 TRT FP16 > 20ms  => switch to ViT-S/16 D=384

Usage (on Orin NX):
    python scripts/benchmark_encoder_orin.py [--warmup 20] [--iters 100]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def benchmark_pytorch(model_id: str, warmup: int, iters: int) -> dict:
    """Benchmark with plain PyTorch (FP16, CUDA)."""
    import torch

    from vllatent.encode.dinov3 import DinoV3Encoder

    encoder = DinoV3Encoder(device="cuda")

    dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    for _ in range(warmup):
        encoder.encode_rgb(dummy_rgb)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder.encode_rgb(dummy_rgb)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "backend": "pytorch_fp16",
        "model_id": model_id,
        "warmup": warmup,
        "iters": iters,
        "median_ms": float(np.median(arr)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def benchmark_trt(model_id: str, warmup: int, iters: int) -> dict | None:
    """Benchmark with TensorRT FP16 via torch.compile or torch2trt."""
    import torch

    try:
        import torch_tensorrt  # noqa: F401
    except ImportError:
        print("[benchmark] torch_tensorrt not available, skipping TRT benchmark")
        return None

    from vllatent.encode.dinov3 import DinoV3Encoder

    encoder = DinoV3Encoder(device="cuda")
    backbone = encoder._backbone

    dummy = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

    try:
        compiled = torch.compile(
            backbone,
            backend="torch_tensorrt",
            options={"precision": torch.float16, "workspace_size": 1 << 30},
        )
        compiled(dummy)
    except Exception as e:
        print(f"[benchmark] TRT compile failed: {e}")
        print("[benchmark] Falling back to pytorch-only benchmark")
        return None

    for _ in range(warmup):
        compiled(dummy)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "backend": "tensorrt_fp16",
        "model_id": model_id,
        "warmup": warmup,
        "iters": iters,
        "median_ms": float(np.median(arr)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark DINOv3 encoder on Orin NX")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args(argv)

    from vllatent.config import Config
    cfg = Config()
    model_id = cfg.encoder.model_id

    print("=" * 60)
    print("  DINOv3 ENCODER BENCHMARK — B1.11 Gate")
    print("=" * 60)
    print(f"  Model:    {model_id}")
    print(f"  Warmup:   {args.warmup}")
    print(f"  Iters:    {args.iters}")

    import torch
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print()

    pt_result = benchmark_pytorch(model_id, args.warmup, args.iters)
    print(f"  [PyTorch FP16]")
    print(f"    median = {pt_result['median_ms']:.2f} ms")
    print(f"    p99    = {pt_result['p99_ms']:.2f} ms")
    print(f"    mean   = {pt_result['mean_ms']:.2f} ms")
    print(f"    min    = {pt_result['min_ms']:.2f} ms")
    print(f"    max    = {pt_result['max_ms']:.2f} ms")
    print()

    trt_result = benchmark_trt(model_id, args.warmup, args.iters)
    if trt_result is not None:
        print(f"  [TensorRT FP16]")
        print(f"    median = {trt_result['median_ms']:.2f} ms")
        print(f"    p99    = {trt_result['p99_ms']:.2f} ms")
        print(f"    mean   = {trt_result['mean_ms']:.2f} ms")
        print(f"    min    = {trt_result['min_ms']:.2f} ms")
        print(f"    max    = {trt_result['max_ms']:.2f} ms")
        print()

    best = trt_result if trt_result is not None else pt_result
    median = best["median_ms"]
    print("  DECISION")
    print("  " + "-" * 56)
    if median < 20.0:
        print(f"  median {median:.2f} ms < 20 ms => KEEP ViT-B/16 (D=768)")
        print("  B1.12 = NO-OP")
    else:
        print(f"  median {median:.2f} ms >= 20 ms => SWITCH to ViT-S/16 (D=384)")
        print("  B1.12 = update EMBED_DIM, PredictorConfig, CLIP lift")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
