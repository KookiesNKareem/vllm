# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the dp4a W4A8 decode GEMM against the layer path it replaces.

Usage:
    python benchmarks/kernels/benchmark_w4a8_dp4a.py [--shapes NxK ...]
"""

import argparse

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.dp4a import _pack_w4

GROUP = 128


def bench_us(fn, iters: int = 200) -> float:
    graph = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["4096x4096", "11008x4096", "4096x11008", "4096x14336"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    for shape in args.shapes:
        n, k = (int(v) for v in shape.split("x"))
        torch.manual_seed(0)
        w_s4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
        scales = (torch.rand(n, k // GROUP, device="cuda") * 0.9 + 0.1).to(dtype)
        packed = _pack_w4(w_s4)
        folded = (scales.float() / 16.0).to(dtype)
        for bs in args.batches:
            x = torch.rand(bs, k, device="cuda", dtype=dtype) - 0.5
            out = torch.empty(bs, n, dtype=dtype, device="cuda")

            def run() -> None:
                x_q, x_s, _ = ops.scaled_int8_quant(x, symmetric=True)
                ops.w4a8_dp4a_gemm(out, x_q, x_s.view(-1).float(), packed, folded)

            us = bench_us(run)
            gib = (n * k / 2 + n * k // GROUP * 2 + bs * k + bs * n * 2) / us / 1e3
            print(f"{shape} bs={bs}: {us:7.1f} us  ({gib:6.0f} GB/s)")


if __name__ == "__main__":
    main()
