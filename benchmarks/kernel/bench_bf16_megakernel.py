#!/usr/bin/env python3
"""Benchmark: fp16 megakernel (Python casts) vs bf16 megakernel (kernel-internal casts).

Measures end-to-end latency for each path, including any Python-level
torch.to() overhead, for a range of sequence lengths T.

The bf16 megakernel eliminates 4 separate torch.to(torch.float16) dispatches
(for q, k, v, beta) and replaces them with in-kernel BF16→FP16 cast operations.
The tradeoff:
  - Saved: 4 Python dispatch overhead calls (~100 µs each ≈ 400 µs total)
  - Extra: PTO cast is slower than torch.to() for large cold tensors

Usage::

    GDN_NPU_DEVICE=npu:0 python benchmarks/kernel/bench_bf16_megakernel.py
    GDN_NPU_DEVICE=npu:0 python benchmarks/kernel/bench_bf16_megakernel.py \\
        --H 16 --T-list 256,512,1024,2048,8192,16384 --output-json outputs/data/bench_bf16_mega.json

"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.kernel_libs import total_chunks
from megagdn_pto.mega_kernel import run_mega_kernel, run_mega_kernel_bf16

C_PTO = 128
D = 128


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _bench_npu(fn, warmup: int = 5, iters: int = 15) -> float:
    """Time an NPU function using Event pairs (ms). Cache-flush between iters."""
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    cache  = torch.empty(256 * 1024 * 1024, dtype=torch.int8).npu()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def bench_bf16_vs_fp16(H: int, HG: int, T: int, dev: torch.device,
                        stream) -> dict:
    """Compare fp16+Python-casts path vs bf16-megakernel path.

    Returns a dict with timing in ms and speedup.
    """
    scale = D ** -0.5
    N_seq = 1
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C_PTO, cu_seqlens)

    # Create bf16 inputs (as they arrive from vllm model layers)
    torch.manual_seed(0)
    q_bf16    = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
    k_bf16    = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
    v_bf16    = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    beta_bf16 = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
    g_bf16    = torch.randn(1, T, H, device=dev, dtype=torch.float32).sigmoid().log()

    # ------------------------------------------------------------------
    # Path A: fp16 megakernel with Python casts (current production path)
    # Includes: 4× torch.to(fp16) + g.float() + run_mega_kernel
    # ------------------------------------------------------------------
    def run_fp16_path():
        q_w    = q_bf16.to(torch.float16)
        k_w    = k_bf16.to(torch.float16)
        v_w    = v_bf16.to(torch.float16)
        beta_w = beta_bf16.to(torch.float16)
        run_mega_kernel(q_w, k_w, v_w, g_bf16, beta_w, cu_seqlens,
                        stream=stream, chunk_size=C_PTO, scale=scale, key_heads=HG)

    # Warmup to ensure compilation is done
    run_fp16_path()
    torch.npu.synchronize()

    ms_fp16 = _bench_npu(run_fp16_path)

    # ------------------------------------------------------------------
    # Path B: bf16 megakernel (casts inside kernel)
    # Includes: g.float() + run_mega_kernel_bf16 (no separate fp16 casts)
    # ------------------------------------------------------------------
    def run_bf16_path():
        run_mega_kernel_bf16(q_bf16, k_bf16, v_bf16, g_bf16, beta_bf16, cu_seqlens,
                             stream=stream, chunk_size=C_PTO, scale=scale, key_heads=HG)

    run_bf16_path()
    torch.npu.synchronize()

    ms_bf16 = _bench_npu(run_bf16_path)

    speedup = ms_fp16 / ms_bf16 if ms_bf16 > 0 else float("inf")

    print(f"  T={T:6d}  H={H}  Hg={HG}:  "
          f"fp16+casts={ms_fp16:.3f}ms  bf16-kernel={ms_bf16:.3f}ms  "
          f"speedup={speedup:.3f}x  ({'faster' if speedup > 1 else 'slower'})")

    return {
        "T": T, "H": H, "Hg": HG, "D": D, "C_pto": C_PTO,
        "fp16_cast_plus_kernel_ms": ms_fp16,
        "bf16_kernel_ms": ms_bf16,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--H",  type=int, default=16, help="Value head count H")
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg")
    parser.add_argument("--T-list", default="256,512,1024,2048,4096,8192,16384",
                        help="Comma-separated total token counts")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    os.environ.setdefault("GDN_NPU_DEVICE", args.device)
    torch.npu.set_device(args.device)
    dev    = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_

    T_list = [int(x) for x in args.T_list.split(",")]
    H, HG  = args.H, args.hg
    assert H % HG == 0

    print(f"Device: {args.device}  BLOCK_DIM={BLOCK_DIM}  H={H}  Hg={HG}  D={D}")
    print(f"Benchmarking fp16+Python-casts vs bf16-megakernel path")
    print(f"{'T':>8}  {'H':>4}  {'fp16+casts':>12}  {'bf16-kernel':>13}  {'speedup':>9}")
    print("-" * 60)

    results = []
    for T in T_list:
        row = bench_bf16_vs_fp16(H, HG, T, dev, stream)
        results.append(row)
        gc.collect()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "H": H, "Hg": HG, "D": D, "C_pto": C_PTO,
            "note": (
                "Speedup > 1 means bf16 megakernel is FASTER than fp16+Python-casts. "
                "For short T, Python dispatch savings dominate. "
                "For long T, the slower in-kernel PTO cast may dominate."
            ),
            "results": results,
        }
        out_path.write_text(json.dumps(meta, indent=2))
        print(f"\nSaved: {out_path}")

    print()
    print("Summary:")
    faster = [r for r in results if r["speedup"] > 1.0]
    slower = [r for r in results if r["speedup"] <= 1.0]
    if faster:
        print(f"  bf16 megakernel FASTER: T={[r['T'] for r in faster]}")
    if slower:
        print(f"  bf16 megakernel SLOWER: T={[r['T'] for r in slower]} "
              f"(cast overhead > Python dispatch savings)")


if __name__ == "__main__":
    main()
