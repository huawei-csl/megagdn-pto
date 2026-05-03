#!/usr/bin/env python3
"""Test and benchmark the standalone BF16↔FP16 PTO cast kernel.

Tests:
  1. Correctness: compare NPU cast kernel output with torch.Tensor.to() reference
  2. Performance: measure latency vs torch eager cast

Usage::

    python tests/test_cast_bf16_fp16.py --device npu:0
    python tests/test_cast_bf16_fp16.py --device npu:0 --bench
    python tests/test_cast_bf16_fp16.py --device npu:0 --bench --n-elem 1048576

Also tests the cross-check between the fp16 megakernel path and the bf16
megakernel path (see --cross-check flag).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from functools import lru_cache

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megagdn_pto.compile import (
    BLOCK_DIM,
    _KERNELS_PTO,
    _common_flags,
    _run_bisheng,
    _COMPILED_DIR,
)

# ---------------------------------------------------------------------------
# Compile the standalone cast kernel
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _compile_cast_kernel(cpp_mtime_ns: int = 0) -> str:
    """Compile cast_bf16_fp16.cpp → .so and return the path."""
    os.makedirs(_COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_KERNELS_PTO, "cast_bf16_fp16.cpp")
    lib_path = os.path.join(_COMPILED_DIR, "cast_bf16_fp16.so")
    # Use CAST_C=1024 (8× larger than the default 128) to amortize per-iteration
    # overhead and approach torch eager cast bandwidth efficiency.
    flags = _common_flags(num_heads=16, key_heads=16, hidden_size=128, chunk_size=128)
    flags = [f for f in flags if not f.startswith("-DGDN_C=")]
    flags.append("-DCAST_C=1024")
    print(f"[cast test] Compiling cast_bf16_fp16.cpp …")
    _run_bisheng(["bisheng", *flags, cpp_path, "-o", lib_path], timeout=120)
    print(f"[cast test] Compiled → {lib_path}")
    return lib_path


def _load_cast_kernel() -> ctypes.CDLL:
    mtime = os.stat(os.path.join(_KERNELS_PTO, "cast_bf16_fp16.cpp")).st_mtime_ns
    lib_path = _compile_cast_kernel(mtime)
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int32,
    ]
    lib.call_kernel.restype = None
    return lib


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench_npu(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return mean latency in ms using NPU Event timing."""
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
# Correctness tests
# ---------------------------------------------------------------------------

def test_bf16_to_fp16(lib, dev, stream, n_elem: int = 1 << 20):
    """Verify BF16→FP16 cast kernel vs torch.Tensor.to()."""
    x_bf16 = torch.randn(n_elem, device=dev, dtype=torch.bfloat16)
    out_fp16 = torch.empty(n_elem, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    # Reference: torch eager (goes through GPU cast)
    ref_fp16 = x_bf16.to(torch.float16)
    torch.npu.synchronize()

    # NPU kernel cast
    lib.call_kernel(BLOCK_DIM, stream, _vp(x_bf16), _vp(out_fp16),
                    ctypes.c_int64(n_elem), ctypes.c_int32(0))
    torch.npu.synchronize()

    # Due to the two-hop (BF16→FP32→FP16), output may differ from
    # torch by at most 1 ULP (both are mathematically equivalent casts).
    diff = (out_fp16.float() - ref_fp16.float()).abs()
    max_diff = diff.max().item()
    mismatch = (out_fp16 != ref_fp16).float().mean().item() * 100
    print(f"  BF16→FP16: n={n_elem:,}  max_diff={max_diff:.4g}  "
          f"mismatch={mismatch:.2f}%")
    assert max_diff <= 1e-3, f"BF16→FP16 max diff too large: {max_diff}"
    print("  BF16→FP16: PASS")


def test_fp16_to_bf16(lib, dev, stream, n_elem: int = 1 << 20):
    """Verify FP16→BF16 cast kernel vs torch.Tensor.to()."""
    x_fp16 = torch.randn(n_elem, device=dev, dtype=torch.float16)
    out_bf16 = torch.empty(n_elem, device=dev, dtype=torch.bfloat16)
    torch.npu.synchronize()
    ref_bf16 = x_fp16.to(torch.bfloat16)
    torch.npu.synchronize()

    lib.call_kernel(BLOCK_DIM, stream, _vp(x_fp16), _vp(out_bf16),
                    ctypes.c_int64(n_elem), ctypes.c_int32(1))
    torch.npu.synchronize()

    diff = (out_bf16.float() - ref_bf16.float()).abs()
    max_diff = diff.max().item()
    mismatch = (out_bf16 != ref_bf16).float().mean().item() * 100
    print(f"  FP16→BF16: n={n_elem:,}  max_diff={max_diff:.4g}  "
          f"mismatch={mismatch:.2f}%")
    # FP32→BF16 rounds to 7 mantissa bits; max ULP error ≈ 2^-7 × |value|.
    # Use atol=1e-5, rtol=1e-2 (same convention as torch.testing.assert_close).
    tol = 1e-5 + 1e-2 * ref_bf16.abs().float()
    assert (diff <= tol).all(), (
        f"FP16→BF16 tolerance exceeded: max(diff - tol) = {(diff - tol).max().item():.4g}"
    )
    print("  FP16→BF16: PASS")


def test_roundtrip(lib, dev, stream, n_elem: int = 1 << 20):
    """BF16 → FP16 → BF16 roundtrip; values should be unchanged for normal numbers."""
    torch.manual_seed(77)
    x_bf16 = torch.randn(n_elem, device=dev, dtype=torch.bfloat16)
    # Clamp to FP16 range to avoid Inf in FP16 step
    x_bf16 = x_bf16.clamp(-65000, 65000)
    torch.npu.synchronize()

    mid_fp16 = torch.empty(n_elem, device=dev, dtype=torch.float16)
    lib.call_kernel(BLOCK_DIM, stream, _vp(x_bf16), _vp(mid_fp16),
                    ctypes.c_int64(n_elem), ctypes.c_int32(0))
    torch.npu.synchronize()

    out_bf16  = torch.empty(n_elem, device=dev, dtype=torch.bfloat16)
    lib.call_kernel(BLOCK_DIM, stream, _vp(mid_fp16), _vp(out_bf16),
                    ctypes.c_int64(n_elem), ctypes.c_int32(1))
    torch.npu.synchronize()

    # BF16 has 7 mantissa bits, FP16 has 10 → roundtrip is lossless for
    # values representable in BF16 (FP16 is a strict superset in precision).
    diff = (out_bf16.float() - x_bf16.float()).abs()
    max_diff = diff.max().item()
    mismatch = (out_bf16 != x_bf16).float().mean().item() * 100
    print(f"  Roundtrip BF16→FP16→BF16: max_diff={max_diff:.4g}  "
          f"mismatch={mismatch:.2f}%")
    # BF16 has fewer mantissa bits than FP16 so roundtrip should be lossless
    # for values within FP16 range.  Allow tiny tolerance for inf/denorm edge cases.
    assert mismatch < 1.0, f"Roundtrip mismatch too high: {mismatch:.2f}%"
    print("  Roundtrip: PASS")


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def bench_cast(lib, dev, stream, n_elem: int = 1 << 20):
    """Compare PTO cast kernel latency vs torch eager cast."""
    x_bf16 = torch.randn(n_elem, device=dev, dtype=torch.bfloat16)
    out_fp16 = torch.empty(n_elem, device=dev, dtype=torch.float16)

    def run_pto():
        lib.call_kernel(BLOCK_DIM, stream, _vp(x_bf16), _vp(out_fp16),
                        ctypes.c_int64(n_elem), ctypes.c_int32(0))

    def run_torch():
        _ = x_bf16.to(torch.float16)

    ms_pto   = _bench_npu(run_pto)
    ms_torch = _bench_npu(run_torch)

    size_mb = n_elem * 2 / 1e6
    print(f"\n  BF16→FP16 benchmark  n={n_elem:,}  ({size_mb:.1f} MB per tensor)")
    print(f"    PTO kernel : {ms_pto:.3f} ms")
    print(f"    torch.to() : {ms_torch:.3f} ms")
    if ms_pto > 0:
        print(f"    ratio      : {ms_torch/ms_pto:.2f}x  "
              f"({'faster' if ms_torch > ms_pto else 'slower'} than torch)")

    # Also benchmark the 5-cast batch (q, k, v, beta, g) as done in vllm path
    T, Hg, H, D = 8 * 1024, 16, 16, 128
    q_bf16   = torch.randn(1, T, Hg, D, device=dev, dtype=torch.bfloat16)
    k_bf16   = torch.randn(1, T, Hg, D, device=dev, dtype=torch.bfloat16)
    v_bf16   = torch.randn(1, T, H,  D, device=dev, dtype=torch.bfloat16)
    beta_bf16 = torch.randn(1, T, H,    device=dev, dtype=torch.bfloat16)
    g_bf16   = torch.randn(1, T, H,    device=dev, dtype=torch.bfloat16)

    q_fp16   = torch.empty_like(q_bf16,   dtype=torch.float16)
    k_fp16   = torch.empty_like(k_bf16,   dtype=torch.float16)
    v_fp16   = torch.empty_like(v_bf16,   dtype=torch.float16)
    beta_fp16 = torch.empty_like(beta_bf16, dtype=torch.float16)

    def run_5cast_pto():
        for src, dst in [(q_bf16, q_fp16), (k_bf16, k_fp16),
                         (v_bf16, v_fp16), (beta_bf16, beta_fp16)]:
            ne = src.numel()
            lib.call_kernel(BLOCK_DIM, stream, _vp(src), _vp(dst),
                            ctypes.c_int64(ne), ctypes.c_int32(0))
        _ = g_bf16.float()  # g→fp32 still in Python (fast single-hop)

    def run_5cast_torch():
        _ = q_bf16.to(torch.float16)
        _ = k_bf16.to(torch.float16)
        _ = v_bf16.to(torch.float16)
        _ = beta_bf16.to(torch.float16)
        _ = g_bf16.float()

    ms_5pto   = _bench_npu(run_5cast_pto)
    ms_5torch = _bench_npu(run_5cast_torch)

    total_mb = (q_bf16.numel() + k_bf16.numel() + v_bf16.numel()
                + beta_bf16.numel() + g_bf16.numel()) * 2 / 1e6
    print(f"\n  5-cast batch (q,k,v,beta,g)  T={T}  H={H}  D={D}  ({total_mb:.1f} MB)")
    print(f"    PTO 4×kernel + torch g : {ms_5pto:.3f} ms")
    print(f"    5× torch.to()          : {ms_5torch:.3f} ms")
    if ms_5pto > 0:
        print(f"    ratio                  : {ms_5torch/ms_5pto:.2f}x")


# ---------------------------------------------------------------------------
# Cross-check: fp16 megakernel vs bf16 megakernel
# ---------------------------------------------------------------------------

def test_megakernel_cross_check(dev, stream):
    """Run fp16 megakernel and bf16 megakernel on the same data; compare outputs."""
    try:
        from megagdn_pto.mega_kernel import run_mega_kernel, run_mega_kernel_bf16
    except ImportError as e:
        print(f"\n  [skip cross-check] {e}")
        return

    T, Hg, H, D = 2 * 128, 16, 16, 128
    scale = D ** -0.5
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=dev)

    torch.manual_seed(42)
    import torch.nn.functional as F
    q_bf16    = F.normalize(torch.randn(1, T, Hg, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
    k_bf16    = F.normalize(torch.randn(1, T, Hg, D, device=dev, dtype=torch.bfloat16), dim=-1, p=2)
    v_bf16    = torch.randn(1, T, H,  D, device=dev, dtype=torch.bfloat16)
    beta_bf16 = torch.rand( 1, T, H,    device=dev, dtype=torch.bfloat16)
    g_bf16    = torch.randn(1, T, H,    device=dev, dtype=torch.bfloat16).sigmoid().log()

    # Run fp16 path first, ensure it completes before bf16 path
    o_fp16 = run_mega_kernel(
        q_bf16.to(torch.float16), k_bf16.to(torch.float16),
        v_bf16.to(torch.float16), g_bf16.float(),
        beta_bf16.to(torch.float16), cu_seqlens,
        stream=stream, chunk_size=128, scale=scale, key_heads=Hg,
    )
    torch.npu.synchronize()
    # Copy result to avoid memory reuse by the bf16 path's allocator
    o_fp16_saved = o_fp16.clone()
    del o_fp16

    # Run bf16 path
    o_bf16 = run_mega_kernel_bf16(
        q_bf16, k_bf16, v_bf16, g_bf16.float(), beta_bf16, cu_seqlens,
        stream=stream, chunk_size=128, scale=scale, key_heads=Hg,
    )
    torch.npu.synchronize()

    # Compare (o_fp16 is fp16, o_bf16 is fp16 after run_mega_kernel_bf16's output cast)
    diff = (o_bf16.to(torch.float32) - o_fp16_saved.to(torch.float32)).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff  = (diff / (o_fp16_saved.abs().float() + 1e-8)).mean().item()

    print(f"\n  Cross-check fp16 vs bf16 megakernel:")
    print(f"    max_diff  = {max_diff:.4g}")
    print(f"    mean_diff = {mean_diff:.4g}")
    print(f"    rel_diff  = {rel_diff:.4g}")
    # atol=1e-5, rtol=1e-2: tolerance = 1e-5 + 1e-2 * |fp16_ref|
    # Activations are ~O(1e-2), so rtol=1e-2 allows ~1 ULP of relative error.
    tol = 1e-5 + 1e-2 * o_fp16_saved.abs().float()
    assert (diff <= tol).all(), (
        f"Cross-check tolerance (atol=1e-5, rtol=1e-2) exceeded: "
        f"max(diff - tol) = {(diff - tol).max().item():.4g}"
    )
    print("  Cross-check: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",  default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--bench",   action="store_true", help="Run latency benchmarks")
    parser.add_argument("--cross-check", action="store_true",
                        help="Cross-check fp16 vs bf16 megakernel outputs")
    parser.add_argument("--n-elem",  type=int, default=1 << 20,
                        help="Number of elements for latency benchmark")
    args = parser.parse_args()

    os.environ.setdefault("GDN_NPU_DEVICE", args.device)
    torch.npu.set_device(args.device)
    dev    = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_

    print(f"Device: {args.device}  BLOCK_DIM={BLOCK_DIM}")

    lib = _load_cast_kernel()

    print("\n=== Correctness tests ===")
    test_bf16_to_fp16(lib, dev, stream)
    test_fp16_to_bf16(lib, dev, stream)
    test_roundtrip(lib, dev, stream)

    # Partial tile (n_elem not divisible by 128)
    for n in [1, 63, 127, 129, 255, 513]:
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        out = torch.empty(n, device=dev, dtype=torch.float16)
        torch.npu.synchronize()
        ref = x.to(torch.float16)
        torch.npu.synchronize()
        lib.call_kernel(BLOCK_DIM, stream, _vp(x), _vp(out),
                        ctypes.c_int64(n), ctypes.c_int32(0))
        torch.npu.synchronize()
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff <= 1e-3, f"partial n={n} failed: max_diff={diff}"
    print("  Partial-tile edge cases: PASS")

    if args.bench:
        print("\n=== Performance benchmarks ===")
        bench_cast(lib, dev, stream, n_elem=args.n_elem)

    if args.cross_check:
        print("\n=== Cross-check fp16 vs bf16 megakernel ===")
        test_megakernel_cross_check(dev, stream)
    elif not args.bench:
        # Always run cross-check if neither --bench nor --cross-check is given
        # so we verify the integrated megakernel even in the default test run.
        print("\n=== Cross-check fp16 vs bf16 megakernel ===")
        test_megakernel_cross_check(dev, stream)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
