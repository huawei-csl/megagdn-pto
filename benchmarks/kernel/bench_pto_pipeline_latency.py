#!/usr/bin/env python3
"""Time PTO megakernel vs PTO staged vs optional Triton six-stage chain (BT=64).

Measures end-to-end NPU timings for the same varlen-packed shape as the
micro-benchmarks (``N_seq`` sequences × ``L_seg`` tokens):

  * **mega_ms** — fused ``run_mega_kernel`` (PTO C=128).
  * **separate_ms** — six PTO kernel launches chained without inter-stage synchronize
    (``chunk_gated_delta_rule._staged_forward`` ordering; cf. ``tests/test_e2e.pto_pipeline``).
  * **triton_bt64_chain_ms** — chained Triton pipeline at chunk/BT=64 via
    ``tests/test_e2e.triton_pipeline``. Omitted when import, compile, or launch fails.

``_bench_loop`` records events around ``fn()`` and synchronizes once per trial after timing.

Outputs JSON for ``scripts/plot_results.py --pto-pipeline-json``.

Usage::

    GDN_NPU_DEVICE=npu:0 python benchmarks/kernel/bench_pto_pipeline_latency.py \\
        --output-json outputs/data/pto_pipeline_latency.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bench_triton_e2e(
    *,
    q_w,
    k_w,
    v_w,
    g_w,
    beta_w,
    cu_long: torch.Tensor,
    H: int,
    Hg: int,
    scale: float,
    warmup: int,
    iters: int,
    error_note: list[str],
) -> float | None:
    """``tests/test_e2e.triton_pipeline`` timings, or ``None`` on failure."""
    from tests.test_e2e import _triton_available, triton_pipeline

    if not _triton_available():
        error_note.append("triton unavailable")
        return None

    def run_triton():
        triton_pipeline(q_w, k_w, v_w, g_w, beta_w, cu_long, H, Hg, scale=scale)

    try:
        torch.npu.synchronize()
        run_triton()
        torch.npu.synchronize()
        return _bench_loop(run_triton, warmup=warmup, iters=iters)
    except Exception as e:
        line = f"{type(e).__name__}: {e}".split("\n")[0][:200]
        error_note.append(line)
        return None


def _load_chunk_patch():
    vp = _REPO_ROOT / "vllm_patch" / "chunk_gated_delta_rule.py"
    spec = importlib.util.spec_from_file_location("pto_chunk_patch", vp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {vp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench_loop(fn, *, warmup: int = 4, iters: int = 18) -> float:
    """Mean elapsed ms per call (includes final sync inside timing loop)."""
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=torch.npu.current_device())

    def _timed():
        fn()
        torch.npu.synchronize()

    for _ in range(warmup):
        _timed()
    torch.npu.synchronize()
    times: list[float] = []
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
        ends[i].synchronize()
        times.append(starts[i].elapsed_time(ends[i]))
    return sum(times) / len(times)


def staged_separate_launch(
    patch_mod,
    *,
    q,
    k,
    v,
    g,
    beta,
    cu32,
    scale,
    Hg,
    H,
    tri_inv_func,
):
    """Same stages as `_staged_forward` except no inter-stage synchronize."""
    C_PTO = patch_mod.C_PTO
    dev = q.device
    T = q.shape[1]
    Ddim = q.shape[3]
    N_seq = int(cu32.numel()) - 1
    stream = torch.npu.current_stream()._as_parameter_

    from megagdn_pto.kernel_libs import (
        run_chunk_h,
        run_chunk_o,
        run_scaled_dot_kkt,
        run_wy_fast,
        total_chunks,
        transpose_beta,
        transpose_gates,
    )
    from megagdn_pto.fast_inverse import solve_tril

    g_sum = patch_mod._chunk_cumsum(g.float(), cu32, C_PTO)
    g_t = transpose_gates(g_sum)
    beta_t = transpose_beta(beta)

    msk_lower = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1).float()
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()

    A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(
        k, beta, g_sum, msk_lower, A,
        stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
        cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg,
    )

    A_inv = solve_tril(A, cu32, C_PTO, H, tri_inv_func)

    w = torch.empty_like(v)
    u = torch.empty_like(v)
    run_wy_fast(
        k, v, beta, g_sum, A_inv, w, u,
        stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
        cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg,
    )

    tc_n = total_chunks(N_seq, T, C_PTO, cu32)
    s = torch.zeros(tc_n * H, Ddim, Ddim, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs = torch.zeros(N_seq * H, Ddim, Ddim, device=dev, dtype=torch.float16)
    run_chunk_h(
        k, w, u, g_sum, s, v_new, fs,
        stream=stream, g_t=g_t, chunk_size=C_PTO,
        cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg,
    )

    o = torch.empty_like(v)
    run_chunk_o(
        q, k, v_new, s, g_sum, msk_full, o,
        stream=stream, g_t=g_t, chunk_size=C_PTO,
        cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg,
    )
    return (o * scale).to(q.dtype), fs


def _build_tensors(T: int, H: int, Hg: int, dev: torch.device):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(1, T, Hg, 128, device=dev, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, Hg, 128, device=dev, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, 128, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g = torch.randn(1, T, H, device=dev, dtype=torch.float32).sigmoid().log()
    return q, k, v, g, beta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--l-list", default="8192,16384",
                    help="Comma-separated L_seg values (tokens per seq).")
    ap.add_argument("--h-list", default="16,32,48,64",
                    help="Comma-separated head counts H (value heads).")
    ap.add_argument("--hg", type=int, default=16)
    ap.add_argument("--output-json", type=Path,
                    default=_REPO_ROOT / "outputs" / "data" / "pto_pipeline_latency.json")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16)
    args = ap.parse_args()

    patch_mod = _load_chunk_patch()
    from megagdn_pto.fast_inverse import load_tri_inverse
    from megagdn_pto.mega_kernel import run_mega_kernel

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    stream = torch.npu.current_stream()._as_parameter_
    tri = load_tri_inverse()
    scale = 128 ** -0.5
    Hg = args.hg

    N_seq = args.n_seq
    L_list = [int(x.strip()) for x in args.l_list.split(",") if x.strip()]
    H_list = [int(x.strip()) for x in args.h_list.split(",") if x.strip()]
    results: list[dict] = []

    print(f"# N_seq={N_seq}  warm={args.warmup}  iters={args.iters}  device={args.device}")

    for L_seg in L_list:
        T = N_seq * L_seg
        cu32 = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
        for H in H_list:
            assert H % Hg == 0, f"H={H} must be divisible by hg={Hg}"
            q16, k16, v16, g32, beta16 = _build_tensors(T, H, Hg, dev)
            q_w = q16.to(torch.float16).contiguous()
            k_w = k16.contiguous()
            v_w = v16.contiguous()
            beta_w = beta16.contiguous()
            g_w = g32.float().contiguous()
            cu = cu32.contiguous()

            def run_mega():
                run_mega_kernel(q_w, k_w, v_w, g_w, beta_w, cu, stream=stream,
                               chunk_size=patch_mod.C_PTO, scale=scale, key_heads=Hg)

            def run_separate():
                staged_separate_launch(
                    patch_mod,
                    q=q_w, k=k_w, v=v_w, g=g_w, beta=beta_w,
                    cu32=cu, scale=scale, Hg=Hg, H=H,
                    tri_inv_func=tri,
                )

            torch.npu.synchronize()
            run_mega()
            torch.npu.synchronize()
            ms_mega = _bench_loop(run_mega, warmup=args.warmup, iters=args.iters)

            ms_sep = _bench_loop(run_separate, warmup=args.warmup, iters=args.iters)

            tri_note: list[str] = []
            ms_triton = _bench_triton_e2e(
                q_w=q_w,
                k_w=k_w,
                v_w=v_w,
                g_w=g_w,
                beta_w=beta_w,
                cu_long=cu.long(),
                H=H,
                Hg=Hg,
                scale=scale,
                warmup=args.warmup,
                iters=args.iters,
                error_note=tri_note,
            )

            row = {
                "N_seq": N_seq,
                "L_seg": L_seg,
                "H": H,
                "Hg": Hg,
                "D": 128,
                "C_pto": int(patch_mod.C_PTO),
                "mega_ms": ms_mega,
                "separate_ms": ms_sep,
                "separate_over_mega": ms_sep / ms_mega if ms_mega > 0 else None,
            }
            if ms_triton is not None:
                row["triton_bt64_chain_ms"] = ms_triton
                row["triton_chain_over_mega"] = ms_triton / ms_mega if ms_mega > 0 else None
            elif tri_note:
                row["triton_bt64_chain_note"] = tri_note[-1]

            results.append(row)
            tri_disp = (
                f"  triton(BT64)={ms_triton:.2f} ms"
                if ms_triton is not None
                else f"  triton(skip: {tri_note[-1]})" if tri_note else "  triton(skip)"
            )
            print(
                f"  L_seg={L_seg} H={H}: mega={ms_mega:.2f} ms  separate={ms_sep:.2f} ms{tri_disp}"
            )

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "N_seq": N_seq,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
