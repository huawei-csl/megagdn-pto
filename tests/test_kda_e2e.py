#!/usr/bin/env python3
"""End-to-end KDA pipeline test: NPU-accelerated stages vs CPU float32 reference.

Stages implemented on NPU (PTO kernels):
  [1] gate_cumsum_kda — within-chunk prefix sum of g [B, T, HV, K]

Stages using CPU reference (placeholders until NPU kernels are implemented):
  [2] kkt_kda       — L matrix (gated K×K product)
  [3] inversion_kda — (I + L)^{-1}
  [4] wy_kda        — u and w transforms
  [5] chunk_h_kda   — sequential state pass (snapshots + v_corr)
  [6] chunk_o_kda   — output pass

Usage::

    python tests/test_kda_e2e.py --device npu:0
    python tests/test_kda_e2e.py --device npu:0 --H 4 --HV 8
    python tests/test_kda_e2e.py --device npu:0 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# Both helper modules are in the same tests/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_kda_single_kernels import (
    ref_gate_cumsum,
    ref_kkt_kda,
    ref_inversion_kda,
    ref_wy_kda,
    ref_chunk_h_kda,
    ref_chunk_o_kda,
    cpu_pipeline_kda,
    stats_ok,
    K,
    V_DIM,
    CHUNK,
)

# ---------------------------------------------------------------------------
# PTO pipeline (gate_cumsum on NPU; stages 2-6 are CPU reference placeholders)
# ---------------------------------------------------------------------------

def pto_pipeline_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta_sig: torch.Tensor,
    cu_seqlens_list,
    scale: float,
    dev: torch.device,
    chunk_size: int = CHUNK,
) -> torch.Tensor:
    """KDA pipeline with stage 1 on NPU and stages 2-6 as CPU reference placeholders.

    Args:
        q:               [B, T, H,  K]  float32, L2-normalised queries
        k:               [B, T, H,  K]  float32, L2-normalised keys
        v:               [B, T, HV, V]  float32
        g_log:           [B, T, HV, K]  float32, log-space per-dim gate values
        beta_sig:        [B, T, HV]     float32, post-sigmoid beta in (0, 1)
        cu_seqlens_list: list[int] | None
        scale:           float, query scale (typically K**-0.5)
        dev:             NPU device
        chunk_size:      int, must match compiled kernel

    Returns:
        o: [B, T, HV, V]  float32
    """
    from megagdn_pto.kda_kernel_libs import run_gate_cumsum_kda

    B, T, H, Kd = q.shape
    HV = v.shape[2]
    G  = HV // H

    # GQA expansion + scale (mirrors cpu_pipeline_kda)
    qf = q.float().repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    kf = k.float().repeat_interleave(G, dim=2)           # [B, T, HV, K]
    vf = v.float()
    bf = beta_sig.float()

    # ── Stage 1: gate_cumsum_kda — NPU PTO kernel ────────────────────────────
    g_dev     = g_log.to(dev)
    g_sum_dev = torch.empty_like(g_dev)
    cu_dev    = (torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dev)
                 if cu_seqlens_list else None)
    N_seq  = len(cu_seqlens_list) - 1 if cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_
    run_gate_cumsum_kda(
        g_dev, g_sum_dev,
        stream=stream,
        chunk_size=chunk_size,
        cu_seqlens=cu_dev,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()
    g_cs = g_sum_dev.cpu()   # [B, T, HV, K]  float32

    # ── Stages 2-6: CPU reference placeholders ───────────────────────────────
    L    = ref_kkt_kda(kf, g_cs, bf, chunk_size, cu_seqlens_list)
    INV  = ref_inversion_kda(L, chunk_size, cu_seqlens_list)
    u, w = ref_wy_kda(kf, vf, g_cs, bf, INV, chunk_size, cu_seqlens_list)
    s_snapshots, v_corr = ref_chunk_h_kda(kf, u, w, g_cs, chunk_size, cu_seqlens_list)
    return ref_chunk_o_kda(qf, kf, v_corr, s_snapshots, g_cs, chunk_size, cu_seqlens_list)


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------

TEST_SHAPES: list[tuple] = [
    (16,  None),
    (32,  None),
    (64,  None),
    (128, None),
    ([0, 16, 32],       32),
    ([0, 32, 48],       48),
    ([0, 16, 32, 48],   48),
]


# ---------------------------------------------------------------------------
# Single-shape runner
# ---------------------------------------------------------------------------

def run_one(
    T_or_cu,
    T_total,
    H: int,
    HV: int,
    dev: torch.device,
    scale: float,
    chunk_size: int,
) -> tuple[bool, str]:
    cu_list = T_or_cu if isinstance(T_or_cu, list) else None
    T       = T_total if cu_list else T_or_cu
    label   = f"varlen {cu_list}" if cu_list else f"T={T}"

    torch.manual_seed(0)
    q        = F.normalize(torch.randn(1, T, H,  K),     dim=-1, p=2)
    k        = F.normalize(torch.randn(1, T, H,  K),     dim=-1, p=2)
    v        = torch.randn(1, T, HV, V_DIM)
    g_log    = -torch.rand(1, T, HV, K) * 2.0    # values in (-2, 0)
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))

    # PTO pipeline (stage 1 on NPU)
    o_pto = pto_pipeline_kda(q, k, v, g_log, beta_sig, cu_list, scale, dev, chunk_size)

    # CPU reference pipeline (all stages on CPU)
    o_cpu = cpu_pipeline_kda(q, k, v, g_log, beta_sig, cu_list, scale, chunk_size)

    ok = stats_ok(o_pto.float(), o_cpu.float())
    return ok, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--H",  type=int, default=4, help="Number of Q/K heads")
    parser.add_argument("--HV", type=int, default=None,
                        help="Number of V/gate heads (default: same as H)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    parser.add_argument("--quick", action="store_true", help="Run only T=16 fixed-length")
    args = parser.parse_args()

    H          = args.H
    HV         = args.HV if args.HV is not None else H
    chunk_size = args.chunk_size
    assert HV % H == 0, f"HV={HV} must be a multiple of H={H}"

    torch.npu.set_device(args.device)
    dev   = torch.device(args.device)
    scale = K ** -0.5

    shapes = [(16, None)] if args.quick else TEST_SHAPES

    print(f"device={args.device}  H={H}  HV={HV}  K={K}  V={V_DIM}  CHUNK={chunk_size}")
    print(f"NPU stages:         [1] gate_cumsum_kda")
    print(f"CPU placeholder stages: [2] kkt  [3] inversion  [4] wy  [5] chunk_h  [6] chunk_o")
    print(f"\n{'='*60}")

    all_pass = True
    for i, (T_or_cu, T_total) in enumerate(shapes):
        t0 = time.time()
        ok, label = run_one(T_or_cu, T_total, H, HV, dev, scale, chunk_size)
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{i+1:2d}/{len(shapes)}] {status}  {label}  ({dt:.2f}s)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
