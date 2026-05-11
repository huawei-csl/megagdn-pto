#!/usr/bin/env python3
"""End-to-end accuracy test: cpu_pipeline_kda vs naive_chunk_kda (ground truth).

Uses ``kda_naive.naive_chunk_kda`` as the gold reference and checks that
``test_kda_single_kernels.cpu_pipeline_kda`` matches it across a range of
fixed-length and variable-length shapes.

Shapes must satisfy ``T % chunk_size == 0`` (naive_chunk_kda requirement).

Usage::

    python tests/test_kda_e2e.py
    python tests/test_kda_e2e.py --quick
    python tests/test_kda_e2e.py --H 4 --hv 8     # GQA: 4 query heads, 8 value heads
    python tests/test_kda_e2e.py --H-list 4,8,16
    python tests/test_kda_e2e.py --chunk-size 32
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Both helper modules are in the same tests/ directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kda_naive import naive_chunk_kda
from test_kda_single_kernels import cpu_pipeline_kda, stats_ok, K, V_DIM as V, CHUNK

# ---------------------------------------------------------------------------
# Test shapes  (T must be divisible by chunk_size)
# ---------------------------------------------------------------------------

def _make_shapes(chunk_size: int) -> list[tuple]:
    """Return (T_or_cu_list, T_total) pairs where T_total % chunk_size == 0."""
    cs = chunk_size
    return [
        (cs,          None),
        (cs * 2,      None),
        (cs * 4,      None),
        (cs * 8,      None),
        # varlen: each sequence length also a multiple of cs (naive processes full chunks)
        ([0, cs, cs * 2],           cs * 2),
        ([0, cs, cs * 3, cs * 4],   cs * 4),
        ([0, cs * 2, cs * 3, cs * 6], cs * 6),
    ]


# ---------------------------------------------------------------------------
# Single-shape runner
# ---------------------------------------------------------------------------

def run_one(
    T_or_cu,
    T_total,
    H: int,
    HV: int,
    scale: float,
    chunk_size: int,
) -> tuple[bool, str]:
    cu_list = T_or_cu if isinstance(T_or_cu, list) else None
    T = T_total if cu_list else T_or_cu
    label = f"varlen {cu_list}" if cu_list else f"T={T}"

    torch.manual_seed(0)
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    # L2-normalize q and k: the real KDA model always normalizes, and unnormalized
    # keys with K=128 produce ill-conditioned L matrices that cause float32 blowup.
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(1, T, HV, V)
    # g: log-space per-dim gates, values in (-2, 0) so exp(g) in (0.14, 1)
    g_log = -torch.rand(1, T, HV, K) * 2.0
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))

    # ---- Our CPU reference ----
    o_ours = cpu_pipeline_kda(
        q, k, v, g_log, beta_sig,
        cu_list,
        scale, chunk_size,
    )

    # ---- naive_chunk_kda (ground truth) ----
    # naive requires T % chunk_size == 0 and contiguous fixed-length sequences;
    # for varlen we test each sequence independently then concatenate.
    if cu_list is None:
        o_naive, _ = naive_chunk_kda(
            q, k, v, g_log, beta_sig,
            scale=scale, chunk_size=chunk_size,
        )
        o_naive = o_naive.float()
    else:
        segs = []
        for i in range(len(cu_list) - 1):
            bos, eos = cu_list[i], cu_list[i + 1]
            seq_len = eos - bos
            assert seq_len % chunk_size == 0, (
                f"Sequence length {seq_len} not divisible by chunk_size {chunk_size}"
            )
            q_s = q[:, bos:eos]
            k_s = k[:, bos:eos]
            v_s = v[:, bos:eos]
            g_s = g_log[:, bos:eos]
            b_s = beta_sig[:, bos:eos]
            o_s, _ = naive_chunk_kda(q_s, k_s, v_s, g_s, b_s, scale=scale, chunk_size=chunk_size)
            segs.append(o_s.float())
        o_naive = torch.cat(segs, dim=1)

    ok = stats_ok(o_ours, o_naive)
    return ok, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--hv", type=int, default=None,
                        help="Number of value heads HV (default: same as H).")
    parser.add_argument("--H-list", default=None, help="Comma-separated H values.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    args = parser.parse_args()

    chunk_size = args.chunk_size
    heads = [int(x) for x in args.H_list.split(",")] if args.H_list else [args.H]
    scale = K ** -0.5

    shapes = [(chunk_size, None)] if args.quick else _make_shapes(chunk_size)

    print(f"K={K}  V={V}  chunk_size={chunk_size}  shapes={len(shapes)}")

    all_pass = True
    for H in heads:
        HV = args.hv if args.hv is not None else H
        if HV % H != 0:
            sys.exit(f"HV={HV} must be divisible by H={H}")
        print(f"\n{'='*60}\nH={H}  HV={HV}\n{'='*60}")
        for T_or_cu, T_total in shapes:
            ok, label = run_one(T_or_cu, T_total, H, HV, scale, chunk_size)
            if not ok:
                all_pass = False
            print(f"  {'PASS' if ok else 'FAIL'}  {label}  ours_vs_naive={'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
