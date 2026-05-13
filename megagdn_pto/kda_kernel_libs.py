"""Loaders and runners for individual KDA PTO kernels.

KDA (Kimi Linear Attention / GDN variant) pipeline:
  gate_cumsum_kda → kkt_kda → inversion_kda → wy_kda → chunk_h_kda → chunk_o_kda

This module provides standalone wrappers for each stage, analogous to the GDN
wrappers in ``megagdn_pto.kernel_libs``.  The full fused pipeline is in
``megagdn_pto.kda_mega_kernel``.

Key shape difference from GDN:
  - GDN gates: ``[B, T, H]``     (scalar per token/head)
  - KDA gates: ``[B, T, HV, K]`` (K-vector per token/head)
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import (
    BLOCK_DIM,
    _KERNELS_PTO,
    compile_chunk_kernel,
)
from megagdn_pto.kernel_libs import _ensure_int32, _vp


def _mtime(name: str) -> int:
    return os.stat(os.path.join(_KERNELS_PTO, name)).st_mtime_ns


# ---------------------------------------------------------------------------
# gate_cumsum_kda — within-chunk prefix sum of g [B, T, HV, K]
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_gate_cumsum_kda(
    num_heads: int,
    k_dim: int = 128,
    chunk_size: int = 16,
) -> ctypes.CDLL:
    """Compile + load the KDA gate cumsum kernel.

    Template parameters injected at compile time:
        GDN_H = num_heads  (HV, number of value/gate heads)
        GDN_D = k_dim      (K, key/gate vector dimension)
        GDN_C = chunk_size (C, tokens per chunk)

    C signature::
        void call_kernel(uint32_t block_dim, void *stream,
                         uint8_t *g, uint8_t *g_sum, uint8_t *cu_seqlens,
                         int64_t batch_size, int64_t seq_len)
    """
    lib_path = compile_chunk_kernel(
        "gate_cumsum_kda.cpp",
        "gate_cumsum_kda",
        num_heads=num_heads,
        hidden_size=k_dim,
        chunk_size=chunk_size,
        key_heads=None,
        cpp_mtime_ns=_mtime("gate_cumsum_kda.cpp"),
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int64, ctypes.c_int64]
    )
    lib.call_kernel.restype = None
    return lib


def run_gate_cumsum_kda(
    g: torch.Tensor,
    g_sum: torch.Tensor,
    *,
    stream,
    chunk_size: int = 16,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Compute within-chunk cumulative sum of KDA gate logits into ``g_sum``.

    Args:
        g:       ``[B, T, HV, K]`` float32, raw per-dimension gate values.
        g_sum:   ``[B, T, HV, K]`` float32, output (cumulative sums).
        stream:  NPU stream handle (``torch.npu.current_stream()._as_parameter_``).
        chunk_size: Tokens per chunk C.  Must match the compiled kernel.
        cu_seqlens: ``int32`` cumulative sequence lengths for packed varlen input.
                    If ``None``, assumes a single sequence of length ``T``.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim: AI-Core count; auto-detected if ``None``.
    """
    assert g.dtype == torch.float32 and g_sum.dtype == torch.float32
    assert g.shape == g_sum.shape

    HV = g.shape[2]
    K  = g.shape[3]
    T  = g.shape[1]
    bd = block_dim or BLOCK_DIM
    batch = g.shape[0] if batch_size_override is None else batch_size_override
    cu32 = _ensure_int32(cu_seqlens)

    lib = load_gate_cumsum_kda(HV, K, chunk_size)
    lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu32), batch, T)
