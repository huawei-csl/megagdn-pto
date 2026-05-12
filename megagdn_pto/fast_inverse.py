"""Triangular-inverse CubeCore kernel (``tri_inverse``).

Inverts lower-triangular (or upper-triangular with transposed load/store) matrices
of size 16 / 32 / 64 / 128 in fp16 → fp32, using a recursive unrolled algorithm.
Used as the ``solve_tril`` stage in the GDN pipeline.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from megagdn_pto.compile import BLOCK_DIM, compile_tri_inverse, _KERNELS_PTO
from megagdn_pto.kernel_libs import (
    _record_current_stream,
    precomputed_minus_identity,
    total_chunks,
)


def _vp(t: torch.Tensor | None) -> ctypes.c_void_p:
    if t is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(t.data_ptr())


@lru_cache(maxsize=1)
def _tri_inverse_cdll():
    """Load ``tri_inverse`` shared library once (ctypes + signature)."""
    mtime = os.stat(os.path.join(_KERNELS_PTO, "tri_inverse.cpp")).st_mtime_ns
    lib_path = compile_tri_inverse(cpp_mtime_ns=mtime)
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,    # blockDim
        ctypes.c_void_p,    # stream
        ctypes.c_void_p,    # tensor_out  (fp32)
        ctypes.c_void_p,    # tensor_in   (fp16)
        ctypes.c_void_p,    # minus_identity  (fp16)
        ctypes.c_uint32,    # matrix_size
        ctypes.c_uint32,    # num_matrices
        ctypes.c_uint32,    # num_bsnd_heads  (bit 16 = is_lower flag)
        ctypes.c_void_p,    # cu_seqlens (optional int32)
    ]
    lib.call_kernel.restype = None
    return lib


def launch_tri_inverse_kernel(
    tensor_out: torch.Tensor,
    tensor_in: torch.Tensor,
    minus_identity: torch.Tensor,
    matrix_size: int,
    num_matrices: int,
    num_bsnd_heads: int,
    *,
    cu_seqlens: torch.Tensor | None,
    block_dim: int,
    stream_ptr,
    is_lower: bool,
) -> None:
    """Single ``lib.call_kernel`` dispatch — no stream lookup or dtype conversion.

    For micro-benchmarks, capture ``stream_ptr = torch.npu.current_stream()._as_parameter_``
    once outside the timed loop. If ``cu_seqlens`` is given, it must already be ``int32``.
    """
    lib = _tri_inverse_cdll()
    eff_bd = min(block_dim, num_matrices)
    heads_with_flag = (num_bsnd_heads & 0xFFFF) | (0x10000 if is_lower else 0)
    lib.call_kernel(
        eff_bd, stream_ptr,
        _vp(tensor_out), _vp(tensor_in), _vp(minus_identity),
        matrix_size, num_matrices, heads_with_flag,
        _vp(cu_seqlens) if cu_seqlens is not None else ctypes.c_void_p(),
    )
    _record_current_stream(tensor_out, tensor_in, minus_identity, cu_seqlens)


@lru_cache(maxsize=1)
def load_tri_inverse():
    """Compile (once) and return a callable wrapper for the triangular-inverse kernel."""

    def tri_inverse(
        tensor_out: torch.Tensor,
        tensor_in: torch.Tensor,
        minus_identity: torch.Tensor,
        matrix_size: int,
        num_matrices: int,
        num_bsnd_heads: int = 0,
        cu_seqlens: torch.Tensor | None = None,
        block_dim: int = BLOCK_DIM,
        stream_ptr=None,
        is_lower: bool = False,
    ) -> None:
        """Invert ``num_matrices`` triangular matrices of side ``matrix_size``.

        Args:
            tensor_out:       Output buffer (fp32), same shape as ``tensor_in``.
            tensor_in:        Input fp16 buffer of triangular matrices.
            minus_identity:   fp16 ``-I`` matrix of shape ``[matrix_size, matrix_size]``.
            matrix_size:      Side length; one of 16, 32, 64, 128.
            num_matrices:     Total number of matrices to invert.
            num_bsnd_heads:   ``N`` heads for BSND layout (0 = dense).
            cu_seqlens:       Optional ``int32`` varlen boundaries.
            block_dim:        Number of AI-Core blocks to use.
            stream_ptr:       NPU stream handle (defaults to current stream).
            is_lower:         If True, input is lower-triangular (kernel transposes on load/store).
        """
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32.")
        launch_tri_inverse_kernel(
            tensor_out,
            tensor_in,
            minus_identity,
            matrix_size,
            num_matrices,
            num_bsnd_heads,
            cu_seqlens=cu_seqlens,
            block_dim=block_dim,
            stream_ptr=stream_ptr,
            is_lower=is_lower,
        )

    return tri_inverse


def _solve_tril_num_matrices(cu32: torch.Tensor, chunk_size: int, num_heads: int) -> int:
    """Tile count (= ``total_chunks(...) * num_heads``) without host reads when lengths match."""
    nseq = cu32.numel() - 1
    if nseq <= 0:
        return 0
    d = torch.diff(cu32)
    if d.numel() > 0 and bool(torch.all(d == d[0]).item()):
        seg = int(d[0].item())
        nt_seg = (seg + chunk_size - 1) // chunk_size
        return nt_seg * int(nseq) * num_heads
    tc = total_chunks(int(nseq), int(cu32[-1].item()), chunk_size, cu32)
    return tc * num_heads


def solve_tril(
    A_fp16: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    num_heads: int,
    tri_inv_func=None,
    *,
    workspace_fp32: torch.Tensor | None = None,
    out_fp16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert the lower-triangular ``A`` tiles (solve_tril stage).

    Args:
        A_fp16:      ``[1, T, H, C]`` fp16 attention matrix from ``scaled_dot_kkt``.
        cu_seqlens:  ``int32`` varlen boundaries.
        chunk_size:  Chunk size C (matrix side length).
        num_heads:   Number of value heads H.
        tri_inv_func: Pre-loaded tri-inverse callable (auto-loaded if None).
        workspace_fp32: Optional fp32 workspace matching ``A_fp16.shape`` — reuse to skip
            allocations in benchmarks / fused callers.
        out_fp16:    Optional fp16 output buffer matching ``A_fp16.shape``.

    Returns:
        ``A_inv`` of the same shape and fp16 dtype.
    """
    if tri_inv_func is None:
        tri_inv_func = load_tri_inverse()

    if cu_seqlens is None:
        T = A_fp16.shape[1]
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=A_fp16.device)
    cu32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    dev = A_fp16.device
    dt, di = dev.type, dev.index if dev.index is not None else -1
    num_matrices = _solve_tril_num_matrices(cu32, chunk_size, num_heads)

    if workspace_fp32 is None:
        tensor_out = torch.zeros_like(A_fp16, dtype=torch.float32)
    else:
        if workspace_fp32.shape != A_fp16.shape or workspace_fp32.dtype != torch.float32:
            raise ValueError("workspace_fp32 must match A_fp16 shape and be float32.")
        tensor_out = workspace_fp32

    minus_identity = precomputed_minus_identity(dt, di, chunk_size)

    tri_inv_func(
        tensor_out, A_fp16, minus_identity,
        chunk_size, num_matrices, num_heads,
        cu_seqlens=cu32,
        block_dim=BLOCK_DIM,
        is_lower=True,
    )
    if out_fp16 is None:
        dest = torch.empty_like(A_fp16)
    else:
        if out_fp16.shape != A_fp16.shape:
            raise ValueError("out_fp16 shape must match A_fp16.")
        dest = out_fp16
    dest.copy_(tensor_out)
    return dest
