"""Launcher for the state-sized KDN GM/UB copy bandwidth ceiling."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from typing import Callable

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_kdn_memcpy_bound


def _vp(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


@lru_cache(maxsize=None)
def load_kdn_memcpy_bound(k_dim: int = 128, v_dim: int = 128, v_tile: int = 32) -> ctypes.CDLL:
    cpp_path = os.path.join(_KERNELS_PTO, "kdn_memcpy_bound.cpp")
    lib_path = compile_kdn_memcpy_bound(
        k_dim=k_dim, v_dim=v_dim, v_tile=v_tile,
        cpp_mtime_ns=os.stat(cpp_path).st_mtime_ns,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int32,
    ]
    lib.call_kernel.restype = None
    return lib


def _prepare(
    state: torch.Tensor, *, v_tile: int = 32, block_dim: int | None = None,
) -> tuple[ctypes.CDLL, tuple]:
    """Validate and marshal one copy launch; see ``run_kdn_memcpy_bound``."""
    if state.ndim != 4 or state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous fp32 [B,H,V,K]")
    if state.device.type != "npu":
        raise ValueError("state must be on an NPU")
    batch, heads, v_dim, k_dim = state.shape
    if k_dim <= 0 or v_dim <= 0 or k_dim % 8 or v_dim % v_tile:
        raise ValueError("K must be a positive multiple of 8 and v_tile must divide V")
    tasks = batch * heads * (v_dim // v_tile)
    if block_dim is None:
        block_dim = min(BLOCK_DIM, max(1, (tasks + 1) // 2))
    lib = load_kdn_memcpy_bound(k_dim, v_dim, v_tile)
    stream = torch.npu.current_stream(state.device)._as_parameter_
    args = (
        ctypes.c_uint32(block_dim), stream, _vp(state),
        ctypes.c_int64(batch), ctypes.c_int32(heads),
    )
    return lib, args


def run_kdn_memcpy_bound(
    state: torch.Tensor, *, v_tile: int = 32, block_dim: int | None = None,
) -> None:
    """Round-trip every fp32 state element through UB in-place.

    ``state`` is contiguous ``[B, H, V, K]`` fp32.  The tile mapping and block
    sizing match ``run_kdn_decode``; this function intentionally performs no
    vector arithmetic, providing a state-traffic upper bound.
    """
    lib, args = _prepare(state, v_tile=v_tile, block_dim=block_dim)
    lib.call_kernel(*args)


def prepare_kdn_memcpy_bound(
    state: torch.Tensor, *, v_tile: int = 32, block_dim: int | None = None,
) -> Callable[[], None]:
    """Validate and marshal once; return a callable that is one kernel launch.

    The ceiling this kernel measures is only meaningful if it is timed the same
    way as the kernel it bounds -- otherwise host-side dispatch shows up as
    "bandwidth" and the ceiling can land *below* the thing it is supposed to
    bound.  Pinned to the stream current at prepare time and to this exact
    buffer.
    """
    lib, args = _prepare(state, v_tile=v_tile, block_dim=block_dim)

    # `args` holds a raw data_ptr(), so the default arg pins the owning tensor.
    def launch(_keepalive=state) -> None:
        lib.call_kernel(*args)

    return launch
