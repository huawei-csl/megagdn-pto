"""JIT launcher for the vector-only PTO KDN recurrent decode kernel."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from typing import Callable, NamedTuple

import torch

from megagdn_pto.compile import BLOCK_DIM, _KERNELS_PTO, compile_kdn_decode


def _vp(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p() if tensor is None else ctypes.c_void_p(tensor.data_ptr())


@lru_cache(maxsize=None)
def load_kdn_decode(k_dim: int = 128, v_dim: int = 128, v_tile: int = 128) -> ctypes.CDLL:
    cpp_path = os.path.join(_KERNELS_PTO, "kdn_decode.cpp")
    lib_path = compile_kdn_decode(
        k_dim=k_dim, v_dim=v_dim, v_tile=v_tile,
        cpp_mtime_ns=os.stat(cpp_path).st_mtime_ns,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 10
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
           ctypes.c_float, ctypes.c_int32]
    )
    lib.call_kernel.restype = None
    return lib


class _Prepared(NamedTuple):
    lib: ctypes.CDLL
    args: tuple
    out: torch.Tensor
    work_out: torch.Tensor
    state: torch.Tensor
    keepalive: tuple  # tensors whose data_ptr()s are baked into `args`


class PreparedDecode(NamedTuple):
    """A pre-marshalled decode launch.

    ``launch()`` performs exactly one kernel launch and nothing else -- no
    validation, no allocation, no dtype conversion, no ctypes marshalling.  Use
    it to time the kernel itself; use ``run_kdn_decode`` for real work.
    """

    launch: Callable[[], None]
    out: torch.Tensor
    state: torch.Tensor


def _prepare(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, initial_state: torch.Tensor | None = None, *,
    state_indices: torch.Tensor | None = None, scale: float | None = None,
    out: torch.Tensor | None = None, stream=None, v_tile: int = 128,
    block_dim: int | None = None, cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm: bool = False, state_out: torch.Tensor | None = None,
    zero_out: bool | None = None,
) -> "_Prepared":
    """Validate inputs and marshal one kernel launch; see ``run_kdn_decode``."""
    if q.ndim != 4:
        raise ValueError("q must be [B,T,H,K]")
    b, t, h, kdim = q.shape
    if tuple(k.shape) != tuple(q.shape) or tuple(g.shape) != tuple(q.shape):
        raise ValueError("q, k, and g must have identical [B,T,H,K] shapes")
    if v.ndim != 4 or tuple(v.shape[:3]) != (b, t, h):
        raise ValueError("v must be [B,T,H,V]")
    vdim = v.shape[-1]
    if tuple(beta.shape) != (b, t, h):
        raise ValueError("beta must be [B,T,H]")
    if t <= 0 or kdim <= 0 or vdim <= 0 or kdim % 8 or v_tile % 8:
        raise ValueError("T/K/V and v_tile must be positive; K and v_tile must be multiples of 8")
    tensors = (q, k, v, g, beta)
    if any(x.dtype not in (torch.bfloat16, torch.float16) for x in tensors):
        raise TypeError("q, k, v, g, and beta must be bf16 or fp16")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("q, k, v, g, and beta must be contiguous")
    if any(x.device != q.device for x in tensors[1:]):
        raise ValueError("all inputs must share a device")
    if cu_seqlens is None:
        n_seq = b
    else:
        if b != 1:
            raise ValueError("cu_seqlens requires B == 1 (flatten sequences onto the token axis)")
        if (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
                or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous()
                or cu_seqlens.device != q.device):
            raise ValueError("cu_seqlens must be contiguous int32 [N+1] on q's device")
        n_seq = cu_seqlens.numel() - 1
    if initial_state is None:
        initial_state = torch.zeros((n_seq, h, vdim, kdim), dtype=torch.float32, device=q.device)
    if (initial_state.ndim != 4 or tuple(initial_state.shape[1:]) != (h, vdim, kdim)
            or initial_state.dtype != torch.float32 or not initial_state.is_contiguous()):
        raise ValueError("initial_state must be contiguous fp32 [slots,H,V,K]")
    if initial_state.device != q.device:
        raise ValueError("initial_state must share a device with q")
    if state_out is not None:
        if state_indices is not None:
            raise ValueError(
                "state_out cannot be combined with state_indices: a gathered run only "
                "writes the selected slots, leaving the rest of state_out undefined"
            )
        if (tuple(state_out.shape) != tuple(initial_state.shape)
                or state_out.dtype != torch.float32
                or not state_out.is_contiguous()
                or state_out.device != q.device):
            raise ValueError(
                "state_out must match initial_state's shape/dtype/device and be contiguous"
            )
    if state_indices is None:
        if initial_state.shape[0] < n_seq:
            raise ValueError("initial_state needs one slot per sequence without state_indices")
    else:
        if (tuple(state_indices.shape) != (n_seq,) or state_indices.dtype != torch.int32
                or not state_indices.is_contiguous() or state_indices.device != q.device):
            raise ValueError("state_indices must be contiguous int32 [N] on q's device")
    # The kernel writes every output row it is responsible for, so pre-zeroing
    # is only load-bearing when rows can be *skipped*: a negative/out-of-range
    # state_indices slot, or an empty cu_seqlens span.  Without either, every
    # (sequence, head, v-tile) is visited and the zero-fill is a wasted pass.
    if zero_out is None:
        zero_out = state_indices is not None or cu_seqlens is not None
    if out is None:
        out = torch.zeros_like(v) if zero_out else torch.empty_like(v)
    elif (tuple(out.shape) != tuple(v.shape) or out.dtype != v.dtype
          or not out.is_contiguous() or out.device != q.device):
        raise ValueError("out must be contiguous with v's shape/dtype/device")
    elif zero_out:
        out.zero_()
    if scale is None:
        scale = kdim ** -0.5
    tasks = n_seq * h * ((vdim + v_tile - 1) // v_tile)
    if block_dim is None:
        block_dim = min(BLOCK_DIM, max(1, (tasks + 1) // 2))
    # CANN's C220 vector compiler does not implement bf16->fp32 TCVT.  Keep
    # the kernel's wire format fp16 and perform one explicit model-dtype cast
    # at the launcher boundary.
    work_inputs = tuple(x if x.dtype == torch.float16 else x.to(torch.float16)
                        for x in tensors)
    work_out = out if out.dtype == torch.float16 else torch.empty_like(out, dtype=torch.float16)
    if work_out is not out and zero_out:
        work_out.zero_()
    lib = load_kdn_decode(kdim, vdim, v_tile)
    stream = torch.npu.current_stream(q.device)._as_parameter_
    args = (
        ctypes.c_uint32(block_dim), stream, *(_vp(x) for x in work_inputs),
        _vp(initial_state), _vp(state_out),
        _vp(work_out), _vp(state_indices), _vp(cu_seqlens),
        ctypes.c_int64(n_seq), ctypes.c_int64(t), ctypes.c_int32(h),
        ctypes.c_int32(initial_state.shape[0]), ctypes.c_float(scale),
        ctypes.c_int32(1 if use_qk_l2norm else 0),
    )
    return _Prepared(
        lib=lib, args=args, out=out, work_out=work_out,
        state=initial_state if state_out is None else state_out,
        keepalive=(*work_inputs, initial_state, state_out, work_out,
                   state_indices, cu_seqlens),
    )


def run_kdn_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, initial_state: torch.Tensor | None = None, **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the default ``fused_recurrent_kda`` recurrence.

    q/k/g are [B,T,H,K], v is [B,T,H,V], beta is [B,T,H], in bf16/fp16.
    State is fp32 V-first [slots,H,V,K] and is updated in place.

    ``cu_seqlens`` is an int32 ``[N+1]`` prefix sum of sequence lengths; it
    requires ``B == 1`` and turns the token axis into ``N`` variable-length
    sequences, one state slot each.  ``use_qk_l2norm`` normalizes q and k over
    K inside the kernel, matching the fused sglang reference.

    ``state_out`` makes the recurrence out of place: the kernel reads
    ``initial_state`` and writes the final state to ``state_out``, which may be
    uninitialized.  It is rejected together with ``state_indices`` -- a gathered
    run only visits the selected slots, so every other slot of ``state_out``
    would be garbage, and pre-copying the pool to fix that costs far more
    traffic than the decode itself.  Paged/gathered callers want the default
    in-place update.

    ``zero_out`` forces (``True``) or suppresses (``False``) the pre-zeroing of
    ``out``.  The default auto-detects: rows can only be left unwritten when
    ``state_indices`` or ``cu_seqlens`` is given, so the zero-fill is skipped
    otherwise.  Passing ``False`` alongside either of those leaves the skipped
    rows holding whatever ``out`` already contained.

    Note that bf16 inputs cost five device-side casts to the kernel's fp16 wire
    format, and a bf16 ``out`` costs a further staging buffer plus a copy back.
    Pass fp16 tensors and a preallocated fp16 ``out`` to avoid all of it -- or
    use ``prepare_kdn_decode`` to hoist the whole marshalling step out of a hot
    loop entirely.
    """
    p = _prepare(q, k, v, g, beta, initial_state, **kwargs)
    p.lib.call_kernel(*p.args)
    if p.work_out is not p.out:
        p.out.copy_(p.work_out)
    return p.out, p.state


def prepare_kdn_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, initial_state: torch.Tensor | None = None, **kwargs,
) -> PreparedDecode:
    """Do all of ``run_kdn_decode``'s host-side work once, up front.

    Returns a ``PreparedDecode`` whose ``launch()`` is a single kernel launch,
    so timing it measures the kernel rather than the launcher.  Requires inputs
    that need no conversion: fp16 q/k/v/g/beta, and either no ``out`` or an fp16
    one.  The launch is pinned to the stream current at prepare time and to
    these exact buffers -- mutate their contents freely, but do not free or
    replace them.
    """
    if any(x.dtype != torch.float16 for x in (q, k, v, g, beta)):
        raise ValueError(
            "prepare_kdn_decode requires fp16 q/k/v/g/beta: converting bf16 here "
            "would snapshot the inputs once, so later writes to the originals "
            "would be silently ignored by launch()"
        )
    p = _prepare(q, k, v, g, beta, initial_state, **kwargs)
    if p.work_out is not p.out:
        raise ValueError(
            "prepare_kdn_decode requires an fp16 `out` (or none): a bf16 output "
            "forces a staging buffer and a copy back, which cannot be hoisted "
            "out of the launch"
        )
    lib, args = p.lib, p.args

    # `args` holds raw data_ptr()s, so the default arg pins the owning tensors
    # for as long as the closure lives.
    def launch(_keepalive=p.keepalive) -> None:
        lib.call_kernel(*args)

    return PreparedDecode(launch=launch, out=p.out, state=p.state)
