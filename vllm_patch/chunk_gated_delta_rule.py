"""PTO-backed ``chunk_gated_delta_rule`` replacement for vLLM-Ascend prefill.

Falls back transparently to the Triton implementation for:
  - Non-zero ``initial_state`` (decode with KV cache)
  - Missing ``cu_seqlens`` (non-varlen path)
  - Multi-device pipeline-parallel (PCP) groups
  - Mismatched Q/K/V head dimensions
  - Non-NPU device

Two execution modes (controlled by ``VLLM_PTO_MEGAKERNEL`` env var):
  - **Staged** (default): six separate kernel launches via ``megagdn_pto.kernel_libs``.
  - **Megakernel** (``VLLM_PTO_MEGAKERNEL=1``): single fused launch via
    ``megagdn_pto.mega_kernel``.

GQA is supported: if ``v.shape[2] > q.shape[2]``, the GQA path is taken.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from einops import rearrange

from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kernel_libs import (
    chunk_gdn_causal_masks,
    run_chunk_cumsum,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
    transpose_beta,
    transpose_gates,
)
from megagdn_pto.mega_kernel import run_mega_kernel, run_mega_kernel_bf16

C_PTO = 128


@lru_cache(maxsize=1)
def _pto_tri_inverse_cached():
    return load_tri_inverse()


# ---------------------------------------------------------------------------
# Fallback detection helpers
# ---------------------------------------------------------------------------

def _needs_triton_fallback(
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    if initial_state is not None and torch.any(initial_state != 0):
        return True
    return cu_seqlens is None


def _is_gqa(q: torch.Tensor, v: torch.Tensor) -> bool:
    return v.shape[2] != q.shape[2]


def _head_dims_compatible(q: torch.Tensor, v: torch.Tensor) -> bool:
    return q.shape[3] == v.shape[3]


def _megakernel_enabled() -> bool:
    return os.environ.get("VLLM_PTO_MEGAKERNEL", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Staged pipeline (6 kernel launches)
# ---------------------------------------------------------------------------

def _staged_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu32: torch.Tensor,
    scale: float,
    *,
    Hg: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = q.device
    T = q.shape[1]
    D = q.shape[3]
    N_seq = int(cu32.numel()) - 1
    stream = torch.npu.current_stream()._as_parameter_

    dt, di = dev.type, dev.index if dev.index is not None else -1
    msk_lower, msk_full = chunk_gdn_causal_masks(dt, di, C_PTO)

    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    run_chunk_cumsum(
        g.float(), g_sum,
        stream=stream, chunk_size=C_PTO,
        cu_seqlens=cu32, batch_size_override=N_seq,
    )
    g_t = transpose_gates(g_sum)
    beta_t = transpose_beta(beta)
    torch.npu.synchronize()

    A = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    with torch.autograd.profiler.record_function("PTO_kkt"):
        run_scaled_dot_kkt(k, beta, g_sum, msk_lower, A,
                           stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                           cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    with torch.autograd.profiler.record_function("PTO_solve_tril"):
        A_inv = solve_tril(A, cu32, C_PTO, H, _pto_tri_inverse_cached())
    torch.npu.synchronize()

    w = torch.empty_like(v)
    u = torch.empty_like(v)
    with torch.autograd.profiler.record_function("PTO_wy_fast"):
        run_wy_fast(k, v, beta, g_sum, A_inv, w, u,
                    stream=stream, g_t=g_t, beta_t=beta_t, chunk_size=C_PTO,
                    cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    tc_n = total_chunks(N_seq, T, C_PTO, cu32)
    s = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    with torch.autograd.profiler.record_function("PTO_chunk_h"):
        run_chunk_h(k, w, u, g_sum, s, v_new, fs,
                    stream=stream, g_t=g_t, chunk_size=C_PTO,
                    cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    o = torch.empty_like(v)
    with torch.autograd.profiler.record_function("PTO_chunk_o"):
        run_chunk_o(q, k, v_new, s, g_sum, msk_full, o,
                    stream=stream, g_t=g_t, chunk_size=C_PTO,
                    cu_seqlens=cu32, batch_size_override=N_seq, key_heads=Hg)
    torch.npu.synchronize()

    return (o * scale).to(q.dtype), fs.view(N_seq, H, D, D).to(q.dtype)


# ---------------------------------------------------------------------------
# Megakernel pipeline (1 fused launch)
# ---------------------------------------------------------------------------

def _mega_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu32: torch.Tensor,
    scale: float,
    *,
    Hg: int,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    stream = torch.npu.current_stream()._as_parameter_
    with torch.autograd.profiler.record_function("PTO_mega_kernel"):
        if q.dtype == torch.bfloat16:
            # BF16 path: cast q,k,v,beta→fp16 INSIDE the kernel.
            # g must be float32; caller passes g.float().
            result = run_mega_kernel_bf16(
                q, k, v, g, beta, cu32,
                stream=stream, chunk_size=C_PTO, scale=scale,
                key_heads=Hg, return_final_state=output_final_state,
            )
        else:
            # FP16 path: existing, unchanged.
            result = run_mega_kernel(
                q, k, v, g.float(), beta, cu32,
                stream=stream, chunk_size=C_PTO, scale=scale,
                key_heads=Hg, return_final_state=output_final_state,
            )
    if output_final_state:
        o, fs = result
        return o.to(q.dtype), fs.to(q.dtype)
    return result.to(q.dtype), None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@torch.compiler.disable
def chunk_gated_delta_rule_pto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    _triton_impl,
):
    """PTO drop-in for ``vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule``."""
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "Use bfloat16 or float16, not float32."
    assert beta.ndim == 3, "beta must be [B, T, H] (head_first=False)."

    if head_first:
        q, k, v, beta, g = (rearrange(x, "b h t ... -> b t h ...") for x in (q, k, v, beta, g))

    if scale is None:
        scale = float(k.shape[-1] ** -0.5)

    if use_qk_l2norm_in_kernel:
        from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
        q, k = l2norm_fwd(q), l2norm_fwd(k)

    def _triton(*args, **kw):
        return _triton_impl(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta, head_first=False, use_qk_l2norm_in_kernel=False,
        )

    # --- Triton fallback conditions ---
    if q.device.type != "npu":
        return _triton()
    if _needs_triton_fallback(initial_state, cu_seqlens):
        return _triton()
    if not _head_dims_compatible(q, v):
        return _triton()
    if _is_gqa(q, v) and v.shape[2] % q.shape[2] != 0:
        return _triton()
    try:
        from vllm.distributed import get_pcp_group
        if get_pcp_group().world_size > 1:
            return _triton()
    except Exception:
        pass

    # --- PTO forward ---
    Hg = q.shape[2]
    H = v.shape[2]
    cu32 = cu_seqlens.to(torch.int32).contiguous()

    if _megakernel_enabled() and q.dtype == torch.bfloat16:
        # BF16 megakernel path: pass original bf16 tensors directly.
        # The kernel handles bf16→fp16 casting internally, eliminating 4-5
        # separate Python-level torch.to() calls.
        # g is still converted to float32 here (small tensor, torch is efficient).
        g_w = g.float()
        o, final_state = _mega_forward(
            q, k, v, g_w, beta, cu32, scale, Hg=Hg, output_final_state=output_final_state,
        )
    elif _megakernel_enabled():
        # FP16 megakernel path: existing behaviour, unchanged.
        q_w = q.to(torch.float16)
        k_w = k.to(torch.float16)
        v_w = v.to(torch.float16)
        beta_w = beta.to(torch.float16)
        g_w = g.float()
        o, final_state = _mega_forward(
            q_w, k_w, v_w, g_w, beta_w, cu32, scale, Hg=Hg, output_final_state=output_final_state,
        )
    else:
        # Staged path: always casts to fp16 (individual stage kernels only speak fp16).
        q_w = q.to(torch.float16)
        k_w = k.to(torch.float16)
        v_w = v.to(torch.float16)
        beta_w = beta.to(torch.float16)
        g_w = g.float()
        o, final_state = _staged_forward(q_w, k_w, v_w, g_w, beta_w, cu32, scale, Hg=Hg, H=H)

    if not output_final_state:
        final_state = None

    o = o.to(q.dtype)
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, final_state


def bind_triton(_triton_impl):
    """Return a callable matching the vLLM public API with the Triton fallback bound."""

    def _bound(
        q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
        cu_seqlens=None, prebuilt_meta=None, head_first=False, use_qk_l2norm_in_kernel=False,
    ):
        return chunk_gated_delta_rule_pto(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, prebuilt_meta=prebuilt_meta,
            head_first=head_first, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            _triton_impl=_triton_impl,
        )

    _bound.__name__ = "chunk_gated_delta_rule"
    _bound._vllm_pto_chunk_wrapper_installed = True
    return _bound
