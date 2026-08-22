"""PTO-backed ``chunk_kda`` replacement for vLLM-Ascend prefill.

Falls back transparently to the Triton implementation for:
  - Non-zero ``initial_state`` (decode or continuation with existing state)
  - Missing ``cu_seqlens`` (non-varlen path)
  - Non-NPU device

Execution mode: KDA megakernel (``VLLM_PTO_KDA_MEGAKERNEL=1``), all six
stages fused into a single NPU launch via ``megagdn_pto.kda_mega_kernel``.

vLLM 0.23 moved KimiLinear onto ``chunk_kda_with_fused_gate``, which takes the
raw gate projection plus ``A_log``/``g_bias`` and fuses the gate computation
into the cumsum. ``bind_triton_fused_gate`` covers that entry point: it runs
vLLM's own ``fused_kda_gate`` for the elementwise part and hands the resulting
gate to the PTO megakernel, which does its own cumsum.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

C_PTO = 128


def _needs_triton_fallback(
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    if initial_state is not None and torch.any(initial_state != 0):
        return True
    return cu_seqlens is None


def _use_triton(
    q: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Whether this call must go to Triton rather than the PTO megakernel."""
    if os.environ.get("VLLM_PTO_KDA_FORCE_TRITON") == "1":
        return True
    if q.device.type != "npu":
        return True
    return _needs_triton_fallback(initial_state, cu_seqlens)


@torch.compiler.disable
def chunk_kda_pto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    *,
    _triton_impl,
    **kwargs,
):
    """PTO drop-in for ``vllm.model_executor.layers.fla.ops.kda.chunk_kda``."""

    def _triton():
        return _triton_impl(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

    if _use_triton(q, initial_state, cu_seqlens):
        return _triton()

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

    if scale is None:
        scale = float(q.shape[-1] ** -0.5)

    from megagdn_pto.kda_mega_kernel import run_mega_kernel_kda

    stream = torch.npu.current_stream()._as_parameter_
    cu32 = cu_seqlens.to(torch.int32).contiguous()
    N_seq = int(cu32.numel()) - 1

    q_w = (q * scale).to(torch.float16)
    k_w = k.to(torch.float16)
    v_w = v.to(torch.float16)
    g_w = g.to(torch.float16)
    beta_w = beta.to(torch.float16)

    o, final_state = run_mega_kernel_kda(
        q_w, k_w, v_w, g_w, beta_w, cu32,
        stream=stream,
        chunk_size=C_PTO,
        batch_size_override=N_seq,
        return_final_state=True,
    )

    o = o.to(q.dtype)
    if output_final_state:
        # vllm expects [N_seq, HV, K, V]; _extract_final_states already returns
        # that. Match the cache dtype (fp32 recurrent state) — aclnnIndexPutImpl
        # cannot cast on write-back.
        state_dtype = initial_state.dtype if initial_state is not None else torch.float32
        return o, final_state.to(state_dtype).contiguous()
    return o, None


def bind_triton(_triton_impl):
    """Return a callable matching the vLLM public API with the Triton fallback bound."""

    def _bound(
        q, k, v, g, beta,
        scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs,
    ):
        return chunk_kda_pto(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            _triton_impl=_triton_impl,
            **kwargs,
        )

    _bound.__name__ = "chunk_kda"
    _bound._vllm_pto_kda_wrapper_installed = True
    return _bound


# ---------------------------------------------------------------------------
# vLLM 0.23+ fused-gate entry point (KimiLinear)
# ---------------------------------------------------------------------------

def _kda_gate(raw_g: torch.Tensor, A_log: torch.Tensor, g_bias: torch.Tensor | None) -> torch.Tensor:
    """Elementwise part of ``chunk_kda_with_fused_gate``: raw projection -> gate.

    ``raw_g`` is ``[..., H, D]``; ``fused_kda_gate`` wants ``[..., H*D]`` and
    returns ``[..., H, D]`` again — the same pre-cumsum gate that plain
    ``chunk_kda`` takes as ``g``.
    """
    from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate

    n_heads, head_dim = raw_g.shape[-2], raw_g.shape[-1]
    flat = raw_g.reshape(*raw_g.shape[:-2], n_heads * head_dim)
    return fused_kda_gate(flat, A_log, head_dim, g_bias=g_bias)


def bind_triton_fused_gate(_triton_impl):
    """Return a ``chunk_kda_with_fused_gate`` drop-in backed by the PTO megakernel."""

    def _bound(
        q, k, v, raw_g, beta, A_log, g_bias,
        scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs,
    ):
        def _fallback(*_args, **_kw):
            # ``chunk_kda_pto`` hands back the tensors it was given; ignore them
            # and let upstream redo the gate from ``raw_g`` itself.
            return _triton_impl(
                q, k, v, raw_g, beta, A_log, g_bias,
                scale=scale, initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens, **kwargs,
            )

        # Decide before computing the gate: ``fused_kda_gate`` is itself a Triton
        # kernel launch, wasted work (and a hard failure off-NPU) on a call that
        # is going to fall back anyway.
        if _use_triton(q, initial_state, cu_seqlens):
            return _fallback()

        return chunk_kda_pto(
            q, k, v, _kda_gate(raw_g, A_log, g_bias), beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            _triton_impl=_fallback,
        )

    _bound.__name__ = "chunk_kda_with_fused_gate"
    _bound._vllm_pto_kda_wrapper_installed = True
    return _bound
