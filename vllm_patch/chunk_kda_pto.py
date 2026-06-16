"""PTO-backed ``chunk_kda`` replacement for vLLM-Ascend prefill.

Falls back transparently to the Triton implementation for:
  - Non-zero ``initial_state`` (decode or continuation with existing state)
  - Missing ``cu_seqlens`` (non-varlen path)
  - Non-NPU device

Execution mode: KDA megakernel (``VLLM_PTO_KDA_MEGAKERNEL=1``), all six
stages fused into a single NPU launch via ``megagdn_pto.kda_mega_kernel``.
"""

from __future__ import annotations

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

    import os
    if os.environ.get("VLLM_PTO_KDA_FORCE_TRITON") == "1":
        return _triton()

    if q.device.type != "npu":
        return _triton()
    if _needs_triton_fallback(initial_state, cu_seqlens):
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
