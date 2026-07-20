"""Correctness tests for the vector-only PTO KDN decode kernel."""

import torch
import torch.nn.functional as F
from einops import rearrange

from megagdn_pto.kdn_decode import run_kdn_decode


DEVICE = torch.device("npu:0")
OUTPUT_ATOL = 1e-2
OUTPUT_RTOL = 1e-2
STATE_ATOL = 1e-5
STATE_RTOL = 1e-5


# torch reference from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S



def _inputs(batch: int, tokens: int, heads: int, dim: int = 128):
    torch.manual_seed(2026)
    q = F.normalize(torch.randn(batch, tokens, heads, dim), p=2, dim=-1)
    k = F.normalize(torch.randn(batch, tokens, heads, dim), p=2, dim=-1)
    return (
        q.to(torch.bfloat16).to(DEVICE),
        k.to(torch.bfloat16).to(DEVICE),
        torch.randn(batch, tokens, heads, dim, dtype=torch.bfloat16, device=DEVICE),
        torch.randn(batch, tokens, heads, dim, dtype=torch.bfloat16, device=DEVICE),
        torch.randn(batch, tokens, heads, dtype=torch.bfloat16, device=DEVICE),
    )


def _reference(q, k, v, g, beta, state):
    out, final_state = naive_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state.transpose(-1, -2),
        output_final_state=True,
    )
    return out, final_state.transpose(-1, -2).contiguous()


def _assert_matches_reference(batch: int, tokens: int, heads: int) -> None:
    q, k, v, g, beta = _inputs(batch, tokens, heads)
    initial_state = torch.randn(batch, heads, 128, 128, device=DEVICE)

    expected_out, expected_state = _reference(q, k, v, g, beta, initial_state.clone())
    actual_out, actual_state = run_kdn_decode(q, k, v, g, beta, initial_state.clone())
    torch.npu.synchronize()


    print(f"Errors in state: {(expected_state-actual_state).abs().max().item()}")
    print(f"Errors in out: {(expected_out-actual_out).abs().max().item()}")
    torch.testing.assert_close(
        actual_out.float().cpu(), expected_out.float().cpu(), atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(actual_state.cpu(), expected_state.cpu(), atol=STATE_ATOL, rtol=STATE_RTOL)


def test_kdn_decode_state_gather_and_skip() -> None:
    """Gather a non-contiguous state slot and leave a negative slot untouched."""
    q, k, v, g, beta = _inputs(batch=2, tokens=1, heads=2)
    state = torch.randn(3, 2, 128, 128, device=DEVICE)
    state_before = state.clone()
    indices = torch.tensor([2, -1], dtype=torch.int32, device=DEVICE)

    actual_out, actual_state = run_kdn_decode(
        q, k, v, g, beta, state, state_indices=indices
    )
    expected_out, expected_state = _reference(
        q[:1], k[:1], v[:1], g[:1], beta[:1], state_before[2:3].clone()
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_out[:1].float().cpu(), expected_out.float().cpu(), atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(
        actual_state[2].cpu(), expected_state[0].cpu(), atol=STATE_ATOL, rtol=STATE_RTOL
    )
    assert torch.count_nonzero(actual_out[1]).item() == 0
    torch.testing.assert_close(actual_state[0].cpu(), state_before[0].cpu())
    torch.testing.assert_close(actual_state[1].cpu(), state_before[1].cpu())


def test_kdn_decode() -> None:
    """Run the complete lightweight correctness check."""
    _assert_matches_reference(5, 1, 1)
    _assert_matches_reference(22, 1, 32)
    test_kdn_decode_state_gather_and_skip()


if __name__ == "__main__":
    test_kdn_decode()
    print("KDN decode checks passed")
