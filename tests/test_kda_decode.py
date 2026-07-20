"""Correctness tests for the vector-only PTO KDN decode kernel."""

import argparse

import torch
import torch.nn.functional as F
from einops import rearrange

from megagdn_pto.kdn_decode import load_kdn_decode, run_kdn_decode


DEVICE = torch.device("npu:0")
OUTPUT_ATOL = 1e-5
OUTPUT_RTOL = 1e-2
STATE_ATOL = 1e-5
STATE_RTOL = 1e-5


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=0, atol=0
    )


def test_kdn_decode_exact_sparse_recurrence() -> None:
    """Check two hand-computed one-hot updates with mixed sparse values."""
    shape = (1, 2, 1, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    beta = torch.zeros(shape[:-1], dtype=torch.bfloat16, device=DEVICE)

    # Token 0 reads columns 0 and 1, replaces column 0, and forgets column 2.
    q[0, 0, 0, 0] = 1
    q[0, 0, 0, 1] = 1
    k[0, 0, 0, 0] = 1
    v[0, 0, 0, 0] = 3
    v[0, 0, 0, 1] = -2
    v[0, 0, 0, 2] = 1
    g[0, 0, 0, 2] = -torch.inf
    beta[0, 0, 0] = 1

    # Token 1 moves column 1 halfway toward v, then computes column 0 - column 1.
    q[0, 1, 0, 0] = 1
    q[0, 1, 0, 1] = -1
    k[0, 1, 0, 1] = 1
    v[0, 1, 0, 0] = -1
    v[0, 1, 0, 1] = 4
    v[0, 1, 0, 2] = 2
    beta[0, 1, 0] = 0.5

    initial_state = torch.zeros((1, 1, 128, 128), device=DEVICE)
    initial_state[0, 0, 0, 0] = 1
    initial_state[0, 0, 1, 1] = 2
    initial_state[0, 0, 2, 0] = -1
    initial_state[0, 0, 0, 2] = 4
    initial_state[0, 0, 1, 2] = -2
    initial_state[0, 0, 2, 2] = 1

    expected_out = torch.zeros_like(v)
    expected_out[0, 0, 0, :3] = torch.tensor(
        [3, 0, 1], dtype=v.dtype, device=DEVICE
    )
    expected_out[0, 1, 0, :3] = torch.tensor(
        [3.5, -5, 0], dtype=v.dtype, device=DEVICE
    )
    expected_state = torch.zeros_like(initial_state)
    expected_state[0, 0, 0, 0] = 3
    expected_state[0, 0, 1, 0] = -2
    expected_state[0, 0, 2, 0] = 1
    expected_state[0, 0, 0, 1] = -0.5
    expected_state[0, 0, 1, 1] = 3
    expected_state[0, 0, 2, 1] = 1

    actual_out, actual_state = run_kdn_decode(
        q, k, v, g, beta, initial_state, scale=1.0
    )
    torch.npu.synchronize()

    _assert_exact(actual_out, expected_out)
    _assert_exact(actual_state, expected_state)


def test_kdn_decode_exact_mixed_batch_heads() -> None:
    """Check independent sparse behavior across batches and heads."""
    shape = (2, 1, 2, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    beta = torch.zeros(shape[:-1], dtype=torch.bfloat16, device=DEVICE)

    # Batch 0, head 0: beta=1/2 writes 1 and the matching query reads 1.
    q[0, 0, 0, 3] = 1
    k[0, 0, 0, 3] = 1
    v[0, 0, 0, 5] = 2
    beta[0, 0, 0] = 0.5

    # Batch 0, head 1: beta=1/4 writes -1 and q=2 reads -2.
    q[0, 0, 1, 7] = 2
    k[0, 0, 1, 7] = 1
    v[0, 0, 1, 9] = -4
    beta[0, 0, 1] = 0.25

    # Batch 1, head 0: beta=0 preserves the existing state despite nonzero k/v.
    q[1, 0, 0, 13] = 1
    k[1, 0, 0, 14] = 1
    v[1, 0, 0, 15] = 2

    # Batch 1, head 1: orthogonal q/k gives zero output but still updates state.
    q[1, 0, 1, 21] = 1
    k[1, 0, 1, 20] = 1
    v[1, 0, 1, 22] = 5
    beta[1, 0, 1] = 1

    initial_state = torch.zeros((2, 2, 128, 128), device=DEVICE)
    initial_state[1, 0, 11, 13] = 3
    expected_state = torch.zeros_like(initial_state)
    expected_state[0, 0, 5, 3] = 1
    expected_state[0, 1, 9, 7] = -1
    expected_state[1, 0, 11, 13] = 3
    expected_state[1, 1, 22, 20] = 5
    expected_out = torch.zeros_like(v)
    expected_out[0, 0, 0, 5] = 1
    expected_out[0, 0, 1, 9] = -2
    expected_out[1, 0, 0, 11] = 3

    actual_out, actual_state = run_kdn_decode(
        q, k, v, g, beta, initial_state, scale=1.0
    )
    torch.npu.synchronize()

    _assert_exact(actual_out, expected_out)
    _assert_exact(actual_state, expected_state)


def test_kdn_decode_exact_selective_decay() -> None:
    """A mixed zero/-inf gate forgets only the selected key column."""
    shape = (1, 1, 1, 128)
    q = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros_like(q)
    beta = torch.ones(shape[:-1], dtype=torch.bfloat16, device=DEVICE)
    q[0, 0, 0, 0] = 1
    q[0, 0, 0, 1] = 1
    q[0, 0, 0, 2] = -1
    v[0, 0, 0, 0] = 7
    g[0, 0, 0, 1] = -torch.inf

    initial_state = torch.zeros((1, 1, 128, 128), device=DEVICE)
    initial_state[0, 0, 0, :3] = torch.tensor(
        [1, 2, 4], dtype=initial_state.dtype, device=DEVICE
    )
    initial_state[0, 0, 1, :3] = torch.tensor(
        [-1, 3, 5], dtype=initial_state.dtype, device=DEVICE
    )
    expected_state = initial_state.clone()
    expected_state[:, :, :, 1] = 0
    expected_out = torch.zeros_like(v)
    expected_out[0, 0, 0, 0] = -3
    expected_out[0, 0, 0, 1] = -6

    actual_out, actual_state = run_kdn_decode(
        q, k, v, g, beta, initial_state, scale=1.0
    )
    torch.npu.synchronize()

    _assert_exact(actual_out, expected_out)
    _assert_exact(actual_state, expected_state)


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



def _inputs(
    batch: int,
    tokens: int,
    heads: int,
    dim: int = 128,
    *,
    seed: int = 2026,
    model_like: bool = False,
):
    torch.manual_seed(seed)
    shape = (batch, tokens, heads, dim)
    q = F.normalize(torch.randn(shape), p=2, dim=-1)
    k = F.normalize(torch.randn(shape), p=2, dim=-1)
    v = torch.randn(shape)
    g = torch.randn(shape)
    beta = torch.randn(shape[:-1])
    if model_like:
        g = F.logsigmoid(g)
        beta = torch.sigmoid(beta)
    return tuple(
        tensor.to(torch.bfloat16).to(DEVICE)
        for tensor in (q, k, v, g, beta)
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


def test_kdn_decode_multiple_tokens() -> None:
    """Compare dense model-like inputs for several values of T greater than one."""
    batch = 2
    heads = 4
    dim = 128

    for tokens in (2, 5, 13):
        q, k, v, g, beta = _inputs(
            batch,
            tokens,
            heads,
            dim,
            seed=2026 + tokens,
            model_like=True,
        )
        initial_state = (
            0.1 * torch.randn(batch, heads, dim, dim)
        ).to(DEVICE)

        expected_out, expected_state = _reference(
            q, k, v, g, beta, initial_state.clone()
        )
        actual_out, actual_state = run_kdn_decode(
            q, k, v, g, beta, initial_state.clone()
        )
        torch.npu.synchronize()

        torch.testing.assert_close(
            actual_out.float().cpu(),
            expected_out.float().cpu(),
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
        torch.testing.assert_close(
            actual_state.cpu(),
            expected_state.cpu(),
            atol=STATE_ATOL,
            rtol=STATE_RTOL,
        )


def _assert_matches_reference(batch: int, tokens: int, heads: int) -> None:
    q, k, v, g, beta = _inputs(batch, tokens, heads)
    initial_state = torch.randn(batch, heads, 128, 128, device=DEVICE)

    expected_out, expected_state = _reference(q, k, v, g, beta, initial_state.clone())
    actual_out, actual_state = run_kdn_decode(q, k, v, g, beta, initial_state.clone())
    torch.npu.synchronize()


    # err = (expected_out.float() - actual_out.float())
    # rel_l2 = err.norm() / expected_out.float().norm()      # aggregate correctness, well-calibrated
#    assert rel_l2 < 1e-3, rel_l2


    print(f"Errors in state: {(expected_state-actual_state).abs().max().item()}")
    print(f"Errors in out: {(expected_out-actual_out).abs().max().item()}")
    print(f"mag in out: {(expected_out).abs().max().item()}")
    print(f"bias in out: {(actual_out-expected_out)[expected_out.abs()>0.1].mean().item()}")
    print(f"signed bias: {(torch.sign(expected_out)*(actual_out-expected_out))[expected_out.abs()>0.1].mean().item()}")
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
    test_kdn_decode_exact_sparse_recurrence()
    test_kdn_decode_multiple_tokens()
    test_kdn_decode_exact_mixed_batch_heads()
    test_kdn_decode_exact_selective_decay()
    for B in [1, 2, 3, 17, 33]:
        for H in [1, 4, 16, 17, 32]:
            _assert_matches_reference(B, 1, H)
    test_kdn_decode_state_gather_and_skip()


def profile_kdn_decode_once(batch: int, tokens: int, heads: int, dim: int) -> None:
    q, k, v, g, beta = _inputs(
        batch, tokens, heads, dim, model_like=True
    )
    initial_state = (
        0.1 * torch.randn(batch, heads, dim, dim)
    ).to(DEVICE)
    load_kdn_decode(k_dim=dim, v_dim=dim)
    torch.npu.synchronize()
    run_kdn_decode(q, k, v, g, beta, initial_state)
    torch.npu.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run one decode invocation instead of the correctness suite",
    )
    parser.add_argument("--batch-size", type=int, default=1, metavar="B")
    parser.add_argument("--tokens", type=int, default=1, metavar="T")
    parser.add_argument("--heads", type=int, default=1, metavar="H")
    parser.add_argument("--dim", type=int, default=128, metavar="D")
    args = parser.parse_args()
    if args.profile:
        profile_kdn_decode_once( args.batch_size, args.tokens, args.heads, args.dim)
        print(
            "Profile decode complete: "
            f"B={args.batch_size}, T={args.tokens}, H={args.heads}, D={args.dim}"
        )
    else:
        test_kdn_decode()
        print("KDN decode checks passed")
