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
    use_qk_l2norm: bool = False,
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
        use_qk_l2norm (bool):
            Normalize ``q`` and ``k`` over ``K`` before scaling, using the fused
            reference's ``x / (sqrt(sum(x*x)) + 1e-6)``.

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
    if use_qk_l2norm:
        q = q / (q.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        k = k / (k.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
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
    normalize_qk: bool = True,
):
    torch.manual_seed(seed)
    shape = (batch, tokens, heads, dim)
    if normalize_qk:
        q = F.normalize(torch.randn(shape), p=2, dim=-1)
        k = F.normalize(torch.randn(shape), p=2, dim=-1)
    else:
        # Per-token magnitudes far from 1 so in-kernel l2norm cannot be a no-op.
        q = torch.randn(shape) * (0.25 + 4 * torch.rand(shape[:-1] + (1,)))
        k = torch.randn(shape) * (0.25 + 4 * torch.rand(shape[:-1] + (1,)))
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


def _reference(
    q,
    k,
    v,
    g,
    beta,
    state,
    *,
    state_indices=None,
    cu_seqlens=None,
    l2norm=False,
):
    """Mirror ``run_kdn_decode``: every sequence runs on its own state slot.

    ``state`` is the kernel's V-first ``[slots, H, V, K]`` fp32 layout and is
    returned updated (out of place).  Without ``cu_seqlens`` sequence ``n`` is
    batch row ``n``; with it the sequences are the ``cu_seqlens`` spans of the
    single flattened token axis.
    """
    batch, tokens = q.shape[:2]
    if cu_seqlens is None:
        bounds = [n * tokens for n in range(batch + 1)]
    else:
        bounds = cu_seqlens.tolist()
    # [B, T, ...] is contiguous, so flattening it is exactly the varlen layout.
    flat = [x.reshape(1, batch * tokens, *x.shape[2:]) for x in (q, k, v, g, beta)]
    out = torch.zeros_like(flat[2])
    final_state = state.clone()
    for n in range(len(bounds) - 1):
        bos, eos = bounds[n], bounds[n + 1]
        slot = n if state_indices is None else int(state_indices[n])
        if eos <= bos or slot < 0:
            continue
        out_n, state_n = naive_recurrent_kda(
            *(x[:, bos:eos] for x in flat),
            initial_state=state[slot : slot + 1].transpose(-1, -2),
            output_final_state=True,
            use_qk_l2norm=l2norm,
        )
        out[:, bos:eos] = out_n
        final_state[slot] = state_n[0].transpose(-1, -2)
    return out.reshape_as(v), final_state.contiguous()


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
        q, k, v, g, beta, state_before, state_indices=indices
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_out.float().cpu(), expected_out.float().cpu(), atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.cpu(), atol=STATE_ATOL, rtol=STATE_RTOL
    )
    assert torch.count_nonzero(actual_out[1]).item() == 0
    torch.testing.assert_close(actual_state[0].cpu(), state_before[0].cpu())
    torch.testing.assert_close(actual_state[1].cpu(), state_before[1].cpu())


def test_kdn_decode_qk_l2norm() -> None:
    """In-kernel q/k L2 normalization matches the reference and changes the result."""
    batch, heads, dim = 3, 4, 128
    for tokens in (1, 4):
        q, k, v, g, beta = _inputs(
            batch, tokens, heads, dim, seed=7 + tokens,
            model_like=True, normalize_qk=False,
        )
        state = (0.1 * torch.randn(batch, heads, dim, dim)).to(DEVICE)

        expected_out, expected_state = _reference(
            q, k, v, g, beta, state, l2norm=True
        )
        actual_out, actual_state = run_kdn_decode(
            q, k, v, g, beta, state.clone(), use_qk_l2norm=True
        )
        plain_out, _ = run_kdn_decode(q, k, v, g, beta, state.clone())
        torch.npu.synchronize()

        torch.testing.assert_close(
            actual_out.float().cpu(), expected_out.float().cpu(),
            atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL,
        )
        torch.testing.assert_close(
            actual_state.cpu(), expected_state.cpu(),
            atol=STATE_ATOL, rtol=STATE_RTOL,
        )
        # Guard against the flag being ignored: unnormalized inputs must differ.
        assert (plain_out.float() - actual_out.float()).abs().max().item() > 1e-2


def test_kdn_decode_state_indices_dense() -> None:
    """Shuffled non-contiguous slots over an oversized pool: the real decode path.

    Production callers always gather, so this covers the case at head/batch
    counts big enough to spread work over all 48 vector workers, with a slot
    permutation that gives no worker a sequential run of slots.
    """
    for batch, heads in ((17, 32), (33, 16)):
        q, k, v, g, beta = _inputs(batch, 1, heads, 128, seed=1000 + batch)
        slots = 2 * batch + 3
        state = torch.randn(slots, heads, 128, 128, device=DEVICE)
        state_before = state.clone()
        indices = torch.randperm(slots)[:batch].to(torch.int32).to(DEVICE)

        expected_out, expected_state = _reference(
            q, k, v, g, beta, state_before, state_indices=indices
        )
        actual_out, actual_state = run_kdn_decode(
            q, k, v, g, beta, state, state_indices=indices
        )
        torch.npu.synchronize()

        torch.testing.assert_close(
            actual_out.float().cpu(), expected_out.float().cpu(),
            atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL,
        )
        # Covers both the visited slots and the pool entries that must not move.
        torch.testing.assert_close(
            actual_state.cpu(), expected_state.cpu(),
            atol=STATE_ATOL, rtol=STATE_RTOL,
        )


def test_kdn_decode_state_out_rejects_indices() -> None:
    """Out-of-place plus gather would silently leave most slots undefined."""
    q, k, v, g, beta = _inputs(batch=2, tokens=1, heads=2)
    state = torch.randn(4, 2, 128, 128, device=DEVICE)
    indices = torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)
    try:
        run_kdn_decode(
            q, k, v, g, beta, state,
            state_indices=indices, state_out=torch.empty_like(state),
        )
    except ValueError:
        return
    raise AssertionError("state_out combined with state_indices must raise")


def test_kdn_decode_state_out_of_place() -> None:
    """``state_out`` reproduces the in-place update and leaves the input alone."""
    batch, tokens, heads, dim = 3, 2, 4, 128
    q, k, v, g, beta = _inputs(batch, tokens, heads, dim, seed=11, model_like=True)
    state = (0.1 * torch.randn(batch, heads, dim, dim)).to(DEVICE)

    inplace_out, inplace_state = run_kdn_decode(q, k, v, g, beta, state.clone())
    source = state.clone()
    # Deliberately uninitialized: every slot is visited here, so the kernel owns
    # all of it and nothing may leak through from the destination buffer.
    dest = torch.empty_like(source)
    oop_out, oop_state = run_kdn_decode(q, k, v, g, beta, source, state_out=dest)
    torch.npu.synchronize()

    assert oop_state.data_ptr() == dest.data_ptr()
    _assert_exact(oop_out, inplace_out)
    _assert_exact(oop_state, inplace_state)
    _assert_exact(source, state)  # the read-only input state is untouched


def test_kdn_decode_varlen() -> None:
    """``cu_seqlens`` packs unequal sequences (including empty) onto one axis."""
    lengths = [1, 3, 0, 5, 2]
    total, heads, dim = sum(lengths), 4, 128
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32, device=DEVICE,
    )
    # Non-contiguous slots, one skipped sequence, two slots left untouched.
    indices = torch.tensor([4, -1, 1, 3, 0], dtype=torch.int32, device=DEVICE)

    q, k, v, g, beta = _inputs(
        1, total, heads, dim, seed=99, model_like=True, normalize_qk=False
    )
    state = (0.1 * torch.randn(7, heads, dim, dim)).to(DEVICE)
    state_before = state.clone()

    expected_out, expected_state = _reference(
        q, k, v, g, beta, state_before,
        state_indices=indices, cu_seqlens=cu_seqlens, l2norm=True,
    )
    actual_out, actual_state = run_kdn_decode(
        q, k, v, g, beta, state,
        state_indices=indices, cu_seqlens=cu_seqlens, use_qk_l2norm=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        actual_out.float().cpu(), expected_out.float().cpu(),
        atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL,
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.cpu(),
        atol=STATE_ATOL, rtol=STATE_RTOL,
    )
    # The skipped sequence writes no output, and unused slots stay untouched.
    assert torch.count_nonzero(actual_out[0, 1:4]).item() == 0
    for slot in (2, 5, 6):
        torch.testing.assert_close(actual_state[slot].cpu(), state_before[slot].cpu())


def test_kdn_decode_varlen_matches_dense() -> None:
    """Equal-length ``cu_seqlens`` spans reproduce the dense [B, T] path exactly."""
    batch, tokens, heads, dim = 4, 3, 4, 128
    q, k, v, g, beta = _inputs(batch, tokens, heads, dim, seed=5, model_like=True)
    state = (0.1 * torch.randn(batch, heads, dim, dim)).to(DEVICE)
    cu_seqlens = torch.arange(
        0, batch * tokens + 1, tokens, dtype=torch.int32, device=DEVICE
    )

    dense_out, dense_state = run_kdn_decode(q, k, v, g, beta, state.clone())
    flat = [x.reshape(1, batch * tokens, *x.shape[2:]) for x in (q, k, v, g, beta)]
    varlen_out, varlen_state = run_kdn_decode(
        *flat, state.clone(), cu_seqlens=cu_seqlens
    )
    torch.npu.synchronize()

    _assert_exact(varlen_out.reshape_as(dense_out), dense_out)
    _assert_exact(varlen_state, dense_state)


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
    test_kdn_decode_qk_l2norm()
    test_kdn_decode_state_indices_dense()
    test_kdn_decode_state_out_rejects_indices()
    test_kdn_decode_state_out_of_place()
    test_kdn_decode_varlen()
    test_kdn_decode_varlen_matches_dense()


def profile_kdn_decode_once(
    batch: int, tokens: int, heads: int, dim: int, l2norm: bool = False,
    indices: bool = False, v_tile: int = 128,
) -> None:
    q, k, v, g, beta = _inputs(
        batch, tokens, heads, dim, model_like=True, normalize_qk=not l2norm
    )
    # Production always gathers, so profile that shape by default: an oversized
    # pool addressed through a shuffled index vector.
    slots = 2 * batch + 3 if indices else batch
    initial_state = (
        0.1 * torch.randn(slots, heads, dim, dim)
    ).to(DEVICE)
    state_indices = None
    if indices:
        state_indices = torch.randperm(slots)[:batch].to(torch.int32).to(DEVICE)
    load_kdn_decode(k_dim=dim, v_dim=dim, v_tile=v_tile)
    torch.npu.synchronize()
    run_kdn_decode(
        q, k, v, g, beta, initial_state, use_qk_l2norm=l2norm,
        state_indices=state_indices, v_tile=v_tile,
    )
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
    parser.add_argument(
        "--l2norm",
        action="store_true",
        help="enable the in-kernel q/k L2 normalization while profiling",
    )
    parser.add_argument(
        "--indices",
        action="store_true",
        help="gather state through a shuffled state_indices vector (production shape)",
    )
    parser.add_argument("--v-tile", type=int, default=128, metavar="BV")
    args = parser.parse_args()
    if args.profile:
        profile_kdn_decode_once(
            args.batch_size, args.tokens, args.heads, args.dim, args.l2norm,
            args.indices, args.v_tile,
        )
        print(
            "Profile decode complete: "
            f"B={args.batch_size}, T={args.tokens}, H={args.heads}, D={args.dim}, "
            f"l2norm={args.l2norm}, indices={args.indices}, v_tile={args.v_tile}"
        )
    else:
        test_kdn_decode()
        print("KDN decode checks passed")
