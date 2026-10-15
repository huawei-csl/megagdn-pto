#!/usr/bin/env python3
"""Reproduce solve_tril on Qwen3.8-27B layer-0 checkpoint values.

This intentionally obtains ``A`` from pypto-lib's full quantized block
reference.  The synthetic inputs used by ``test_gdn_single_kernels.py`` do not
exercise the trained layer's correlated keys and therefore cannot reproduce
the fp16 doubling overflow.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch_npu  # noqa: F401

from megagdn_pto.fast_inverse import solve_tril
from tests.test_gdn_single_kernels import ACCURACY


def _load_pypto_reference(pypto_lib: Path):
    model_dir = pypto_lib / "models" / "qwen3_8_27b"
    if not (model_dir / "reference.py").is_file():
        raise FileNotFoundError(f"Qwen3.8-27B reference not found under {pypto_lib}")
    sys.path.insert(0, str(model_dir))
    config = importlib.import_module("config")
    reference = importlib.import_module("reference")
    return config, reference


def _matrix_view(a: torch.Tensor, chunk: int) -> torch.Tensor:
    """Convert ``[T, H, chunk]`` to real ``[nchunk, H, chunk, chunk]`` matrices."""
    t, h, width = a.shape
    if width != chunk or t % chunk:
        raise ValueError(
            f"expected [T, H, {chunk}] with T divisible by {chunk}, got {a.shape}"
        )
    return a.reshape(t // chunk, chunk, h, chunk).permute(0, 2, 1, 3).contiguous()


def _doubling_float64(a: torch.Tensor) -> torch.Tensor:
    """Exact-arithmetic control for the matrix view and reference inverse."""
    n = a.shape[-1]
    eye = torch.eye(n, dtype=torch.float64).expand_as(a)
    x = eye - a
    y = a @ a
    for level in range(n.bit_length() - 2):
        x = x + x @ y
        if level + 1 < n.bit_length() - 2:
            y = y @ y
    return x


def run(pypto_lib: Path, checkpoint: Path, device: str, seq_len: int) -> bool:
    """Invert a trained layer's ``A`` on device and check it against fp64."""
    config, reference = _load_pypto_reference(pypto_lib.resolve())
    cfg = config.QWEN3_8_27B
    chunk = config.GDN_TILING.chunk
    heads = cfg.linear_num_value_heads
    if seq_len % chunk:
        raise ValueError(f"--seq-len must be divisible by chunk={chunk}")

    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    quantized_weights = reference.quantize_weights(weights)
    hidden = reference.make_block_inputs(seq_len, cfg)
    a = reference.block(
        hidden,
        quantized_weights,
        cfg,
        chunk,
        quant_x=True,
        quant_y=True,
    )["a"]
    a_fp16 = a.to(torch.float16)
    a_matrices = _matrix_view(a_fp16, chunk)

    eye = torch.eye(chunk, dtype=torch.float64).expand_as(a_matrices)
    expected = torch.linalg.inv(eye + a_matrices.double())
    control = _doubling_float64(a_matrices.double())
    control_frob = torch.linalg.vector_norm(control - expected) / torch.linalg.vector_norm(
        expected
    )
    if not torch.isfinite(control).all() or control_frob > 1e-8:
        raise AssertionError(
            f"float64 doubling control failed (relative Frobenius error {control_frob:.3e}); "
            "check the [T,H,chunk] matrix reshape"
        )

    torch.npu.set_device(device)
    dev = torch.device(device)
    actual_bsnd = solve_tril(
        a_fp16.unsqueeze(0).contiguous().to(dev),
        None,
        chunk,
        heads,
    )
    torch.npu.synchronize()
    actual = _matrix_view(actual_bsnd[0].cpu(), chunk)

    finite = bool(torch.isfinite(actual).all())
    nonfinite = int((~torch.isfinite(actual)).sum())
    diff = actual.double() - expected
    frob = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected))
    a_inf = float(a_matrices.double().abs().sum(dim=-1).max())
    max_diff = float(diff.abs().max())
    print(
        f"Qwen3.8-27B layer 0: T={seq_len} H={heads} chunk={chunk} "
        f"matrices={a_matrices.shape[0] * heads}"
    )
    print(f"A infinity norm={a_inf:.6g}; float64 control frob={float(control_frob):.3e}")
    print(
        f"kernel finite={finite} nonfinite={nonfinite}; frob={frob:.6g}; "
        f"max diff={max_diff:.6g}"
    )
    ok = finite and ACCURACY.stats_ok(actual, expected, chunk_size=chunk)
    print("ACCURACY.stats_ok: " + ("PASS" if ok else "FAIL"))
    return ok


def test_qwen3_8_27b_layer0_solve_tril() -> None:
    """Collectible form of the check above.

    It needs a Qwen3.8-27B layer-0 checkpoint and an NPU, so it skips rather
    than fails when either is missing. Point it at them with ``PYPTO_LIB`` and
    ``QWEN3_8_27B_CHECKPOINT``, or run this file directly with the flags below.
    """
    import pytest

    pypto_lib = Path(os.environ.get("PYPTO_LIB", ""))
    checkpoint = Path(os.environ.get("QWEN3_8_27B_CHECKPOINT", ""))
    if not (pypto_lib / "models" / "qwen3_8_27b" / "reference.py").is_file():
        pytest.skip("set PYPTO_LIB to a pypto-lib checkout")
    if not checkpoint.is_file():
        pytest.skip("set QWEN3_8_27B_CHECKPOINT to a layer-0 checkpoint")
    if not torch.npu.is_available():
        pytest.skip("no NPU available")
    assert run(pypto_lib, checkpoint, os.environ.get("GDN_NPU_DEVICE", "npu:0"), 512)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-lib", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    ok = run(args.pypto_lib, args.checkpoint, args.device, args.seq_len)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
