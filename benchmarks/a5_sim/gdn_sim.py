#!/usr/bin/env python3
"""Run MegaGDN PTO A5 chunk-GDN kernels under cannsim (Ascend950).

All tensor creation uses CPU-first helpers to avoid NPU dynamic ops that
cannsim rejects. Correctness checks reuse CPU references from test_single_kernels.

Usage::

    python3 gdn_sim.py --stage cumsum --mode correctness --n-seq 1 --l-seg 128 --H 16 --Hg 16
    python3 gdn_sim.py --stage kkt --mode bench --output-json outputs/gdn_kkt_smoke.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

_SIM_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SIM_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))

from common.torch_runtime import (  # noqa: E402
    cpu_to_npu,
    empty_npu,
    init_torch_npu,
    stream_ptr,
    sync,
    zeros_npu,
)
from megagdn_pto.compile import BLOCK_DIM  # noqa: E402
from megagdn_pto.kernel_libs import (  # noqa: E402
    load_chunk_cumsum,
    load_chunk_h,
    load_chunk_o,
    load_scaled_dot_kkt,
    load_wy_fast,
    run_chunk_cumsum,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)
from tests.test_single_kernels import (  # noqa: E402
    C,
    D,
    TestCase,
    ref_chunk_h,
    ref_chunk_o,
    ref_cumsum,
    ref_kkt,
    ref_wy,
    stats_ok,
)

C_PTO = C
# Ascend950 AI Core nominal frequency used to convert log_ca cycles → ms.
_AICORE_GHZ = float(os.environ.get("CANNSIM_AICORE_GHZ", "1.8"))
_LOG_CA_RE = re.compile(
    r"start:\s*(\d+),\s*tick:\s*(\d+).*?blkDim:\s*(\d+).*?done"
)


def _parse_hw_predicted_ms_from_log_ca() -> float | None:
    """Parse the last kernel block duration from cannsim log_ca dumps (ps/cycles)."""
    candidates = [Path("log_ca"), Path.cwd() / "log_ca"]
    export = os.environ.get("CANNSIM_EXPORT_DIR")
    if export:
        candidates.insert(0, Path(export) / "log_ca")
    log_ca: Path | None = next((p for p in candidates if p.is_dir()), None)
    if log_ca is None:
        return None

    spans: list[int] = []
    for dump in sorted(log_ca.glob("*.instr_log.dump")):
        for line in dump.read_text(errors="replace").splitlines():
            m = _LOG_CA_RE.search(line)
            if m:
                start, tick = int(m.group(1)), int(m.group(2))
                spans.append(tick - start)
    if not spans:
        return None
    cycles = max(spans)
    return cycles / (_AICORE_GHZ * 1_000_000.0)


STAGES = ("cumsum", "kkt", "wy_fast", "chunk_h", "chunk_o")
_KERNEL_NAMES = {
    "cumsum": "chunk_cumsum",
    "kkt": "scaled_dot_kkt",
    "wy_fast": "wy_fast",
    "chunk_h": "chunk_h",
    "chunk_o": "chunk_o",
}


def _vp(t: torch.Tensor | None) -> ctypes.c_void_p:
    if t is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(t.data_ptr())


def _transpose_gates_cpu(g_sum: torch.Tensor) -> torch.Tensor:
    return cpu_to_npu(g_sum.squeeze(0).t().contiguous())


def _transpose_beta_cpu(beta: torch.Tensor) -> torch.Tensor:
    return cpu_to_npu(beta.squeeze(0).t().contiguous())


def _make_case(n_seq: int, l_seg: int) -> TestCase:
    if n_seq == 1:
        return TestCase(f"smoke T={l_seg}", None, l_seg)
    cu = [i * l_seg for i in range(n_seq + 1)]
    return TestCase(f"smoke n_seq={n_seq} l_seg={l_seg}", cu, cu[-1])


@dataclass
class StageContext:
    tc: TestCase
    dev: torch.device
    H: int
    HG: int
    n_seq: int
    T: int
    stream: int
    block_dim: int
    cu: torch.Tensor | None
    cu_list: list[int] | None


def _build_context(
    tc: TestCase,
    dev: torch.device,
    H: int,
    HG: int,
    block_dim: int,
) -> StageContext:
    cu_list = tc.cu_seqlens_list
    n_seq = len(cu_list) - 1 if cu_list else 1
    cu = cpu_to_npu(torch.tensor(cu_list, dtype=torch.int32)) if cu_list else None
    return StageContext(
        tc=tc,
        dev=dev,
        H=H,
        HG=HG,
        n_seq=n_seq,
        T=tc.T,
        stream=stream_ptr(),
        block_dim=block_dim,
        cu=cu,
        cu_list=cu_list,
    )


def _seed_inputs(ctx: StageContext) -> None:
    torch.manual_seed(42)


def _bench_kernel(fn, *, warmup_iters: int = 0) -> tuple[float | None, float]:
    """Return (hw_predicted_ms, sim_wall_s) for a single timed launch."""
    sync()
    for _ in range(warmup_iters):
        fn()
        sync()

    sync()
    t0 = time.perf_counter()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    sync()
    sim_wall_s = time.perf_counter() - t0

    hw_ms: float | None
    try:
        hw_ms = start.elapsed_time(end)
    except (RuntimeError, AttributeError):
        hw_ms = None
    return hw_ms, sim_wall_s


def _run_cumsum(ctx: StageContext) -> tuple[callable, callable]:
    _seed_inputs(ctx)
    g_cpu = torch.randn(1, ctx.T, ctx.H, dtype=torch.float32)
    g_sum_cpu = torch.empty_like(g_cpu)
    g = cpu_to_npu(g_cpu)
    g_sum = cpu_to_npu(g_sum_cpu)

    def launch() -> None:
        run_chunk_cumsum(
            g,
            g_sum,
            stream=ctx.stream,
            chunk_size=C_PTO,
            cu_seqlens=ctx.cu,
            batch_size_override=ctx.n_seq,
            block_dim=ctx.block_dim,
        )

    def check() -> bool:
        ref = ref_cumsum(g_cpu, C_PTO, ctx.cu_list)
        return stats_ok(g_sum.cpu(), ref)

    return launch, check


def _run_kkt(ctx: StageContext) -> tuple[callable, callable]:
    _seed_inputs(ctx)
    k_cpu = F.normalize(torch.randn(1, ctx.T, ctx.HG, D, dtype=torch.float16), dim=-1, p=2)
    beta_cpu = torch.rand(1, ctx.T, ctx.H, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, ctx.T, ctx.H, dtype=torch.float32))
    g_sum_cpu = ref_cumsum(g_in, C_PTO, ctx.cu_list)
    g_t = _transpose_gates_cpu(g_sum_cpu)
    beta_t = _transpose_beta_cpu(beta_cpu)
    msk = cpu_to_npu(torch.tril(torch.ones(C_PTO, C_PTO), diagonal=-1).float())
    A_out = zeros_npu((1, ctx.T, ctx.H, C_PTO), torch.float16)
    k = cpu_to_npu(k_cpu)
    beta = cpu_to_npu(beta_cpu)
    g_sum = cpu_to_npu(g_sum_cpu)
    ws = zeros_npu((ctx.block_dim * 2, C_PTO, C_PTO), torch.float16)

    def launch() -> None:
        run_scaled_dot_kkt(
            k,
            beta,
            g_sum,
            msk,
            A_out,
            stream=ctx.stream,
            g_t=g_t,
            beta_t=beta_t,
            chunk_size=C_PTO,
            cu_seqlens=ctx.cu,
            batch_size_override=ctx.n_seq,
            block_dim=ctx.block_dim,
            key_heads=ctx.HG,
            workspace=ws,
        )

    def check() -> bool:
        ref = ref_kkt(k_cpu, beta_cpu, g_sum_cpu, C_PTO, ctx.cu_list)
        return stats_ok(A_out.float().cpu(), ref)

    return launch, check


def _run_wy_fast(ctx: StageContext) -> tuple[callable, callable]:
    _seed_inputs(ctx)
    k_cpu = F.normalize(torch.randn(1, ctx.T, ctx.HG, D, dtype=torch.float16), dim=-1, p=2)
    v_cpu = torch.randn(1, ctx.T, ctx.H, D, dtype=torch.float16)
    beta_cpu = torch.rand(1, ctx.T, ctx.H, dtype=torch.float16)
    A_cpu = torch.randn(1, ctx.T, ctx.H, C_PTO, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, ctx.T, ctx.H, dtype=torch.float32))
    g_sum_cpu = ref_cumsum(g_in, C_PTO, ctx.cu_list)
    g_t = _transpose_gates_cpu(g_sum_cpu)
    beta_t = _transpose_beta_cpu(beta_cpu)
    k = cpu_to_npu(k_cpu)
    v = cpu_to_npu(v_cpu)
    beta = cpu_to_npu(beta_cpu)
    g_sum = cpu_to_npu(g_sum_cpu)
    A = cpu_to_npu(A_cpu)
    w_out = empty_npu((1, ctx.T, ctx.H, D), torch.float16)
    u_out = empty_npu((1, ctx.T, ctx.H, D), torch.float16)
    ws_a1 = zeros_npu((ctx.block_dim, C_PTO, C_PTO), torch.float16)
    ws_a2 = zeros_npu((ctx.block_dim, C_PTO, C_PTO), torch.float16)

    def launch() -> None:
        run_wy_fast(
            k,
            v,
            beta,
            g_sum,
            A,
            w_out,
            u_out,
            stream=ctx.stream,
            g_t=g_t,
            beta_t=beta_t,
            chunk_size=C_PTO,
            cu_seqlens=ctx.cu,
            batch_size_override=ctx.n_seq,
            block_dim=ctx.block_dim,
            key_heads=ctx.HG,
            workspace_a1=ws_a1,
            workspace_a2=ws_a2,
        )

    def check() -> bool:
        w_ref, u_ref = ref_wy(k_cpu, v_cpu, beta_cpu, A_cpu, g_sum_cpu, C_PTO, ctx.cu_list)
        return stats_ok(w_out.float().cpu(), w_ref.float()) and stats_ok(
            u_out.float().cpu(), u_ref.float()
        )

    return launch, check


def _prepare_chunk_h_outputs(
    ctx: StageContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run chunk_h once to produce s_out and v_out for chunk_o (not timed)."""
    _seed_inputs(ctx)
    k_cpu = F.normalize(torch.randn(1, ctx.T, ctx.HG, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(1, ctx.T, ctx.H, D, dtype=torch.float16)
    u_cpu = torch.randn(1, ctx.T, ctx.H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, ctx.T, ctx.H, dtype=torch.float32))
    g_sum_cpu = ref_cumsum(g_in, C_PTO, ctx.cu_list)
    g_t = _transpose_gates_cpu(g_sum_cpu)
    tc_n = total_chunks(ctx.n_seq, ctx.T, C_PTO, ctx.cu)
    k = cpu_to_npu(k_cpu)
    w = cpu_to_npu(w_cpu)
    u = cpu_to_npu(u_cpu)
    s_out = zeros_npu((tc_n * ctx.H, D, D), torch.float16)
    v_out = empty_npu((1, ctx.T, ctx.H, D), torch.float16)
    fs_out = zeros_npu((ctx.n_seq * ctx.H, D, D), torch.float16)
    ws = zeros_npu((ctx.block_dim * 4, D, D), torch.float16)
    lib = load_chunk_h(ctx.H, D, C_PTO, key_heads=ctx.HG)
    lib.call_kernel(
        ctx.block_dim,
        ctx.stream,
        _vp(k),
        _vp(w),
        _vp(u),
        _vp(g_t),
        _vp(s_out),
        _vp(v_out),
        _vp(fs_out),
        _vp(ws),
        _vp(ctx.cu),
        1,
        ctx.T,
        ctx.T,
    )
    sync()
    return k_cpu, k, s_out, v_out, g_sum_cpu, g_t


def _run_chunk_h(ctx: StageContext) -> tuple[callable, callable]:
    _seed_inputs(ctx)
    k_cpu = F.normalize(torch.randn(1, ctx.T, ctx.HG, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(1, ctx.T, ctx.H, D, dtype=torch.float16)
    u_cpu = torch.randn(1, ctx.T, ctx.H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, ctx.T, ctx.H, dtype=torch.float32))
    g_sum_cpu = ref_cumsum(g_in, C_PTO, ctx.cu_list)
    g_t = _transpose_gates_cpu(g_sum_cpu)
    tc_n = total_chunks(ctx.n_seq, ctx.T, C_PTO, ctx.cu)
    k = cpu_to_npu(k_cpu)
    w = cpu_to_npu(w_cpu)
    u = cpu_to_npu(u_cpu)
    s_out = zeros_npu((tc_n * ctx.H, D, D), torch.float16)
    v_out = empty_npu((1, ctx.T, ctx.H, D), torch.float16)
    fs_out = zeros_npu((ctx.n_seq * ctx.H, D, D), torch.float16)
    ws = zeros_npu((ctx.block_dim * 4, D, D), torch.float16)
    lib = load_chunk_h(ctx.H, D, C_PTO, key_heads=ctx.HG)

    def launch() -> None:
        lib.call_kernel(
            ctx.block_dim,
            ctx.stream,
            _vp(k),
            _vp(w),
            _vp(u),
            _vp(g_t),
            _vp(s_out),
            _vp(v_out),
            _vp(fs_out),
            _vp(ws),
            _vp(ctx.cu),
            1,
            ctx.T,
            ctx.T,
        )

    def check() -> bool:
        h_ref, v_ref, fs_ref = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_sum_cpu, C_PTO, ctx.cu_list)
        ok_h = stats_ok(s_out.float().cpu().view(tc_n, ctx.H, D, D), h_ref.float())
        ok_v = stats_ok(v_out.float().cpu(), v_ref.float())
        ok_fs = stats_ok(fs_out.float().cpu().view(ctx.n_seq, ctx.H, D, D), fs_ref.float())
        return ok_h and ok_v and ok_fs

    return launch, check


def _run_chunk_o(ctx: StageContext) -> tuple[callable, callable]:
    k_cpu, k, s_out, v_out, g_sum_cpu, g_t = _prepare_chunk_h_outputs(ctx)
    q_cpu = F.normalize(torch.randn(1, ctx.T, ctx.HG, D, dtype=torch.float16), dim=-1, p=2)
    q = cpu_to_npu(q_cpu)
    tc_n = total_chunks(ctx.n_seq, ctx.T, C_PTO, ctx.cu)
    msk = cpu_to_npu(torch.tril(torch.ones(C_PTO, C_PTO), diagonal=0).float())
    o_out = empty_npu((1, ctx.T, ctx.H, D), torch.float16)
    ws_qk = zeros_npu((ctx.block_dim, C_PTO, C_PTO), torch.float16)
    ws_qs = zeros_npu((ctx.block_dim, C_PTO, D), torch.float16)
    ws_gated = zeros_npu((ctx.block_dim, C_PTO, C_PTO), torch.float16)
    lib_o = load_chunk_o(ctx.H, D, C_PTO, key_heads=ctx.HG)

    def launch() -> None:
        lib_o.call_kernel(
            ctx.block_dim,
            ctx.stream,
            _vp(q),
            _vp(k),
            _vp(v_out),
            _vp(s_out),
            _vp(g_t),
            _vp(msk),
            _vp(ws_qk),
            _vp(ws_qs),
            _vp(ws_gated),
            _vp(o_out),
            _vp(ctx.cu),
            1,
            ctx.T,
            ctx.T,
        )

    def check() -> bool:
        s_re = s_out.float().cpu().view(tc_n, ctx.H, D, D)
        o_ref = ref_chunk_o(q_cpu, k_cpu, v_out.cpu(), s_re, g_sum_cpu, C_PTO, ctx.cu_list)
        return stats_ok(o_out.float().cpu(), o_ref.float())

    return launch, check


_RUNNERS = {
    "cumsum": _run_cumsum,
    "kkt": _run_kkt,
    "wy_fast": _run_wy_fast,
    "chunk_h": _run_chunk_h,
    "chunk_o": _run_chunk_o,
}


def _compile_stage(stage: str, H: int, HG: int) -> None:
    if stage == "cumsum":
        load_chunk_cumsum(H, D, C_PTO)
    elif stage == "kkt":
        load_scaled_dot_kkt(H, D, C_PTO, key_heads=HG)
    elif stage == "wy_fast":
        load_wy_fast(H, D, C_PTO, key_heads=HG)
    elif stage == "chunk_h":
        load_chunk_h(H, D, C_PTO, key_heads=HG)
    elif stage == "chunk_o":
        load_chunk_o(H, D, C_PTO, key_heads=HG)


def _run_stage(
    stage: str,
    *,
    mode: str,
    n_seq: int,
    l_seg: int,
    H: int,
    HG: int,
    device: str,
    block_dim: int,
) -> dict:
    tc = _make_case(n_seq, l_seg)
    ctx = _build_context(tc, torch.device(device), H, HG, block_dim)
    print(f"\n==> stage={stage}  H={H}  Hg={HG}  T={ctx.T}", flush=True)

    _compile_stage(stage, H, HG)
    launch, check = _RUNNERS[stage](ctx)

    if mode == "bench":
        hw_ms, sim_wall_s = _bench_kernel(launch, warmup_iters=0)
        ok = check()
        if hw_ms is None or hw_ms <= 0:
            hw_ms = _parse_hw_predicted_ms_from_log_ca()
        row: dict = {
            "stage": stage,
            "kernel": _KERNEL_NAMES[stage],
            "label": tc.label,
            "N_seq": n_seq,
            "L_seg": l_seg,
            "T": ctx.T,
            "H": H,
            "Hg": HG,
            "D": D,
            "C": C_PTO,
            "block_dim": block_dim,
            "correctness_pass": ok,
            "correctness_msg": "PASS" if ok else "FAIL",
            "hw_predicted_ms": hw_ms,
            "sim_wall_s": sim_wall_s,
            "sim_wall_ms": sim_wall_s * 1000.0,
        }
        print(
            f"  {row['correctness_msg']}  {tc.label}  sim={sim_wall_s:.1f}s  hw={hw_ms}",
            flush=True,
        )
    else:
        launch()
        sync()
        ok = check()
        row = {
            "stage": stage,
            "kernel": _KERNEL_NAMES[stage],
            "label": tc.label,
            "N_seq": n_seq,
            "L_seg": l_seg,
            "T": ctx.T,
            "H": H,
            "Hg": HG,
            "D": D,
            "C": C_PTO,
            "block_dim": block_dim,
            "correctness_pass": ok,
            "correctness_msg": "PASS" if ok else "FAIL",
        }
        print(f"  {row['correctness_msg']}  {tc.label}", flush=True)

    if not ok:
        print(f"  CORRECTNESS FAIL", file=sys.stderr, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaGDN PTO A5 cannsim driver")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--mode", choices=("correctness", "bench"), default="correctness")
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--l-seg", type=int, default=128)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--Hg", type=int, default=16)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-dim", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.H % args.Hg != 0:
        raise SystemExit(f"H={args.H} must be divisible by Hg={args.Hg}")

    os.environ.setdefault("MEGAGDN_PTO_ARCH", "a5")
    init_torch_npu(args.device)
    block_dim = args.block_dim if args.block_dim is not None else int(
        os.environ.get("GDN_SIM_BLOCK_DIM", "1")
    )

    row = _run_stage(
        args.stage,
        mode=args.mode,
        n_seq=args.n_seq,
        l_seg=args.l_seg,
        H=args.H,
        HG=args.Hg,
        device=args.device,
        block_dim=block_dim,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arch": "dav-c310",
        "soc_cannsim": "Ascend950",
        "MEGAGDN_PTO_ARCH": os.environ.get("MEGAGDN_PTO_ARCH", "a5"),
        "mode": args.mode,
        "all_pass": row["correctness_pass"],
        "results": [row],
    }
    out = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(out)
    print(out)

    if not row["correctness_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
