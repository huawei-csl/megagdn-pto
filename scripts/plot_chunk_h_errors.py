#!/usr/bin/env python3
"""Generate chunk_h error histograms against the CPU reference."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megagdn_pto.kernel_libs import run_chunk_h, total_chunks, transpose_gates
from tests.test_single_kernels import (
    C,
    D,
    TestCase,
    build_test_cases,
)
from tests.utils import NumericalAccuracy

ACCURACY = NumericalAccuracy()

DEFAULT_HEAD_CONFIGS = [
    (16, 4),
    (16, 16),
    (24, 8),
    (32, 8),
    (48, 16),
    (64, 16),
]


class ChunkHErrorCollector:
    def __init__(self, *, rel_mag_min: float, nbins: int) -> None:
        self.rel_mag_min = rel_mag_min
        self.nbins = nbins
        self.errs = {
            f"{out}_{kind}": []
            for out in ("h", "v", "fs") for kind in ("abs", "rel")
        }
        self.vals = {out: [] for out in ("h", "v", "fs")}
        self.abs_min = float("inf")
        self.abs_max = 0.0
        self.abs_max_by_output = {out: 0.0 for out in ("h", "v", "fs")}
        self.val_max_by_output = {out: 0.0 for out in ("h", "v", "fs")}

    def add(self, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        diff = (actual - expected).abs().numpy().ravel()
        mag = expected.abs().numpy().ravel()
        rel_mask = np.isfinite(diff) & np.isfinite(mag) & (mag > self.rel_mag_min)
        rel = diff[rel_mask] / mag[rel_mask]

        self.abs_min = min(self.abs_min, float(diff.min()))
        self.abs_max = max(self.abs_max, float(diff.max()))
        self.abs_max_by_output[name] = max(self.abs_max_by_output[name], float(diff.max()))
        self.val_max_by_output[name] = max(self.val_max_by_output[name], float(mag.max()))

        mag = mag[np.isfinite(mag)]
        if mag.size:
            self.vals[name].append(mag.astype(np.float32, copy=False))
        for kind, vals in (("abs", diff), ("rel", rel)):
            vals = vals[np.isfinite(vals)]
            if vals.size:
                self.errs[f"{name}_{kind}"].append(vals.astype(np.float32, copy=False))

    def plot(self, path: str) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
        mn = 0.0 if self.abs_min == float("inf") else self.abs_min
        fig.suptitle(f"chunk_h total abs error: min={mn:.3e} max={self.abs_max:.3e}")

        ylabels = ("abs", f"rel |ref|>{self.rel_mag_min:g}", "mag")
        for ax, kind in zip(axes[:, 0], ylabels):
            ax.set_ylabel(f"{kind} count")

        for j, out in enumerate(("h", "v", "fs")):
            for i, kind in enumerate(("abs", "rel")):
                vals = self._concat(self.errs[f"{out}_{kind}"])
                hi = max(float(vals.max()) if vals.size else 1e-4, 1e-12)
                hist, bins = np.histogram(vals, np.linspace(0.0, hi, self.nbins + 1))
                axes[i, j].set_title(f"{out} {kind} max_abs={self.abs_max_by_output[out]:.3e}")
                axes[i, j].stairs(hist, bins)
                self._format_hist_axis(axes[i, j], hist, bins, "error")

            vals = self._concat(self.vals[out])
            hi = max(float(vals.max()) if vals.size else 1e-4, 1e-12)
            hist, bins = np.histogram(vals, np.linspace(0.0, hi, self.nbins + 1))
            axes[2, j].set_title(f"{out} |ref| max={self.val_max_by_output[out]:.3e}")
            axes[2, j].stairs(hist, bins)
            self._format_hist_axis(axes[2, j], hist, bins, "magnitude")

        fig.savefig(path, dpi=160)

    @staticmethod
    def _concat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    @staticmethod
    def _format_hist_axis(ax, hist: np.ndarray, bins: np.ndarray, xlabel: str) -> None:
        if hist.sum() > 0:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(xlabel)


def _parse_head_configs(raw: str | None, h_list: str, hg: int) -> list[tuple[int, int]]:
    if raw:
        configs: list[tuple[int, int]] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            h_s, hg_s = item.split(":", 1)
            configs.append((int(h_s), int(hg_s)))
        return configs
    return [(int(x), hg) for x in h_list.split(",") if x.strip()]


def _collect_case(
    collector: ChunkHErrorCollector,
    tc: TestCase,
    dev: torch.device,
    H: int,
    HG: int,
) -> bool:
    T = tc.T
    cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev) if tc.cu_seqlens_list else None
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1

    torch.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = tc.ref_gdn.cumsum(g_in.cpu(), C, tc.cu_seqlens_list).to(
        device=dev, dtype=torch.float32
    )
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    k_h = k.repeat_interleave(H // HG, dim=2)
    w = (k_h * beta[..., None] * torch.exp(g_sum)[..., None]).to(torch.float16)
    u = (v * beta[..., None]).to(torch.float16)
    g_t = transpose_gates(g_sum)

    tc_n = total_chunks(N_seq, T, C, cu)
    h0 = (0.01 * torch.randn(N_seq, H, D, D, device=dev, dtype=torch.float16)).contiguous()
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq, H, D, D, device=dev, dtype=torch.float16)

    run_chunk_h(
        k, w, u, g_sum, s_out, v_out, fs_out,
        g_t=g_t, chunk_size=C,
        cu_seqlens=cu, batch_size_override=N_seq, key_heads=HG,
        initial_state=h0,
    )
    torch.npu.synchronize()

    h_ref, v_ref, fs_ref = tc.ref_gdn.chunk_h(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g_sum.cpu(),
        C,
        tc.cu_seqlens_list,
        initial_state=h0.cpu(),
    )
    h_act = s_out.float().cpu().view(tc_n, H, D, D)
    v_act = v_out.float().cpu()
    fs_act = fs_out.float().cpu()

    collector.add("h", h_act, h_ref.float())
    collector.add("v", v_act, v_ref.float())
    collector.add("fs", fs_act, fs_ref.float())

    return (
        ACCURACY.stats_ok(h_act, h_ref.float(), chunk_size=C)
        and ACCURACY.stats_ok(v_act, v_ref.float(), chunk_size=C)
        and ACCURACY.stats_ok(fs_act, fs_ref.float(), chunk_size=C)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--quick", action="store_true", help="Run a single small test case.")
    parser.add_argument("--H-list", default="16,32,48,64", help="Comma-separated value head counts.")
    parser.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    parser.add_argument("--head-configs", default=None, help="Comma-separated H:Hg pairs.")
    parser.add_argument("--output", default="chunk_h_error_hist.png")
    parser.add_argument("--nbins", type=int, default=50)
    parser.add_argument("--rel-mag-min", type=float, default=1e-3)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    default_heads = args.H_list == "16,32,48,64" and args.hg == 16
    head_configs = (
        DEFAULT_HEAD_CONFIGS
        if args.head_configs is None and default_heads
        else _parse_head_configs(args.head_configs, args.H_list, args.hg)
    )
    cases = [TestCase("quick T=128", None, 128)] if args.quick else build_test_cases()
    collector = ChunkHErrorCollector(rel_mag_min=args.rel_mag_min, nbins=args.nbins)

    all_pass = True
    for H, HG in head_configs:
        assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
        for tc in cases:
            ok = _collect_case(collector, tc, dev, H, HG)
            all_pass = all_pass and ok
            print(f"{'PASS' if ok else 'FAIL'} H={H} Hg={HG} {tc.label}")

    collector.plot(args.output)
    print(f"chunk_h error histogram written to {args.output}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
