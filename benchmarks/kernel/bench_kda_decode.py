"""Benchmark PTO KDA decode bandwidth.

Benchmark the T=1 recurrent kda decode. It's vector-only, and memory bound.
The memory model is loading the state into, and out of the UB to GM:

    GB/s = 2 * B * H * V * K * sizeof(fp32) / elapsed_seconds / 1e9

Input vectors and output traffic are intentionally excluded from this model;
they are small compared with the recurrent state.  
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from megagdn_pto.kdn_decode import run_kdn_decode
from tests.ref_kda_decode import fused_recurrent_kda


def _time_npu(fn, *, warmup: int, iterations: int) -> float:
    """Return mean kernel time in milliseconds using device-side events."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / iterations


def _plot(results: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib is unavailable; skipping plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    dims = sorted({int(row["dim"]) for row in results})
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")
    for index, dim in enumerate(dims):
        rows = sorted((row for row in results if int(row["dim"]) == dim), key=lambda r: r["work_items"])
        color = colors(0.2 + 0.7 * index / max(1, len(dims) - 1))
        ax.plot(
            [row["work_items"] for row in rows],
            [row["gb_per_s_pto"] for row in rows],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=f"PTO K=V={dim}",
        )
        ax.plot(
            [row["work_items"] for row in rows],
            [row["gb_per_s_ref"] for row in rows],
            marker="x",
            linewidth=1.7,
            linestyle="--",
            markersize=6,
            color=color,
            alpha=0.8,
            label=f"Reference K=V={dim}",
        )
    best = max(results, key=lambda row: row["gb_per_s_pto"])
    ax.scatter(
        [best["work_items"]], [best["gb_per_s_pto"]], s=120, marker="*",
        color="#d62728", edgecolor="white", linewidth=0.8, zorder=5,
        label=f"PTO peak {best['gb_per_s_pto']:.1f} GB/s",
    )
    ax.set_title("PTO KDA decode state bandwidth", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Independent work items (batch × heads)")
    ax.set_ylabel("Effective state bandwidth (GB/s)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--batch-list", default="1,8,64,200", help="Comma-separated batch sizes")
    parser.add_argument("--heads-list", default="1,2,4,8,16,32,64", help="Comma-separated head counts")
    parser.add_argument("--dim-list", default="128", help="Comma-separated K=V dimensions")
    parser.add_argument("--warm", type=int, dest="warmup", default=5)
    parser.add_argument("--its", type=int, dest="iterations", default=20)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/data/kda_decode_bandwidth.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/figure/kda_decode_bandwidth.png"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    batches = [int(value) for value in args.batch_list.split(",") if value]
    heads = [int(value) for value in args.heads_list.split(",") if value]
    dims = [int(value) for value in args.dim_list.split(",") if value]
    results: list[dict] = []

    print("PTO KDA decode bandwidth benchmark")
    print(f"device={args.device}  warmup={args.warmup}  iterations={args.iterations}")
    print("model: 2 × fp32 state bytes (read + write); vectors/output excluded")
    print(f"{'B':>4} {'H':>4} {'K=V':>5} {'PTO ms':>10} {'PTO GB/s':>12} {'Ref ms':>10} {'Ref GB/s':>12}")
    print("-" * 72)

    for dim in dims:
        for batch in batches:
            for head in heads:
                shape = (batch, 1, head, dim)
                q = torch.randn(shape, dtype=torch.bfloat16, device=device)
                k = torch.randn_like(q)
                v = torch.randn(shape, dtype=torch.bfloat16, device=device)
                g = torch.randn_like(q)
                beta = torch.randn((batch, 1, head), dtype=torch.bfloat16, device=device)
                state = torch.randn((batch, head, dim, dim), dtype=torch.float32, device=device)

                def run() -> None:
                    run_kdn_decode(q, k, v, g, beta, state, block_dim=None)

                elapsed_pto_ms = _time_npu(run, warmup=args.warmup, iterations=args.iterations)

                def run_reference() -> None:
                    fused_recurrent_kda(
                        q, k, v, g, beta, initial_state=state,
                        output_final_state=True, state_v_first=True,
                    )

                elapsed_ref_ms = _time_npu(
                    run_reference, warmup=args.warmup, iterations=args.iterations
                )
                state_bytes = 2 * batch * head * dim * dim * 4
                pto_gb_per_s = state_bytes / (elapsed_pto_ms * 1.0e6)
                ref_gb_per_s = state_bytes / (elapsed_ref_ms * 1.0e6)
                row = {
                    "batch": batch, "heads": head, "dim": dim,
                    "work_items": batch * head, "state_bytes": state_bytes,
                    "elapsed_ms_pto": elapsed_pto_ms, "gb_per_s_pto": pto_gb_per_s,
                    "elapsed_ms_ref": elapsed_ref_ms, "gb_per_s_ref": ref_gb_per_s,
                }
                results.append(row)
                print(
                    f"{batch:4d} {head:4d} {dim:5d} {elapsed_pto_ms:10.4f}"
                    f" {pto_gb_per_s:12.2f} {elapsed_ref_ms:10.4f} {ref_gb_per_s:12.2f}"
                )
                del q, k, v, g, beta, state
                gc.collect()
                torch.npu.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "state_dtype": "fp32",
        "traffic_model": "2 * B * H * V * K * sizeof(fp32)",
        "results": results,
    }, indent=2) + "\n")
    print(f"Saved results: {args.output_json}")
    if not args.no_plot:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
