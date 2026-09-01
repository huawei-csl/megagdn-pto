"""Benchmark PTO KDA decode bandwidth.

Benchmark the T=1 recurrent kda decode. It's vector-only, and memory bound.
The memory model is loading the state into, and out of the UB to GM:

    GB/s = 2 * B * H * V * K * sizeof(fp32) / elapsed_seconds / 1e9

Input vectors and output traffic are intentionally excluded from this model;
they are small compared with the recurrent state.

Two timings are reported per configuration:

``kernel``  one ``lib.call_kernel`` and nothing else, via ``prepare_kdn_decode``.
            Inputs are fp16 and the output buffer is preallocated, so no dtype
            conversion, allocation, zero-fill, validation or ctypes marshalling
            falls inside the timed window.  This is the number to compare
            against a memory-model roofline.
``e2e``     the full ``run_kdn_decode`` call on bf16 model tensors, which is
            what a caller pays today: five bf16->fp16 casts, an output
            allocation, an fp16 staging buffer and a copy back.  The gap between
            the two columns is launcher overhead, not kernel time.
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

from megagdn_pto.kdn_decode import prepare_kdn_decode, run_kdn_decode
from megagdn_pto.kdn_memcpy_bound import prepare_kdn_memcpy_bound
#from tests.ref_kda_decode import fused_recurrent_kda


def _time_npu(fn, *, warmup: int, iterations: int, cache_flush_mb: int = 0) -> float:
    """Return mean kernel time in milliseconds using device-side events."""
    flush = None
    if cache_flush_mb:
        flush = torch.empty(
            cache_flush_mb * 1024 * 1024, dtype=torch.int8,
            device=torch.npu.current_device(),
        )
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        if flush is not None:
            flush.zero_()
            torch.npu.synchronize()
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
        rows = sorted((row for row in results if int(row["dim"]) == dim), key=lambda r: (r["v_tile"], r["work_items"]))
        color = colors(0.2 + 0.7 * index / max(1, len(dims) - 1))
        ax.plot(
            [row["work_items"] for row in rows],
            [row["gb_per_s_kernel"] for row in rows],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=f"Kernel only K=V={dim}",
        )
        ax.plot(
            [row["work_items"] for row in rows],
            [row["gb_per_s_pto"] for row in rows],
            marker="x",
            linewidth=1.7,
            linestyle="--",
            markersize=6,
            color=color,
            alpha=0.8,
            label=f"End to end (bf16) K=V={dim}",
        )
        ax.plot(
            [row["work_items"] for row in rows],
            [row["gb_per_s_upper"] for row in rows],
            marker="s", linewidth=1.7, linestyle=":", markersize=5,
            color=color, alpha=0.9, label=f"GM↔UB ceiling K=V={dim}",
        )
    best = max(results, key=lambda row: row["gb_per_s_kernel"])
    ax.scatter(
        [best["work_items"]], [best["gb_per_s_kernel"]], s=120, marker="*",
        color="#d62728", edgecolor="white", linewidth=0.8, zorder=5,
        label=f"Kernel peak {best['gb_per_s_kernel']:.1f} GB/s",
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
    parser.add_argument("--v-tile-list", default="32", help="Comma-separated state tile heights")
    parser.add_argument("--warm", type=int, dest="warmup", default=5)
    parser.add_argument("--its", type=int, dest="iterations", default=20)
    parser.add_argument(
        "--cache-flush-mb", type=int, default=256,
        help="GM scratch written before each timed trial to reduce cache reuse (0 disables)",
    )
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
    v_tiles = [int(value) for value in args.v_tile_list.split(",") if value]
    results: list[dict] = []

    print("PTO KDA decode bandwidth benchmark")
    print(f"device={args.device}  warmup={args.warmup}  iterations={args.iterations} cache_flush={args.cache_flush_mb} MiB")
    print("model: 2 × fp32 state bytes (read + write); vectors/output excluded")
    print("kernel = one call_kernel launch (fp16 in, preallocated out); e2e = full run_kdn_decode on bf16")
    print(f"{'B':>4} {'H':>4} {'K=V':>5} {'BV':>4} {'kern ms':>10} {'kern GB/s':>11}"
          f" {'e2e ms':>10} {'e2e GB/s':>11} {'ovh %':>7} {'Copy ms':>10} {'Copy GB/s':>11}")
    print("-" * 106)

    for dim in dims:
        for v_tile in v_tiles:
            for batch in batches:
                for head in heads:
                    if dim % v_tile:
                        raise ValueError(f"dim={dim} must be divisible by v_tile={v_tile}")
                    shape = (batch, 1, head, dim)
                    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
                    k = torch.randn_like(q)
                    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
                    g = torch.randn_like(q)
                    beta = torch.randn((batch, 1, head), dtype=torch.bfloat16, device=device)
                    state = torch.randn((batch, head, dim, dim), dtype=torch.float32, device=device)

                    # Kernel-only: hoist every host-side step out of the timed
                    # window.  fp16 inputs mean no casts, and prepare_kdn_decode
                    # allocates the fp16 output and marshals the ctypes args once,
                    # so launch() is a single call_kernel.
                    q16, k16, v16, g16, beta16 = (
                        x.to(torch.float16) for x in (q, k, v, g, beta)
                    )
                    prepared = prepare_kdn_decode(
                        q16, k16, v16, g16, beta16, state,
                        v_tile=v_tile, block_dim=None,
                    )
                    elapsed_kernel_ms = _time_npu(
                        prepared.launch, warmup=args.warmup, iterations=args.iterations,
                        cache_flush_mb=args.cache_flush_mb,
                    )

                    # End to end: what a bf16 caller pays today.
                    def run() -> None:
                        run_kdn_decode(q, k, v, g, beta, state, v_tile=v_tile, block_dim=None)

                    elapsed_pto_ms = _time_npu(
                        run, warmup=args.warmup, iterations=args.iterations,
                        cache_flush_mb=args.cache_flush_mb,
                    )

                    # Timed the same way as the kernel above, so the ceiling is a
                    # ceiling: with the marshalling inside the window it came out
                    # *below* the kernel it is meant to bound.
                    copy_launch = prepare_kdn_memcpy_bound(
                        state, v_tile=v_tile, block_dim=None,
                    )
                    elapsed_upper_ms = _time_npu(
                        copy_launch, warmup=args.warmup, iterations=args.iterations,
                        cache_flush_mb=args.cache_flush_mb,
                    )

                    def run_reference() -> None:
                        pass
                        # fused_recurrent_kda(
                        #     q, k, v, g, beta, initial_state=state,
                        #     output_final_state=True, state_v_first=True,
                        # )

                    elapsed_ref_ms = 1000
                    #  _time_npu(
                    #     run_reference, warmup=args.warmup, iterations=args.iterations,
                    #     cache_flush_mb=args.cache_flush_mb,
                    # )
                    state_bytes = 2 * batch * head * dim * dim * 4
                    kernel_gb_per_s = state_bytes / (elapsed_kernel_ms * 1.0e6)
                    pto_gb_per_s = state_bytes / (elapsed_pto_ms * 1.0e6)
                    upper_gb_per_s = state_bytes / (elapsed_upper_ms * 1.0e6)
                    ref_gb_per_s = state_bytes / (elapsed_ref_ms * 1.0e6)
                    overhead_pct = 100.0 * (elapsed_pto_ms - elapsed_kernel_ms) / elapsed_pto_ms
                    row = {
                        "batch": batch, "heads": head, "dim": dim, "v_tile": v_tile,
                        "work_items": batch * head, "state_bytes": state_bytes,
                        "elapsed_ms_kernel": elapsed_kernel_ms, "gb_per_s_kernel": kernel_gb_per_s,
                        "elapsed_ms_pto": elapsed_pto_ms, "gb_per_s_pto": pto_gb_per_s,
                        "launcher_overhead_pct": overhead_pct,
                        "elapsed_ms_upper": elapsed_upper_ms, "gb_per_s_upper": upper_gb_per_s,
                        "elapsed_ms_ref": elapsed_ref_ms, "gb_per_s_ref": ref_gb_per_s,
                    }
                    results.append(row)
                    print(
                        f"{batch:4d} {head:4d} {dim:5d} {v_tile:4d} {elapsed_kernel_ms:10.4f}"
                        f" {kernel_gb_per_s:11.2f} {elapsed_pto_ms:10.4f} {pto_gb_per_s:11.2f}"
                        f" {overhead_pct:7.1f} {elapsed_upper_ms:10.4f} {upper_gb_per_s:11.2f}"
                    )
                    del prepared, copy_launch, q16, k16, v16, g16, beta16
                    del q, k, v, g, beta, state
                    gc.collect()
                    torch.npu.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "state_dtype": "fp32",
        "traffic_model": "2 * B * H * V * K * sizeof(fp32)",
        "cache_flush_mb": args.cache_flush_mb,
        "cache_flush_note": "scratch GM write before each timed trial; reduces cache reuse but is not a hardware cache invalidation",
        "results": results,
    }, indent=2) + "\n")
    print(f"Saved results: {args.output_json}")
    if not args.no_plot:
        _plot(results, args.plot)


if __name__ == "__main__":
    main()
