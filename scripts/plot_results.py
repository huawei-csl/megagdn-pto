#!/usr/bin/env python3
"""Plot benchmark results: kernel speedups and vLLM prefill throughput.

Reads JSONL files from the prefill benchmark and JSON files from the kernel
benchmark, then produces figures in outputs/figure/.

Usage::

    # After running benchmarks/vllm_prefill/run_prefill_sweep.sh
    python scripts/plot_results.py --prefill-dir outputs/data/prefill_<stamp>

    # Plot all available data
    python scripts/plot_results.py --auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIGURE_DIR = _REPO_ROOT / "outputs" / "figure"

CASE_STYLE = {
    "pto_mega": {"color": "#1f77b4", "label": "PTO megakernel", "marker": "o", "linestyle": "-"},
    "pto":      {"color": "#ff7f0e", "label": "PTO staged",     "marker": "s", "linestyle": "-"},
    "triton":   {"color": "#444444", "label": "Triton (baseline)", "marker": "^", "linestyle": "--"},
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_seq_len(rows: list[dict]) -> dict[int, dict]:
    return {int(r["seq_len"]): r for r in rows}


def plot_prefill_model(model_dir: Path, out_dir: Path) -> None:
    cases = ["triton", "pto", "pto_mega"]
    data: dict[str, dict[int, dict]] = {}
    for c in cases:
        p = model_dir / f"{c}.jsonl"
        if not p.is_file():
            print(f"  [skip] {p} not found")
            return
        rows = _load_jsonl(p)
        if not rows:
            return
        data[c] = _by_seq_len(rows)

    model_name = model_dir.name
    seq_lens = sorted(set().union(*(set(d.keys()) for d in data.values())))
    baseline_lens = sorted(data.get("triton", {}).keys())
    if not baseline_lens:
        print(f"  [skip] no triton data for {model_name}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    ax_sp, ax_ttft, ax_tps = axes
    fig.suptitle(f"Prefill benchmark: {model_name}", fontsize=13)

    for c in ("pto_mega", "pto", "triton"):
        if c not in data:
            continue
        sty = CASE_STYLE[c]
        xs = sorted(data[c].keys())
        ttft = np.array([float(data[c][sl]["median_ttft_ms"]) for sl in xs])
        tps = np.array([float(data[c][sl]["input_tps"]) for sl in xs])
        speedups = []
        for sl in baseline_lens:
            if sl in data[c] and sl in data["triton"]:
                t_ref = float(data["triton"][sl]["median_ttft_ms"])
                t_c = float(data[c][sl]["median_ttft_ms"])
                speedups.append(t_ref / t_c if t_c > 0 else float("nan"))
            else:
                speedups.append(float("nan"))
        lkw = {k: sty[k] for k in ("color", "label", "marker", "linestyle")}
        lkw["linewidth"] = 2; lkw["markersize"] = 5
        ax_sp.plot(np.array(baseline_lens, float), speedups, **lkw)
        ax_ttft.plot(np.array(xs, float), ttft, **lkw)
        ax_tps.plot(np.array(xs, float), tps, **lkw)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_sp.set_ylabel("Speedup vs Triton")
    ax_sp.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax_ttft.set_ylabel("Median TTFT (ms)")
    ax_tps.set_ylabel("Throughput (tokens/s)")
    ax_tps.set_xlabel("Prompt length (tokens)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prefill_{model_name}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all_prefill(prefill_root: Path, out_dir: Path) -> None:
    print(f"Plotting prefill results from {prefill_root}")
    for model_dir in sorted(prefill_root.iterdir()):
        if model_dir.is_dir():
            print(f"  Model: {model_dir.name}")
            plot_prefill_model(model_dir, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefill-dir", type=Path, default=None,
                    help="Directory with per-model JSONL files from run_prefill_sweep.sh")
    ap.add_argument("--out-dir", type=Path, default=_FIGURE_DIR)
    ap.add_argument("--auto", action="store_true",
                    help="Auto-detect latest results under outputs/data/")
    args = ap.parse_args()

    prefill_dir = args.prefill_dir
    if args.auto and prefill_dir is None:
        data_dir = _REPO_ROOT / "outputs" / "data"
        candidates = sorted(data_dir.glob("prefill_*"), key=lambda p: p.name)
        if candidates:
            prefill_dir = candidates[-1]
            print(f"Auto-detected: {prefill_dir}")

    if prefill_dir is not None and prefill_dir.is_dir():
        plot_all_prefill(prefill_dir, args.out_dir)
    else:
        print("No prefill data directory found. Run run_prefill_sweep.sh first.")
        print("Then: python scripts/plot_results.py --prefill-dir outputs/data/prefill_<stamp>")


if __name__ == "__main__":
    main()
