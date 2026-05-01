#!/usr/bin/env python3
"""Plot benchmark results: kernel speedups, vLLM prefill throughput, and accuracy.

Reads JSONL files from the prefill benchmark, JSON files from the kernel
benchmark, and eval JSON files from the lm-eval benchmark, then produces
figures in outputs/figure/.

Usage::

    # After running benchmarks/vllm_prefill/run_prefill_sweep.sh
    python scripts/plot_results.py --prefill-dir outputs/data/prefill_<stamp>

    # After running benchmarks/eval_acc/run_eval_suite.sh
    python scripts/plot_results.py --eval-dir outputs/data/eval_<stamp>

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
    # Load triton (required) and pto_mega (required); pto_staged is optional.
    required = {"triton", "pto_mega"}
    data: dict[str, dict[int, dict]] = {}
    for c in ("triton", "pto_mega", "pto"):
        p = model_dir / f"{c}.jsonl"
        if not p.is_file():
            if c in required:
                print(f"  [skip] missing {p.name} — run: "
                      f"bash benchmarks/vllm_prefill/run_prefill_sweep.sh "
                      f"  (or set CASES={c} to run only this backend)")
                return
            continue
        rows = _load_jsonl(p)
        if not rows:
            if c in required:
                print(f"  [skip] {p.name} is empty — re-run the prefill sweep")
                return
            continue
        data[c] = _by_seq_len(rows)

    model_name = model_dir.name
    baseline_lens = sorted(data["triton"].keys())

    # Only plot pto_mega vs triton; omit pto_staged regardless of file presence.
    plot_cases = ["pto_mega", "triton"]

    seq_lens = sorted(set().union(*(set(data[c].keys()) for c in plot_cases)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
    ax_sp, ax_ttft, ax_tps = axes
    fig.suptitle(f"Prefill benchmark: {model_name}", fontsize=13)

    for c in plot_cases:
        if c not in data:
            continue
        sty = CASE_STYLE[c]
        xs = sorted(data[c].keys())
        ttft = np.array([float(data[c][sl]["median_ttft_ms"]) for sl in xs])
        tps = np.array([float(data[c][sl]["input_tps"]) for sl in xs])
        speedups = [
            float(data["triton"][sl]["median_ttft_ms"]) / float(data[c][sl]["median_ttft_ms"])
            if sl in data[c] and sl in data["triton"] else float("nan")
            for sl in baseline_lens
        ]
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


_PRESET_LABELS = {
    "qwen35_0_8b":          "Qwen3.5-0.8B",
    "qwen35_9b":            "Qwen3.5-9B",
    "qwen36_27b_w8a8":      "Qwen3.6-27B-w8a8",
    "qwen36_35b_a3b_w8a8":  "Qwen3.6-35B-A3B-w8a8",
}


def plot_eval_accuracy(eval_root: Path, out_dir: Path) -> None:
    """Bar chart comparing PTO megakernel vs Triton: wikitext PPL and MMLU acc."""
    print(f"Plotting eval accuracy from {eval_root}")
    presets = [p for p in _PRESET_LABELS if (eval_root / p).is_dir()]
    backends = ["pto_mega", "triton"]

    ppl: dict[str, dict[str, float]] = {}
    mmlu: dict[str, dict[str, float]] = {}
    for pr in presets:
        ppl[pr] = {}
        mmlu[pr] = {}
        for b in backends:
            f = eval_root / pr / b / "eval.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            r = d.get("results", {})
            ppl[pr][b] = r.get("wikitext", {}).get("word_perplexity,none", float("nan"))
            mmlu_vals = [v.get("acc,none", 0.0) for k, v in r.items() if k.startswith("mmlu_")]
            mmlu[pr][b] = float(np.mean(mmlu_vals)) * 100 if mmlu_vals else float("nan")

    labels = [_PRESET_LABELS.get(p, p) for p in presets]
    x = np.arange(len(presets))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("PTO megakernel vs Triton: accuracy (lm-eval, wikitext limit=256)", fontsize=12)

    colors = {"pto_mega": CASE_STYLE["pto_mega"]["color"],
              "triton":   CASE_STYLE["triton"]["color"]}
    labels_b = {"pto_mega": "PTO megakernel", "triton": "Triton (baseline)"}

    for i, b in enumerate(backends):
        ppl_vals = [ppl[p].get(b, float("nan")) for p in presets]
        mmlu_vals = [mmlu[p].get(b, float("nan")) for p in presets]
        offset = (i - 0.5) * width
        ax1.bar(x + offset, ppl_vals, width, label=labels_b[b], color=colors[b], alpha=0.85)
        ax2.bar(x + offset, mmlu_vals, width, label=labels_b[b], color=colors[b], alpha=0.85)

    ax1.set_ylabel("Word perplexity ↓")
    ax1.set_title("Wikitext perplexity (lower = better)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("Accuracy ↑ (%)")
    ax2.set_title("MMLU accuracy (6-subject subset, higher = better)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 100)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_accuracy.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefill-dir", type=Path, default=None,
                    help="Directory with per-model JSONL files from run_prefill_sweep.sh")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Directory with eval JSON files from run_eval_suite.sh")
    ap.add_argument("--out-dir", type=Path, default=_FIGURE_DIR)
    ap.add_argument("--auto", action="store_true",
                    help="Auto-detect latest results under outputs/data/")
    args = ap.parse_args()

    prefill_dir = args.prefill_dir
    eval_dir = args.eval_dir

    if args.auto:
        data_dir = _REPO_ROOT / "outputs" / "data"
        if prefill_dir is None:
            candidates = sorted(data_dir.glob("prefill_*"), key=lambda p: p.name)
            if candidates:
                prefill_dir = candidates[-1]
                print(f"Auto-detected prefill: {prefill_dir}")
        if eval_dir is None:
            candidates = sorted(data_dir.glob("eval_*"), key=lambda p: p.name)
            if candidates:
                eval_dir = candidates[-1]
                print(f"Auto-detected eval:    {eval_dir}")

    if prefill_dir is not None and prefill_dir.is_dir():
        plot_all_prefill(prefill_dir, args.out_dir)
    elif prefill_dir is not None:
        print(f"Prefill directory not found: {prefill_dir}")
    else:
        print("No prefill data — run:  bash benchmarks/vllm_prefill/run_prefill_sweep.sh")

    if eval_dir is not None and eval_dir.is_dir():
        plot_eval_accuracy(eval_dir, args.out_dir)
    elif eval_dir is not None:
        print(f"Eval directory not found: {eval_dir}")
    else:
        print("No eval data — run:     bash benchmarks/eval_acc/run_eval_suite.sh")


if __name__ == "__main__":
    main()
