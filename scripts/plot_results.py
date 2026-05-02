#!/usr/bin/env python3
"""Plot benchmark results: kernel speedups, vLLM prefill throughput, and accuracy.

Reads JSONL files from the prefill benchmark, JSON files from the kernel
benchmark, and eval JSON files from the lm-eval benchmark, then produces
figures under ``outputs/figure/`` as **PNG**.

Usage::

    # After running benchmarks/vllm_prefill/run_prefill_sweep.sh
    python scripts/plot_results.py --prefill-dir outputs/data/prefill_<stamp>

    # After running benchmarks/eval_acc/run_eval_suite.sh
    python scripts/plot_results.py --eval-dir outputs/data/eval_<stamp>

    # Kernel micro-benchmark stacked bars (PTO vs Triton BT=64/BT=128 per stage)
    python scripts/plot_results.py \\
        --kernel-stage-json outputs/data/kernel_bench_L8192.json --kernel-stage-h 16

    # PTO megakernel vs chained staged pipelines (needs bench_pto_pipeline_latency.py JSON)
    python scripts/plot_results.py \\
        --pto-pipeline-json outputs/data/pto_pipeline_latency.json

    # Plot all available data (prefill/eval via --auto paths; omit kernel unless flags set)
    python scripts/plot_results.py --auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIGURE_DIR = _REPO_ROOT / "outputs" / "figure"
_FIGURE_DPI = 150


def _plot_keep_prompt_len(model_name: str, seq_len: int) -> bool:
    """Drop outliers / partial sweep points from charts only."""
    if model_name == "qwen36_27b_w8a8" and seq_len == 65536:
        return False
    return True


CASE_STYLE = {
    "pto_mega": {"color": "#1f77b4", "label": "PTO megakernel", "marker": "o", "linestyle": "-"},
    "pto":      {"color": "#ff7f0e", "label": "PTO staged",     "marker": "s", "linestyle": "-"},
    "triton":   {"color": "#444444", "label": "Triton (baseline)", "marker": "^", "linestyle": "--"},
}

# Matches ``bench_gdn_kernels.py --output-json`` (PTO per-stage ms + Triton BT=64/128).
KERNEL_STAGE_FIELDS: list[tuple[str, str, str, str]] = [
    ("chunk_cumsum", "cumsum_pto_ms", "cumsum_triton64_ms", "cumsum_triton128_ms"),
    ("scaled_dot_kkt", "kkt_pto_ms", "kkt_triton64_ms", "kkt_triton128_ms"),
    ("solve_tril", "solve_tril_pto_ms", "solve_tril_triton64_ms", "solve_tril_triton128_ms"),
    ("wy_fast", "wy_fast_pto_ms", "wy_fast_triton64_ms", "wy_fast_triton128_ms"),
    ("chunk_h", "chunk_h_pto_ms", "chunk_h_triton64_ms", "chunk_h_triton128_ms"),
    ("chunk_o", "chunk_o_pto_ms", "chunk_o_triton64_ms", "chunk_o_triton128_ms"),
]
KERNEL_STAGE_COLORS = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f"]

PIPELINE_COLORS = {
    "mega": "#08519c",
    "separate": "#31a354",
    "triton": "#444444",
}


def _stage_label(name: str) -> str:
    if name == "scaled_dot_kkt":
        return "scaled dot KKT"
    return name.replace("_", " ")


def plot_kernel_stage_stacked(
    kernel_json_path: Path,
    h_value: int,
    out_dir: Path,
    outfile: str | None = None,
) -> Path | None:
    """Stacked latency: PTO per-stage vs Triton BT=64 vs Triton BT=128.

    Columns are **not** summed micro-times for Triton staging — each column uses the
    per-stage NPU timings produced by ``bench_gdn_kernels.py`` (device/kernel time).
    Torch eager/driver launch overhead is amortized across the benchmark repetitions.

    Where a Triton BT column has no measurement for a stage (``null`` / missing),
    a hatched slice (height derived from matching PTO stage time) marks failure / not
    applicable — same hatch style for BT=64 and BT=128 independently.
    """
    if not kernel_json_path.is_file():
        print(f"  [skip] kernel JSON missing: {kernel_json_path}")
        return None

    blob = json.loads(kernel_json_path.read_text())
    rows_raw = blob.get("results")
    if not isinstance(rows_raw, list):
        print(f"  [skip] malformed kernel JSON (no results list): {kernel_json_path}")
        return None

    row = next((r for r in rows_raw if int(r.get("H", -1)) == int(h_value)), None)
    if row is None:
        print(f"  [skip] no H={h_value} entry in {kernel_json_path}")
        return None

    n_seq = int(blob.get("N_seq", row.get("N_seq", 16)))
    l_seg = int(blob.get("L_seg", row.get("L_seg", 0)))
    hg = int(row.get("Hg", 16))

    xt_pto, xt_t64, xt_t128 = (0.0, 1.0, 2.0)
    width = 0.55
    labels = [
        "PTO",
        "Triton (BT=64)",
        "Triton (BT=128)",
    ]

    fig, ax = plt.subplots(figsize=(11.0, 7.2))

    def _column_stack_triton(x_center: float, mode: str) -> tuple[float, bool]:
        """BT=64 / BT=128 stacks: only measured Triton time (or fail hatch scaled to PTO)."""
        bottom = 0.0
        has_gap = False
        for (_stage_name, pk, tk64, tk128), col in zip(KERNEL_STAGE_FIELDS, KERNEL_STAGE_COLORS):
            tk_name = tk64 if mode == "t64" else tk128
            pv = float(row[pk])

            tr_raw = row.get(tk_name)
            if tr_raw is None:
                has_gap = True
                fail_seg = max(2.25, pv * 0.19)
                light = tuple(np.clip(np.array(mcolors.to_rgb(col)) * 0.58 + 0.42, 0, 1))
                ax.bar(
                    x_center,
                    fail_seg,
                    width,
                    bottom=bottom,
                    facecolor=light,
                    hatch="////",
                    edgecolor="crimson",
                    linewidth=2.0,
                    label="_nolegend_",
                )
                ax.text(
                    x_center,
                    bottom + fail_seg * 0.5,
                    "fail",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="darkred",
                    fontweight="bold",
                    rotation=90,
                )
                bottom += fail_seg
            else:
                tv = float(tr_raw)
                ax.bar(
                    x_center,
                    tv,
                    width,
                    bottom=bottom,
                    facecolor=col,
                    edgecolor="#1a1a1a",
                    linewidth=0.45,
                    label="_nolegend_",
                )
                bottom += tv

        return bottom, has_gap

    bottoms_pto = 0.0
    for (stage, pk, tk64, tk128), col in zip(KERNEL_STAGE_FIELDS, KERNEL_STAGE_COLORS):
        pv = float(row[pk])
        ax.bar(
            xt_pto,
            pv,
            width,
            bottom=bottoms_pto,
            facecolor=col,
            edgecolor="#1a1a1a",
            linewidth=0.45,
            label=_stage_label(stage),
        )
        bottoms_pto += pv

    _, gap64 = _column_stack_triton(xt_t64, "t64")
    _, gap128 = _column_stack_triton(xt_t128, "t128")
    has_any_triton_gap = gap64 or gap128

    ax.set_xticks([xt_pto, xt_t64, xt_t128])
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(
        f"Per-stage kernel time (micro-benchmark) — device / kernel only; "
        f"PyTorch eager launch overhead amortized across iters.\n"
        f"N={n_seq} seqs × L={l_seg} tokens, H={h_value}, Hg={hg}, "
        f"D={int(row.get('D', blob.get('D', 128)))}, "
        f"C_PTO={int(row.get('C_pto', blob.get('C_pto', 128)))}",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)

    handles, leg_labels = ax.get_legend_handles_labels()
    merged: list[tuple] = []
    seen: set[str] = set()
    for hdl, lb in zip(handles, leg_labels):
        if lb.startswith("_") or lb in seen:
            continue
        seen.add(lb)
        merged.append((hdl, lb))
    extra: list[Patch] = []
    if has_any_triton_gap:
        extra.append(
            Patch(
                facecolor="#f0f0f0",
                edgecolor="crimson",
                hatch="////",
                linewidth=2,
                label="Triton stage failed / n/a",
            )
        )
    if extra:
        ax.legend(
            [m[0] for m in merged] + extra,
            [m[1] for m in merged] + [p.get_label() for p in extra],
            loc="upper left",
            fontsize=10,
            ncol=2,
            framealpha=0.92,
        )
    else:
        ax.legend(
            [m[0] for m in merged],
            [m[1] for m in merged],
            loc="upper left",
            fontsize=10,
            ncol=2,
            framealpha=0.92,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if outfile:
        out_name = outfile
    else:
        out_name = f"kernel_stages_N{n_seq}_L{l_seg}_H{h_value}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_pto_pipeline_benefits(
    pipeline_json_path: Path,
    out_dir: Path,
    outfile: str | None = None,
) -> Path | None:
    """Horizontal bars: PTO megakernel vs staged PTO; optional per-row Triton e2e (BT=64).

    JSON from ``bench_pto_pipeline_latency.py`` — host ``time.perf_counter`` means (see ``bench_timer``).
    """
    def _row_separate_ms(r: dict) -> float:
        if "separate_ms" in r:
            return float(r["separate_ms"])
        if "staged_nosync_ms" in r:
            return float(r["staged_nosync_ms"])
        raise KeyError("expected separate_ms or staged_nosync_ms in pipeline JSON row")

    if not pipeline_json_path.is_file():
        print(f"  [skip] PTO pipeline JSON missing: {pipeline_json_path}")
        return None

    blob = json.loads(pipeline_json_path.read_text())
    rows_raw = blob.get("results")
    if not isinstance(rows_raw, list) or not rows_raw:
        print(f"  [skip] malformed or empty pipeline JSON: {pipeline_json_path}")
        return None

    rows = sorted(
        rows_raw,
        key=lambda r: (int(r.get("L_seg", 0)), int(r.get("H", 0))),
    )

    n_seq = int(blob.get("N_seq", rows[0].get("N_seq", 16)))
    y_labels = [
        (
            rf"$L_\mathrm{{seg}}={int(r['L_seg'])//1024}\mathrm{{k}}$, "
            rf"$H={int(r['H'])}$"
        )
        for r in rows
    ]

    mega = [float(r["mega_ms"]) for r in rows]
    sep = [_row_separate_ms(r) for r in rows]
    triton_ms: list[float | None] = []
    for r in rows:
        raw = r.get("triton_bt64_chain_ms")
        triton_ms.append(float(raw) if raw is not None else None)

    n = len(rows)
    fig_h = max(10.5, n * 0.58 + 1.9)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))

    indices = np.arange(n, dtype=float)
    dy = 0.205
    hbar = dy * 0.86

    y_mega = indices + dy
    y_sep = indices
    y_tri = indices - dy

    ax.barh(
        y_mega,
        mega,
        hbar,
        label="PTO megakernel",
        color=PIPELINE_COLORS["mega"],
        edgecolor="#1a3a62",
        linewidth=0.5,
        zorder=3,
    )
    ax.barh(
        y_sep,
        sep,
        hbar,
        label="PTO separate",
        color=PIPELINE_COLORS["separate"],
        edgecolor="#246b30",
        linewidth=0.5,
        zorder=2,
    )

    triton_labeled = False
    any_triton = False
    for i, tm in enumerate(triton_ms):
        if tm is None:
            continue
        any_triton = True
        lbl = "Triton chained (BT=64)" if not triton_labeled else "_nolegend_"
        triton_labeled = True
        ax.barh(
            y_tri[i],
            tm,
            hbar,
            label=lbl,
            color=PIPELINE_COLORS["triton"],
            edgecolor="#2a2a2a",
            linewidth=0.5,
            zorder=2,
        )

    for_vals = [*mega, *sep, *(t for t in triton_ms if t is not None)]
    xmax = float(max(for_vals)) * 1.12 if for_vals else 1.0
    xmax = max(xmax, 1e-6)
    ax.set_xlim(0.0, xmax)
    ax.set_yticks(indices)
    ax.set_yticklabels(y_labels, fontsize=9)

    hg = int(rows[0].get("Hg", 16))
    cpto = int(rows[0].get("C_pto", blob.get("C_pto", 128)))
    stamp = blob.get("timestamp")
    dev = blob.get("device")

    title2 = (
        "Wall-clock latency (mean ``time.perf_counter()`` per ``fn()`` on host): "
        "``torch.npu.synchronize()`` before/after measured region + between trials idle queue — "
        "includes PyTorch eager / interpreter cost (multi-call **PTO separate** vs one-call **mega**)."
    )
    if stamp:
        title2 += f" — {stamp}"
    tri_txt = "; Triton: six kernels, chunk size 64, where launches succeed" if any_triton else ""
    ax.set_title(
        "PTO megakernel vs staged kernels vs optional Triton e2e chain\n"
        + title2,
        fontsize=11,
        pad=10,
    )
    footer = f"N_seq={n_seq}, Hg={hg}, C_PTO={cpto}{tri_txt}"
    if dev:
        footer += f", device {dev}"
    if blob.get("bench_timing_between_iters"):
        footer += " Timing: host perf_counter ± NPU sync boundaries (idle between trials)."
    footer += "."

    ax.text(
        0.0,
        -0.06,
        footer,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        alpha=0.9,
    )
    ax.set_xlabel("Latency (ms)")
    ax.grid(axis="x", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9.0, framealpha=0.95)

    out_dir.mkdir(parents=True, exist_ok=True)
    if outfile:
        out_name = outfile
    else:
        out_name = "pto_pipeline_megakernel_vs_staged.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


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
    for ck in list(data.keys()):
        data[ck] = {
            sl: row for sl, row in data[ck].items()
            if _plot_keep_prompt_len(model_name, sl)
        }
    baseline_lens = sorted(data["triton"].keys())
    if not baseline_lens:
        print(f"  [skip] filtered Triton baseline empty for {model_name}")
        return

    # Only plot pto_mega vs triton; omit pto_staged regardless of file presence.
    plot_cases = ["pto_mega", "triton"]

    seq_lens = sorted(set().union(*(set(data[c].keys()) for c in plot_cases)))
    if not seq_lens:
        print(f"  [skip] no sequence lengths left after filters for {model_name}")
        return

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
        speedups = []
        for sl in baseline_lens:
            if sl in data[c] and sl in data["triton"]:
                num = float(data["triton"][sl]["median_ttft_ms"])
                den = float(data[c][sl]["median_ttft_ms"])
                speedups.append(num / den if den > 0 else float("nan"))
            else:
                speedups.append(float("nan"))
        lkw = {k: sty[k] for k in ("color", "label", "marker", "linestyle")}
        lkw["linewidth"] = 2; lkw["markersize"] = 5
        ax_sp.plot(np.asarray(baseline_lens, dtype=float), np.asarray(speedups, dtype=float), **lkw)
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
    out_path = out_dir / f"prefill_{model_name}.png"
    fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
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
    out_path = out_dir / "eval_accuracy.png"
    fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
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
    ap.add_argument("--kernel-stage-json", type=Path, default=None,
                    help="JSON from benchmarks/kernel/bench_gdn_kernels.py (--output-json)")
    ap.add_argument("--kernel-stage-h", type=int, default=16,
                    help="Value-head count H row to plot from the kernel JSON")
    ap.add_argument("--kernel-stage-out", type=str, default=None,
                    help="Output PNG basename (default: kernel_stages_N{L}_H{H}.png)")
    ap.add_argument("--pto-pipeline-json", type=Path, default=None,
                    help="JSON from benchmarks/kernel/bench_pto_pipeline_latency.py "
                         "(megakernel vs staged pipeline)")
    ap.add_argument("--pto-pipeline-out", type=str, default=None,
                    help="Output PNG basename (default: pto_pipeline_megakernel_vs_staged.png)")
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

    if eval_dir is not None and eval_dir.is_dir():
        plot_eval_accuracy(eval_dir, args.out_dir)
    elif eval_dir is not None:
        print(f"Eval directory not found: {eval_dir}")

    if args.kernel_stage_json is not None:
        k_path = args.kernel_stage_json
        if not k_path.is_absolute():
            k_path = (_REPO_ROOT / k_path).resolve()
        print(f"Kernel stacked stages from {k_path} (H={args.kernel_stage_h})")
        plot_kernel_stage_stacked(
            k_path, args.kernel_stage_h, args.out_dir, outfile=args.kernel_stage_out,
        )

    if args.pto_pipeline_json is not None:
        p_path = args.pto_pipeline_json
        if not p_path.is_absolute():
            p_path = (_REPO_ROOT / p_path).resolve()
        print(f"PTO pipeline latency from {p_path}")
        plot_pto_pipeline_benefits(
            p_path, args.out_dir, outfile=args.pto_pipeline_out,
        )


if __name__ == "__main__":
    main()
