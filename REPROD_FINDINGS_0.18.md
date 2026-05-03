# Reproduction findings (fresh user, README workflow)

Environment: host instructions followed from repository root **`/workdir/megagdn-pto/README.md`** (not `megagdn_pto/README.md` — that directory is the Python package only and has no README). Container: `quay.io/ascend/vllm-ascend:v0.18.0rc1`. Submodule `third_party/pto-isa` was already initialized. NPU devices reported healthy via `npu-smi info`. Model weights were present under **`/scratch/model_weights/`** including the Hugging Face snapshot hashes hardcoded in the benchmark presets.

Commands executed **without edits** to existing scripts:

| README section | Command / script | Outcome |
|----------------|------------------|---------|
| § Dependencies | `pip install -e .` and `pip install -e '.[eval]'` | Succeeded (`exit_code: 0`). Pip reported **`opencv-python-headless` conflict** with vLLM’s pin — matches the README dependency note; eval still ran successfully. |
| §3 Patch | `python vllm_patch/install_hook.py` | Succeeded; patched **`/vllm-workspace/vllm-ascend/...`** in-container paths. |
| §2 Kernel bench | `GDN_NPU_DEVICE=npu:0 python benchmarks/kernel/bench_gdn_kernels.py ... --output-json outputs/data/kernel_bench_repro.json` | Completed (~5 min wall time). Timing within a few percent of committed **`outputs/data/kernel_bench.json`** (e.g. H=16 megakernel **55.43 ms** vs reference **54.56 ms**). |
| §4 Prefill | `bash benchmarks/vllm_prefill/run_prefill_sweep.sh` | **Did not finish.** |
| §5 Accuracy | `bash benchmarks/eval_acc/run_eval_suite.sh` | Completed successfully (~**2.1 h**). All eight runs (**4 presets × `pto_mega` / `triton`**) wrote **`eval.json`** under **`outputs/data/eval_20260502_115558/`**. |

Unit tests (**README §1**) were **not** re-run in this pass (focus was kernel + long vLLM jobs); §2 kernel suite already exercised PTO/Triton kernels on-device.

---

## Prefill sweep failure (critical)

After completing **`qwen35_0_8b`** and **`qwen35_9b`** fully, and **`qwen36_27b_w8a8`** `pto_mega`, the sweep died on **`qwen36_27b_w8a8`** with **`case=triton`** when moving past **32768** tokens toward **`65536`**.

Symptoms:

- **`RuntimeError: ACL stream synchronize failed, error code: 507014`**
- Underlying **`aicore timeout`** / **`EZ9999`** device error inside the stock **Triton FLA** path (`chunk_fwd_o` / `prepare_chunk_offsets` stack as in the benchmark log).

**`qwen36_35b_a3b_w8a8` was never reached** because **`set -e`** stops the orchestration script on the failed Python invocation.

Artifacts from this incomplete run:

- **`outputs/data/prefill_20260502_111445/`** — **`pto_mega` JSONL completed with 8 sequence lengths per finished model**, including **`65536`** where the sweep got that far (`qwen36_27b_w8a8` `pto_mega` includes **65536**).
- **`qwen36_27b_w8a8/triton.jsonl`** is **truncated** (7 lines — through **32768** only).

So a “full README” end-to-end prefill reproduction **does not currently complete** on this hardware profile for the default sweep parameters.

---

## Mismatch between README sweep defaults and archived `outputs/`

Default sequence lengths in **`benchmarks/vllm_prefill/run_prefill_sweep.sh`** are:

```bash
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096 8192 16384 32768 65536}"
```

The checked-in reference directory **`outputs/data/prefill_20260501_182705/**/*.jsonl` contains seven records per file and no `65536` anywhere** (grep / line counts confirm). So committed prefill curves **end at `32768`**, while naive following of the README runs **nine** nominal lengths ending at **`65536`**.

Recommendation for documentation: align script defaults with the published artifact **or** state explicitly that **`65536` is optional** and that Triton may **timeout / fail** on large prompts for quantized 27B on some NPUs — and mention **`SEQ_LENS=...`** to match the **`outputs`** layout.

---

## lm-eval parity vs **`outputs/data/eval_20260501_203237/`**

WikiText perplexity (**256-doc subset**) and mean of the six MMLU subset accuracies were **consistent** with the reference JSON (differences negligible or single-task noise, e.g. high-school math one step on 0.8B **`pto_mega`**). Overall this matches the README claim that **PTO megakernel and Triton are numerically comparable** under these evaluation settings.

---

## Other documentation friction for newcomers

1. **Weight paths**: README table + **`benchmark_prefill.py` / `run_lm_eval.py`** embed **specific snapshot subdirectory names** (`models--Qwen--Qwen3.5-…/snapshots/<hash>/`). **`scripts/download_weights.md`** describes downloads but does **not** guarantee matching hashes — if the snapshot folder name differs locally, presets must be overridden with **`--model /path`** (mentioned briefly in **`benchmark_prefill.py` docstring, not prominently in README table).

2. **Eval example paths**: **`README §5`** shows **`--output-json outputs/data/eval/...`**, while **`run_eval_suite.sh`** uses timestamped **`outputs/data/eval_<stamp>/`** trees. Harmless once discovered, easy to confuse when looking for files.

3. **`pip install -e '.[eval]'`**: Dependency churn vs the Docker image is real; **`--no-deps`** escape hatch in README is useful (observed benign conflict on **`opencv-python-headless`**).

---

## Bottom line

- **Kernel benchmarks** and **lm-eval suite** reproduced **in line** with **`outputs/`** (new files: **`outputs/data/kernel_bench_repro.json`**, **`outputs/data/eval_20260502_115558/`**).
- **Full default prefill sweep** as written **failed mid-run** due to **Triton + very long-context** behaviour on **`Qwen3.6-27B-w8a8`**, and did not run MoE checkpoints.
- The **published prefill JSONL** does **not** include **`65536`**, contrary to README default **`SEQ_LENS`** — tighten README or defaults to reduce surprise.

Generated: **2026-05-02** (evaluation run timestamp in paths above).
