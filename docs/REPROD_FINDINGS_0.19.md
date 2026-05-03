# megagdn-pto reproduction findings

**Date:** 2026-05-02 / 2026-05-03 (UTC)  
**Environment:** `quay.io/ascend/vllm-ascend:v0.19.1rc1`, CANN 8.5.1 (`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1`), 8× Ascend 910B2 (`npu-smi 25.5.0`).  
**Repo:** `/workdir/megagdn-pto`, git submodule `third_party/pto-isa` present.

Additional runtime exports used (not all spelled out in README):

- `export PTO_LIB_PATH=/workdir/megagdn-pto/third_party/pto-isa` before vLLM workloads so worker processes reliably see PTO ISA headers (`vllm_patch/apply.py` only auto-falls back to `/sources/pto-isa` when that path exists).

---

## What ran successfully

| Step | Command / artifact | Outcome |
|------|-------------------|--------|
| Editable install | `pip install -e .` and `pip install -e '.[eval]'` | Succeeds; pip reports dependency conflict (see below). |
| vLLM hook | `python vllm_patch/install_hook.py` | Worker hook applied under `/vllm-workspace/vllm-ascend/`. Qwen file patches skipped as expected for v0.19. |
| Single-kernel tests | `tests/test_single_kernels.py --device npu:0 --H-list 16,32,48,64` | **ALL PASS** |
| Kernel benchmarks | `benchmarks/kernel/bench_gdn_kernels.py` with `--mega` for L_seg 16384 / 8192 / 4096 (outputs under `outputs/data/repro_kernel_20260502/`) | Exit 0; stage coverage matches README (e.g. Triton `solve_tril` n/a at L=16384). |
| Prefill sweep | `bash benchmarks/vllm_prefill/run_prefill_sweep.sh` with `OUT_DIR=outputs/data/repro_prefill_20260502` | **prefill_exit=0**; logs show `PTO patch active: fused megakernel (C=128)` in parent and `EngineCore` workers. |
| lm-eval suite | `bash benchmarks/eval_acc/run_eval_suite.sh` | **Partial:** 6/8 runs completed (see below). |

---

## Numeric comparison vs committed reference outputs

### Kernel JSON (L=16384, H=16)

| Metric | Committed `outputs/data/kernel_bench.json` | Repro `outputs/data/repro_kernel_20260502/kernel_bench_L16384.json` |
|--------|---------------------------------------------|----------------------------------------------------------------------|
| `mega_ms` (H=16) | ~54.79 ms | ~55.12 ms |
| `solve_tril_pto_ms` (H=16) | (same ballpark as README) | 15.89 ms |
| Triton `solve_tril` at L=16384 | n/a (grid) | n/a (grid) |

README table headline PTO H=16 values (e.g. megakernel ~54.8 ms) are reproduced within run-to-run variance.

### Prefill TTFT (triton mean TTFT / pto_mega mean TTFT; **>1 ⇒ PTO megakernel faster**)

From `outputs/data/repro_prefill_20260502/**/*.jsonl`:

| Model | Range vs README “1.07–1.25×” |
|-------|------------------------------|
| qwen35_0_8b | 1.06–1.13× |
| qwen35_9b | 1.02–1.14× (65536 not in 9B sweep per script) |
| qwen36_27b_w8a8 | 1.05–1.14× (65536 Triton skipped per script) |
| qwen36_35b_a3b_w8a8 | 0.99–1.15× (two short lengths slightly below 1×, i.e. Triton marginally faster there) |

Overall, prefill behavior matches the paper README: PTO megakernel is faster than Triton for most lengths on most models; 35B shows a few crossings near 1.0×.

### lm-eval (256-doc wikitext subset, six MMLU subjects)

Runs under `outputs/data/repro_eval_20260502/` (first reproduction) plus **35B after `run_lm_eval.py` fix** in `outputs/data/repro_eval_35b_fix/` (see below).

| Preset | Backend | WikiText word PPL ↓ | Mean MMLU acc (6 tasks) |
|--------|---------|---------------------|-------------------------|
| qwen35_0_8b | pto_mega | 19.885 | 0.493 |
| qwen35_0_8b | triton | 19.866 | 0.495 |
| qwen35_9b | pto_mega | 9.263 | 0.796 |
| qwen35_9b | triton | 9.262 | 0.795 |
| qwen36_27b_w8a8 | pto_mega | 8.201 | 0.828 |
| qwen36_27b_w8a8 | triton | 8.207 | 0.827 |
| qwen36_35b_a3b_w8a8 | pto_mega | 8.171 | 0.836 |
| qwen36_35b_a3b_w8a8 | triton | 8.166 | 0.842 |

**Earlier reproduction note:** Before the script change below, `qwen36_35b_a3b_w8a8` hit `KeyError: 'model.embed_tokens.weight'` and `run_eval_suite.sh` exited under `set -e` before the Triton backend ran.

---

## 35B lm-eval fix (2026-05-03)

**Cause:** `run_lm_eval.py` monkey-patched `AutoConfig.from_pretrained` for `model_type == qwen3_5_moe`, replacing the on-disk `Qwen3_5MoeForConditionalGeneration` config with a synthetic `Qwen3MoeConfig` built only from `text_config`. Prefill (`benchmark_prefill.py`) never applied that patch, so vLLM used the native config and Ascend `modelslim` quantization keys (e.g. `model.embed_tokens.weight`) matched. Under the synthetic config, quant lookup raised `KeyError: 'model.embed_tokens.weight'`.

**Change:** On vLLM **≥ 0.19**, skip that monkey-patch for `qwen3_5_moe` so lm-eval matches the prefill loading path.

**Verification:** Both backends completed for `qwen36_35b_a3b_w8a8` (`outputs/data/repro_eval_35b_fix/qwen36_35b_a3b_w8a8/{pto_mega,triton}/eval.json`).

---

## Problems with `README.md` / docs for a new user

1. **README path:** Instructions live at **repository root** `README.md`. There is no `megagdn_pto/README.md` (that directory is the Python package only). External references to `megagdn_pto/README.md` are misleading.

2. **`ASCEND_TOOLKIT_HOME`:** Importing `megagdn_pto` requires `ASCEND_TOOLKIT_HOME` or `ASCEND_HOME_PATH` ([`megagdn_pto/compile.py`](megagdn_pto/compile.py)). README “Environment” mentions `npu-smi` but not this requirement before `pip install -e .`.

3. **`PTO_LIB_PATH` for vLLM workers:** README does not recommend exporting `PTO_LIB_PATH` to the repo’s `third_party/pto-isa`. Without it, behavior depends on whether `/sources/pto-isa` exists in the image. Explicit export avoids opaque JIT failures in spawned workers.

4. **Hardcoded Hugging Face snapshot IDs:** [`benchmark_prefill.py`](benchmarks/vllm_prefill/benchmark_prefill.py) and [`run_lm_eval.py`](benchmarks/eval_acc/run_lm_eval.py) embed fixed `snapshots/<hash>` paths. [`scripts/download_weights.md`](scripts/download_weights.md) moves hub trees to `/scratch/model_weights/` but does **not** tell users they must match those hashes or override `--model` / edit presets. Fresh `hf download` layouts will differ.

5. **`test_e2e.py` vs README expectation:** README states that if Triton misbehaves, cross-checks are skipped and the test still passes. On this stack, **PTO vs CPU passed** but **Triton cross-check failed** for H=32 (`cross=FAIL`), so the command exits **non-zero** unless `--no-triton` is used. New users may interpret that as a broken PTO install.

6. **`pip install -e '.[eval]'`:** Pip warned that `vllm 0.19.1+empty` wants `opencv-python-headless>=4.13.0` while the image has `4.11.0.86`. README mentions possible conflicts; this specific pair appeared in practice. Eval still ran.

7. **Eval sweep fragility:** `run_eval_suite.sh` uses `set -e`; one failed preset aborts the entire suite (this hid the 35B Triton run until `run_lm_eval.py` was fixed). README does not mention this behavior or suggest continuing other backends manually after a failure.

---

## vLLM patch sanity

- `install_hook.py` applied worker hook; dry-run matched v0.19 expectations (Qwen patches skipped).
- Prefill and lm-eval logs include **`PTO patch active: fused megakernel (C=128)`** on the main process and on `EngineCore` workers.
- Workers still print **`Using Triton/FLA GDN prefill kernel`** from [`gdn_linear_attn.py`](vllm-ascend) — consistent with README “Known warnings”: the bound function is still the PTO-wrapped path after patching.

---

## Artifacts from this reproduction

| Path | Contents |
|------|----------|
| `outputs/data/repro_kernel_20260502/kernel_bench_L16384.json` | Full L=16384 profile + megakernel |
| `outputs/data/repro_kernel_20260502/kernel_bench_L8192.json` | L=8192 |
| `outputs/data/repro_kernel_20260502/kernel_bench_L4096.json` | L=4096, H∈{16,32} |
| `outputs/data/repro_prefill_20260502/` | 8× JSONL + logs (4 models × 2 backends) |
| `outputs/data/repro_eval_20260502/` | 6× `eval.json` + logs (35B missing in first run; see fix below) |
| `outputs/data/repro_eval_35b_fix/` | Full `qwen36_35b_a3b_w8a8` × 2 backends after `run_lm_eval.py` fix |

---

## Summary

Kernel micro-benchmarks and the full prefill sweep **closely match** committed `outputs/data` and README tables. lm-eval matches README for all four presets **after** fixing `qwen3_5_moe` `AutoConfig` handling on vLLM 0.19+ (see **Update** above). The largest documentation gaps for newcomers are **ASCEND env vars**, **explicit `PTO_LIB_PATH`**, **snapshot path rigidity**, **`test_e2e.py` exit code when Triton cross-check fails**, and **eval suite error handling** (`set -e` aborts the whole suite on one failure).
