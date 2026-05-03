# megagdn-pto README reproduction findings

This report documents a full pass following [README.md](README.md) on **Ascend 910B2** (device 0), **vLLM 0.18.0** / **vllm-ascend** as in `quay.io/ascend/vllm-ascend:v0.18.0rc1`, with `third_party/pto-isa` initialized. No application or benchmark **scripts** in this repository were modified; the environment was adjusted only where necessary (pip packages) to complete **lm-eval**.

## Environment fingerprint

| Item | Value |
|------|--------|
| NPU | 910B2 (npu-smi 25.5.1), `ASCEND_RT_VISIBLE_DEVICES=0` |
| `ASCEND_TOOLKIT_HOME` | `/usr/local/Ascend/cann-8.5.1` |
| vLLM | 0.18.0 |
| PTO install hook | `python vllm_patch/install_hook.py` → patched `/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/` (worker + Qwen patches) |
| Model weights | Present at README preset paths under `/scratch/model_weights/` |

## Commands run (approx. wall time)

| Stage | Outcome | Notes |
|-------|---------|--------|
| `pip install -e .` then `pip install -e '.[eval]'` | Success | Resolver warned: `fsspec` pinned older (datasets); `opencv-python-headless` below vLLM’s stated floor; harmless for these runs |
| `python vllm_patch/install_hook.py` | Success | Required for PTO in **spawned** worker processes |
| `tests/test_single_kernels.py --H-list 16,32,48,64` | **ALL PASS** | ~4 min JIT + tests |
| `tests/test_e2e.py` (default, Triton cross-check **on**) | **FAIL** | All cases: `pto_vs_cpu=PASS`, **Triton cross-check FAIL** at H=32, Hg=16 |
| `tests/test_e2e.py --no-triton` | **ALL PASS** | README-optional path |
| Kernel benches ×3 (`--mega`) | Success | Wrote `outputs/data/repro_kernel_bench_L{16384,8192,4096}.json` |
| `bash benchmarks/vllm_prefill/run_prefill_sweep.sh` | Success | `OUT_DIR=outputs/data/prefill_repro_20260502`, **~40 min** |
| `bash benchmarks/eval_acc/run_eval_suite.sh` | **Failed** on stock `transformers` | See below; **succeeded** after upgrade, **~125 min** |

## vLLM PTO patch verification

- Logs show **`PTO patch active: fused megakernel (C=128).`** on the parent process and on **`EngineCore`** workers (e.g. prefill and eval logs), which indicates `install_hook.py` + `VLLM_PTO_*` env are effective under multiprocessing.
- README’s note about **`Using Triton/FLA GDN prefill kernel`** still appears on workers; with the hook active, the swapped `chunk_gated_delta_rule` is what runs for GDN prefill.
- Prefill TTFT: `median_ttft_ms` for **Triton / PTO megakernel** is **> 1** for every model and every shared sequence length (i.e. PTO faster), consistent with the README’s **~1.07–1.25×** headline:

| Model | min (triton/pto) | max (triton/pto) | ratio at longest common seq |
|------|------------------|------------------|-----------------------------|
| qwen35_0_8b | 1.083 | 1.266 | 1.083 @ 65536 |
| qwen35_9b | 1.068 | 1.204 | 1.068 @ 32768 |
| qwen36_27b_w8a8 | 1.078 | 1.234 | 1.078 @ 32768 (longest **shared** length; Triton skips 65536 per script; PTO still measures 65536) |
| qwen36_35b_a3b_w8a8 | 1.108 | 1.293 | 1.108 @ 65536 |

## Kernel benchmark vs committed `outputs/data`

- Reproduced the **same qualitative structure** (Triton `solve_tril` n/a at large L, known skips, megakernel speedup vs staged PTO).
- **H=16, L=16384** — PTO stage ms and **megakernel** are very close to [outputs/data/kernel_bench.json](outputs/data/kernel_bench.json):

| Metric | Reference JSON | This run (`repro_kernel_bench_L16384.json`) |
|--------|------------------|---------------------------------------------|
| `mega_ms` (H=16) | 54.79 | 54.74 |
| Per-stage PTO ms | README table ~0.32 / 4.67 / 15.89 / 6.97 / 10.12 / 11.12 | Matches within a few % |

- Smaller L and/or more compile noise can widen gaps (e.g. some L8192 stage ms differ more); still same order of magnitude as reference.

## lm-eval vs README accuracy table

`pip install -e '.[eval]'` alone was **not** sufficient: **lm-eval** calls `AutoConfig.from_pretrained` with **`model_type: qwen3_5`**, which **fails on `transformers==4.57.6`** (newest 4.x on PyPI at test time: no `qwen3_5` in the mapping). **Upgrading to `transformers==5.7.0`** fixed lm-eval loading; this **violates** vLLM 0.18.0’s declared `transformers<5` constraint in pip metadata. In this container, vLLM **still ran** evaluations successfully after the upgrade, but **README should warn** that a supported Transformers major line for Qwen3.5 registry may require overriding the image pin.

Subset results (`WIKITEXT_LIMIT=256`, six MMLU subjects from `run_lm_eval.py`), **PTO megakernel vs Triton**:

| Preset | WikiText word PPL (pto / triton) | Mean MMLU acc, 6 tasks (pto / triton) |
|--------|----------------------------------|---------------------------------------|
| qwen35_0_8b | 19.89 / 19.87 | 0.495 / 0.495 |
| qwen35_9b | 9.26 / 9.26 | 0.797 / 0.795 |
| qwen36_27b_w8a8 | 8.20 / 8.21 | 0.829 / 0.827 |
| qwen36_35b_a3b_w8a8 | 8.17 / 8.17 | 0.836 / 0.842 |

These align with the README table within **rounding / task sampling noise**; 35B MMLU shows the same slight **Triton-higher** pattern as the published 83.6% vs 84.2% split.

Outputs: `outputs/data/eval_repro_20260502_v2/<preset>/{pto_mega,triton}/eval.json` and `eval_repro_20260502_v2_master.log`.

## README / instruction gaps (confirmed)

1. **§4 Prefill and §5 lm-eval must follow §3 `install_hook.py`.** README does not say this explicitly; without the hook, parent-only `apply_pto_patch()` is insufficient with **`VLLM_WORKER_MULTIPROC_METHOD=spawn`**.
2. **`tests/test_e2e.py` default does not always reach “ALL PASS”.** Here, **Triton vs PTO cross-check** failed while **CPU reference** passed; `--no-triton` matches README’s optional path. Consider documenting stricter environment expectations or tolerances.
3. **Stock `transformers` in the reference image does not register `qwen3_5` for lm-eval’s `AutoConfig` path**, so **§5 can fail out of the box** until Transformers is upgraded (or another supported integration is documented)—conflicting with vLLM’s pip metadata (`transformers<5`).
4. **`pip install -e '.[eval]'`** can downgrade **`fsspec`** and trigger **resolver warnings** (README mentions conflicts; also expect **`opencv-python-headless`** warnings vs vLLM).
5. **Quick-start** stresses **`GDN_NPU_DEVICE`** for kernel tests and **`ASCEND_RT_VISIBLE_DEVICES`** for vLLM; easy to set only one.
6. **README Environment** could mention **`ASCEND_TOOLKIT_HOME`** (required at `import megagdn_pto` via [megagdn_pto/compile.py](megagdn_pto/compile.py)); often set in the image but not in the doc list.
7. **Hardcoded Hugging Face snapshot directories** in presets: weights must match or be symlinked; **download_weights.md** does not spell out SHA alignment without editing Python.

## Artifacts from this run

| Artifact |
|----------|
| `outputs/data/repro_kernel_bench_L16384.json` |
| `outputs/data/repro_kernel_bench_L8192.json` |
| `outputs/data/repro_kernel_bench_L4096.json` |
| `outputs/data/prefill_repro_20260502/` (+ `prefill_repro_20260502_master.log`) |
| `outputs/data/eval_repro_20260502_v2/` (+ `eval_repro_20260502_v2_master.log`); first attempt failed immediately (Transformers `qwen3_5`) with logs under `eval_repro_20260502/` |

## Summary

Following the README end-to-end: **single-kernel tests pass**, **kernel micro-benchmarks match reference `outputs/data` closely**, **prefill sweep shows consistent TTFT speedups with the PTO patch active on workers**, and **lm-eval reproduces the README accuracy story** after upgrading **Transformers** to a version that knows **`qwen3_5`**. The main friction for a “new user” is the **lm-eval / Transformers / vLLM pin triangle**, plus **`install_hook.py` ordering** and optional **`test_e2e.py` Triton cross-check** strictness.
