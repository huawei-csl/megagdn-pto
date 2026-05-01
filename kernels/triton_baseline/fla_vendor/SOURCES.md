# Vendored Triton sources

The Python modules in this directory are **verbatim copies** (aside from import path fixes noted below) of the vLLM-Ascend FLA Triton ops:

**Upstream:** [vllm-project/vllm-ascend `v0.18.0rc1` — `vllm_ascend/ops/triton/fla`](https://github.com/vllm-project/vllm-ascend/tree/v0.18.0rc1/vllm_ascend/ops/triton/fla)

Also vendored (same tag) from [`vllm_ascend/ops/triton/triton_utils.py`](https://github.com/vllm-project/vllm-ascend/blob/v0.18.0rc1/vllm_ascend/ops/triton/triton_utils.py) as `ascend_triton_utils.py`, imported by `solve_tril.py` so this example does not depend on the `vllm_ascend.ops` package layout.

Runtime still expects **`from vllm.triton_utils import tl, triton`** (vLLM’s Triton bindings for Ascend).

**Local edits:** `solve_tril.py` — `extract_slice` / `insert_slice` import from `.ascend_triton_utils` instead of `vllm_ascend.ops.triton.triton_utils`.
