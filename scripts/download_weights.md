# Model Weight Download Instructions

The benchmarks and accuracy evaluations use four pre-downloaded models.
Weights are stored under `/scratch/model_weights/` (adjust the path for your system).

## Qwen3.5 Models (standard precision, bf16)

```bash
# Install Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash
export PATH=$HOME/.local/bin:$PATH

# Qwen3.5-0.8B  (~1.6 GB)
hf download Qwen/Qwen3.5-0.8B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B /scratch/model_weights/

# Qwen3.5-9B  (~19 GB)
hf download Qwen/Qwen3.5-9B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B /scratch/model_weights/
```

> **Note:** Qwen3.5-9B uses GQA (`linear_num_key_heads ≠ linear_num_value_heads`),
> which exercises the PTO group-value (GQA) kernel path. Models 0.8B and below
> use MHA (equal Q/K/V heads).

## Qwen3.6 Models (W8A8 quantized, requires modelscope)

The 27B and 35B-A3B models are quantized to 8-bit weights / 8-bit activations
(msmodelslim format) and require `--quantization ascend` in vLLM.

```bash
pip install modelscope

# Qwen3.6-27B-w8a8  (~28 GB)
modelscope download \
    --model Eco-Tech/Qwen3.6-27B-w8a8 \
    --local_dir /scratch/model_weights/Qwen3.6-27B-w8a8

# Qwen3.6-35B-A3B-w8a8  (MoE architecture, ~38 GB)
modelscope download \
    --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
    --local_dir /scratch/model_weights/Qwen3.6-35B-A3B-w8a8
```

## Expected directory layout after download

```
/scratch/model_weights/
├── models--Qwen--Qwen3.5-0.8B/snapshots/<sha>/   ← qwen35_0_8b preset
├── models--Qwen--Qwen3.5-9B/snapshots/<sha>/      ← qwen35_9b preset
├── Qwen3.6-27B-w8a8/                              ← qwen36_27b_w8a8 preset
└── Qwen3.6-35B-A3B-w8a8/                          ← qwen36_35b_a3b_w8a8 preset
```

These paths match the `_MODEL_PRESETS` dictionaries in:
- `benchmarks/vllm_prefill/benchmark_prefill.py`
- `benchmarks/eval_acc/run_lm_eval.py`
- `profiling/profile_prefill.py`
