<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

This private research fork extends
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) with
fixed and batch-size-aware dynamic speculative decoding.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Speculative Decoding Features

* **Fixed speculative decoding** with a separate Draft model and configurable `K`.
* **Dynamic K scheduling** based on the active decode batch size, including `K=0` target-only steps.
* **CUDA Graph verification** captured per `(K, batch bucket)` with a shared graph memory pool.
* **Optional K=0 Draft skip** with automatic Draft KV catch-up when speculation resumes.
* **Heterogeneous-request benchmarks** for observing in-process `K=0 -> K=3 -> K=5` transitions.
* **Profiling and experiment scripts** for throughput, acceptance rate, TTFT, TPOT, and Nsight Systems runs.

See [the dynamic speculative decoding report](docs/DYNAMIC_SPECULATIVE_DECODING_REPORT.md)
for the control flow and [DEPENDENCIES.md](DEPENDENCIES.md) for the tested environment.

## Quantization

This fork implements **FP8 W8A8 inference for Qwen3**:

* E4M3 weights with `128 x 128` block-wise scales, loaded directly from Hugging Face FP8 checkpoints.
* Dynamic E4M3 activation quantization in 128-element blocks.
* A fused Triton FP8 matrix-multiplication path for the QKV, attention output, and MLP projections.
* Automatic FP8 checkpoint detection, or explicit `quantization="fp8"` selection.
* A CPU dequantization fallback for correctness tests. Embeddings, normalization, and the LM head keep their checkpoint dtype.

Current limits: tensor parallelism, static activation scales, and non-E4M3/block-128 checkpoints are rejected. Unit tests and packed QKV/MLP checkpoint loading pass; GPU output correctness and performance still need validation on an active CUDA host.

```python
llm = LLM(
    "/YOUR/QWEN3-FP8-MODEL/PATH",
    quantization="fp8",
    tensor_parallel_size=1,
)
```

It also implements **INT4 weight-only (W4A16) inference for AWQ and GPTQ checkpoints**,
adapted from [Brassinai/nano-vllm@e909386a](https://github.com/Brassinai/nano-vllm/tree/e909386a729e5d20113e0b67f7639d5f1b9a6c37/nanovllm/quantization):

* AutoAWQ GEMM checkpoints (4-bit, zero point). The interleaved AWQ nibble order is converted to natural order once at load time.
* Reference nano-vLLM AWQ checkpoints with natural column packing and `zero - 1` storage, selected explicitly with `awq_checkpoint_format="nano-vllm"` (or that value in the checkpoint's `checkpoint_format` metadata). These decode zeros as `unpack(qzeros) + 1`, matching the reference; AutoAWQ decodes zeros without that offset.
* GPTQ and GPTQ v2 checkpoints (4-bit, `desc_act=False`), with the legacy `zero - 1` convention handled.
* `group_size` of any positive value or `-1`.
* One Triton kernel that unpacks INT4 weights and runs an FP16 matmul with FP32 accumulation; activations, the KV cache, embeddings, normalization, and the LM head stay in float16.

AWQ models support tensor parallelism when attention heads, packed output channels,
and row-parallel input shards divide evenly across ranks; row shards must also align
with the AWQ group size. GPTQ models still require `tensor_parallel_size=1`.
Both formats run without speculative decoding.
AWQ TP=2/4 weight loading has CPU test coverage; multi-GPU generation and performance still need GPU validation.
AWQ selects quantized layers from the checkpoint's `qweight` names, keeps dense layers dense, and rejects partially quantized fused QKV/gate-up layers.
Activation-order GPTQ, partially quantized GPTQ, quantized LM heads, and Marlin or other repacked layouts are rejected at startup.
The reference AWQ format requires a positive group size; AutoAWQ also accepts `-1`.

```python
llm = LLM("/YOUR/QWEN3-AWQ-MODEL/PATH", quantization="awq")  # or "gptq"; None auto-detects
llm = LLM("/YOUR/QWEN3-AWQ-MODEL/PATH", quantization="awq", tensor_parallel_size=2)  # requires two GPUs
# For a checkpoint exported by Brassinai/nano-vllm, whose layout differs from AutoAWQ:
llm = LLM("/YOUR/NANO-AWQ-CHECKPOINT", quantization="awq", awq_checkpoint_format="nano-vllm")
```

```bash
python -m unittest discover -s tests -p "test_awq*.py"    # AWQ: AutoAWQ-packed checkpoints (CPU) + Triton kernel (CUDA)
python -m unittest discover -s tests -p "test_gptq*.py"   # GPTQ: AutoGPTQ-packed checkpoints (CPU) + Triton kernel (CUDA)
python -m unittest discover -s tests                      # everything
python benchmarks/bench.py --model /YOUR/QWEN3-AWQ-MODEL/PATH --quantization awq
# Add --awq-checkpoint-format nano-vllm for the reference export format.
python scripts/test_awq.py --model /YOUR/QWEN3-AWQ-MODEL/PATH --disable-compile
```

If PyTorch Inductor fails at import time (e.g. `duplicate template name` in PyTorch 2.9.1),
set `TORCH_COMPILE_DISABLE=1` **before importing nanovllm**:

```bash
TORCH_COMPILE_DISABLE=1 python -m unittest discover -s tests -p "test_awq*.py" -v
TORCH_COMPILE_DISABLE=1 python benchmarks/bench.py --model /YOUR/QWEN3-AWQ-MODEL/PATH --quantization awq
```

This explicitly bypasses the project's `torch.compile` decorators and prints a warning.
Only PyTorch compilation of activation, RMSNorm, RoPE and sampling is disabled; the AWQ/GPTQ Triton JIT kernels,
FlashAttention and CUDA Graph remain enabled. There is no additional quantization or change to the mathematical formulas,
but floating-point rounding can differ and more kernels/intermediate tensors may reduce throughput or increase latency.
Performance measurements in this mode should be labeled accordingly. The default remains compilation enabled;
`enforce_eager=True` controls CUDA Graph separately and does not disable `torch.compile`.

Planned quantization work, in priority order:

1. Validate FP8, AWQ, and GPTQ generation quality and benchmark latency, throughput, and memory on GPU.
2. Add FP8 tensor parallelism and static activation-scale support.
3. Add INT8 W8A8 and quantized KV cache after the weight-only path is measured.

## Installation

```bash
git clone git@github.com:Ethan-el/dynamic-spec.git
cd dynamic-spec
pip install -e .
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `examples/example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Dynamic speculative decoding can be enabled with a batch-size schedule:

```python
llm = LLM(
    target_model_path,
    speculative_model=draft_model_path,
    num_speculative_tokens=5,
    num_speculative_tokens_per_batch_size=[
        (1, 3, 5),
        (4, 7, 3),
        (8, 8, 0),
    ],
    skip_draft_kv_on_k0=True,
    max_num_seqs=8,
)
```

## Benchmark

The formal dynamic speculative benchmark uses an RTX 3090, Qwen3-8B as the
Target model, Qwen3-0.6B as the Draft model, and eight heterogeneous requests.
Each request has 128 input tokens and an output limit from
`[80, 140, 200, 260, 320, 400, 460, 512]`.

The dynamic schedule is `BS=8 -> K=0`, `BS=4..7 -> K=3`, and
`BS=1..3 -> K=5`. CUDA Graph is enabled and K=0 Draft synchronization is
skipped.

| Inference Mode | Output Tokens | Time (s) | Throughput (tokens/s) | Speedup vs Decode |
|---|---:|---:|---:|---:|
| Ordinary Decode | 2,372 | 10.99 | 215.77 | 1.000x |
| Fixed K=3 | 2,372 | 12.60 | 188.31 | 0.873x |
| Fixed K=5 | 2,372 | 11.72 | 202.47 | 0.938x |
| Dynamic K=0/3/5 | 2,372 | 9.80 | **242.06** | **1.122x (+12.2%)** |

See `benchmarks/benchmark_dynamic_k_transition.py` for detailed diagnostics.
