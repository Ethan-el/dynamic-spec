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

Planned quantization work, in priority order:

1. Validate FP8 generation quality and benchmark latency, throughput, and memory on GPU.
2. Add FP8 tensor parallelism and static activation-scale support.
3. Add INT4 weight-only inference (W4A16), starting with AWQ/GPTQ checkpoint loading and a fused GEMM kernel.
4. Add INT8 W8A8 and quantized KV cache after the weight-only path is measured.

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
