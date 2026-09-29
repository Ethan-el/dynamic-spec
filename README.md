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

## Installation

```bash
git clone git@github.com:Ethan-el/nano-vllm-dynamic-spec.git
cd nano-vllm-dynamic-spec
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

See `benchmarks/` for benchmarks.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
