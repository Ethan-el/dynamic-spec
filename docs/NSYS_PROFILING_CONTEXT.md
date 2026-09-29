# nano-vLLM Nsight Systems profiling context

## Objective

Measure the ordinary eager-decoding baseline for Qwen3-1.7B, then use the
same workload and measurement boundary to compare speculative decoding.

The performance benchmark and profiler serve different purposes:

- Run `benchmarks/benchmark_qwen3_1_7b.py` without nsys for throughput numbers.
- Run `profile_qwen3_1_7b.sh` to explain where time is spent.
- Do not compare the throughput printed under nsys with the unprofiled baseline.

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- Driver kernel module: 580.76.05
- PyTorch: 2.8.0+cu128
- Nsight Systems: 2026.5.1
- Tensor parallel size: 1
- Target model:
  `/root/autodl-tmp/models/models/Qwen--Qwen3-1.7B/snapshots/master`
- Draft model available for later speculative tests:
  `/root/autodl-tmp/models/models/Qwen--Qwen3-0.6B/snapshots/master`
- GPU environment activation:
  `source /root/autodl-tmp/activate_gpu_driver.sh`

The activation script selects NVIDIA user-space libraries matching the loaded
580.76.05 kernel driver and sets `OMP_NUM_THREADS=8`.

## Relevant files

- `benchmarks/benchmark_qwen3_1_7b.py`: fixed-length benchmark, matching-shape warmup,
  and NVTX measurement ranges.
- `profile_qwen3_1_7b.sh`: repeatable nsys collection and summary generation.
- `nanovllm/utils/profiling.py`: opt-in NVTX context manager.
- `nanovllm/engine/model_runner.py`: phase ranges such as prefill, decode,
  sampler, and the future speculative draft/verify/accept stages.

NVTX is disabled during normal execution. Enable it before profiling with:

```bash
export NANOVLLM_NVTX=1
```

## NVIDIA requirements implemented

The workflow follows the Nsight Systems CLI documentation:

1. `--trace=cuda,nvtx` collects CUDA activity and NVTX ranges.
2. `--capture-range=nvtx` starts collection at a named NVTX range.
3. `--nvtx-capture=nano.benchmark.repeat_1` selects the measured iteration.
4. `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` enables matching PyTorch's
   non-registered NVTX string.
5. `--capture-range-end=stop` stops collection when the selected range closes
   while allowing the target process to finish normally.
6. `nsys stats` generates `nvtx_sum`, `cuda_gpu_kern_sum`, and `cuda_api_sum`.

Official reference:
<https://docs.nvidia.com/nsight-systems/UserGuide/>

## Run commands

Run an unprofiled eager baseline:

```bash
source /root/autodl-tmp/activate_gpu_driver.sh
cd /root/autodl-tmp/nano-vllm
python benchmarks/benchmark_qwen3_1_7b.py \
  --batch-size 1 \
  --input-len 128 \
  --output-len 128 \
  --repeats 3
```

Collect an nsys report. Arguments are batch size, input length, and output
length:

```bash
bash profile_qwen3_1_7b.sh 1 128 128
bash profile_qwen3_1_7b.sh 2 128 128
```

Each invocation creates a unique directory under:

```text
/root/autodl-tmp/results/nsys_qwen3_17b/
```

The directory contains:

- `trace.nsys-rep`: timeline for the Nsight Systems GUI.
- `trace.sqlite`: exported report database.
- `run.log`: benchmark and nsys output.
- `stats.txt`: NVTX, CUDA kernel, and CUDA API summaries.

## Current baseline

Workload: eager mode, batch 1, input 128 tokens, output 128 tokens.

- Unprofiled mean: 29.59 output tokens/s, 4.327 s.
- Profiled run: 23.96 output tokens/s, 5.343 s.

The profiled value is lower because tracing adds overhead. The unprofiled
number remains the performance baseline.

Batch 2 unprofiled result:

- Mean: 53.91 output tokens/s, 4.750 s.
- Speedup over batch 1: 1.82x.
- Scaling efficiency: about 91.1%.

## First nsys report

Report directory:

```text
/root/autodl-tmp/results/nsys_qwen3_17b/b1_i128_o128_nLc6Qi/
```

Important observations from the summary:

- `nano.benchmark.repeat_1`: 5.343 s.
- `nano.model.prefill`: 48.6 ms, one invocation.
- `nano.model.decode_eager`: 127 invocations, 5.114 s total,
  40.27 ms average.
- Sampler GPU and CPU-transfer ranges each account for about 0.5%.
- Kernel time is led by BF16 GEMM kernels, followed by GEMV and FlashAttention.
- CUDA API activity contains tens of thousands of kernel launches.

The summary shows that decode dominates end-to-end time and that the workload
is highly fragmented into kernel launches. Use the GUI timeline to distinguish
CPU dispatch gaps from GPU execution before drawing a final bottleneck
conclusion.

## Next analysis sequence

1. Collect batch 2 with `bash profile_qwen3_1_7b.sh 2 128 128`.
2. Compare decode-range duration, kernel launch count, GPU idle gaps, and kernel
   duration between batch 1 and batch 2.
3. Complete and validate speculative scheduler/engine support.
4. Profile target Qwen3-1.7B plus draft Qwen3-0.6B with the same workload.
5. Compare draft, verify, accept, sampler, and target decode ranges.

Nsight Compute (`ncu`) is not installed in this image. Use Nsight Systems to
identify representative kernels first; install/use Nsight Compute only when a
kernel-level roofline or memory analysis is needed.
