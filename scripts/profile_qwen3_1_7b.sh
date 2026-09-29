#!/usr/bin/env bash
set -euo pipefail

# Usage: bash profile_qwen3_1_7b.sh [batch_size] [input_len] [output_len]
source /root/autodl-tmp/activate_gpu_driver.sh
cd /root/autodl-tmp/nano-vllm
export NANOVLLM_NVTX=1
# PyTorch push/pop ranges use non-registered strings.
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

batch_size=${1:-1}
input_len=${2:-128}
output_len=${3:-128}
use_cuda_graph=${4:-0}
for value in "$batch_size" "$input_len" "$output_len"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Arguments must be positive integers' >&2; exit 2; }
done

mkdir -p /root/autodl-tmp/results/nsys_qwen3_17b
report_dir=$(mktemp -d /root/autodl-tmp/results/nsys_qwen3_17b/b${batch_size}_i${input_len}_o${output_len}_XXXXXX)
echo "Reports: $report_dir"
nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=nvtx \
    --nvtx-capture=nano.benchmark.repeat_1 \
    --capture-range-end=stop \
    --output="$report_dir/trace" \
    /root/miniconda3/bin/python -u benchmarks/benchmark_qwen3_1_7b.py \
        --batch-size "$batch_size" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --repeats 1 2>&1 | tee "$report_dir/run.log"

nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
    "$report_dir/trace.nsys-rep" 2>&1 | tee "$report_dir/stats.txt"
echo "Report: $report_dir/trace.nsys-rep"
