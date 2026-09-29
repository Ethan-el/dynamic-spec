#!/usr/bin/env bash
set -uo pipefail

profile=0
baseline=0
enforce_eager=0
while (($#)); do
    case "$1" in
        --profile)
            profile=1
            ;;
        --no-profile)
            profile=0
            ;;
        --baseline)
            baseline=1
            ;;
        --speculative)
            baseline=0
            ;;
        --eager)
            enforce_eager=1
            ;;
        --graph)
            enforce_eager=0
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/run_speculative_sweep.sh [MODE] [PROFILING]

Mode:
  --speculative   Target + draft speculative decoding (default)
  --baseline      Target-only ordinary autoregressive decoding

Profiling:
  --no-profile    Save metrics without Nsight Systems (default)
  --profile       Also save trace.nsys-rep and stats.txt

Execution:
  --graph         Use CUDA Graph for decode/draft/verify (default)
  --eager         Disable CUDA Graph

Environment overrides:
  INPUT_LEN=128 OUTPUT_LEN=128 REPEATS=3 SPEC_TOKENS=5 SEED=0
  BATCH_SIZES="1 2 4 8 16 32 64"
  DYNAMIC_SPEC_SCHEDULE='[[1,3,3],[4,8,0]]'
  RESULT_ROOT=/path/to/results

Default mode writes metrics.csv and per-batch logs.
--profile additionally writes trace.nsys-rep and stats.txt; its default
REPEATS is 1 unless REPEATS is explicitly set.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

source /root/autodl-tmp/activate_gpu_driver.sh || exit 1
cd /root/autodl-tmp/nano-vllm || exit 1

input_len=${INPUT_LEN:-128}
output_len=${OUTPUT_LEN:-128}
spec_tokens=${SPEC_TOKENS:-5}
dynamic_spec_schedule=${DYNAMIC_SPEC_SCHEDULE:-}
seed=${SEED:-0}
read -r -a batch_sizes <<< "${BATCH_SIZES:-1 2 4 8 16 32 64}"
if ((profile)); then
    repeats=${REPEATS:-1}
    export NANOVLLM_NVTX=1
    export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
else
    repeats=${REPEATS:-3}
fi
if ((baseline)); then
    result_name=baseline_qwen3_8b
else
    result_name=speculative_qwen3_8b_06b
fi
execution_name=$([[ $enforce_eager -eq 1 ]] && echo eager || echo cudagraph)
result_name="${result_name}_${execution_name}"
if ((profile)); then
    default_result_root="/root/autodl-tmp/results/nsys_${result_name}"
else
    default_result_root="/root/autodl-tmp/results/${result_name}"
fi
result_root=${RESULT_ROOT:-$default_result_root}

mkdir -p "$result_root" || exit 1
run_dir=$(mktemp -d "$result_root/sweep_XXXXXX") || exit 1
csv_path="$run_dir/metrics.csv"
decode_mode=$([[ $baseline -eq 1 ]] && echo baseline || echo speculative)
trace_mode=$([[ $profile -eq 1 ]] && echo profile || echo metrics)
echo "Mode: $decode_mode + $execution_name + $trace_mode"
echo "Results: $run_dir"

failed=0
for batch_size in "${batch_sizes[@]}"; do
    echo "=== batch_size=$batch_size ==="
    report_dir="$run_dir/bs${batch_size}"
    mkdir -p "$report_dir"
    benchmark_command=(
        /root/miniconda3/bin/python -u benchmarks/benchmark_speculative.py
        --batch-size "$batch_size"
        --input-len "$input_len"
        --output-len "$output_len"
        --repeats "$repeats"
        --seed "$seed"
        --csv "$csv_path"
    )
    if ((baseline)); then
        benchmark_command+=(--baseline)
    else
        benchmark_command+=(--num-speculative-tokens "$spec_tokens")
        if [[ -n "$dynamic_spec_schedule" ]]; then
            benchmark_command+=(--dynamic-speculative-schedule "$dynamic_spec_schedule")
        fi
    fi
    if ((enforce_eager)); then
        benchmark_command+=(--enforce-eager)
    fi

    if ((profile)); then
        if ! nsys profile \
            --trace=cuda,nvtx \
            --sample=none \
            --cpuctxsw=none \
            --capture-range=nvtx \
            --nvtx-capture=nano.benchmark.repeat_1 \
            --capture-range-end=stop \
            --output="$report_dir/trace" \
            "${benchmark_command[@]}" \
            2>&1 | tee "$report_dir/run.log"; then
            echo "FAILED profile batch_size=$batch_size" | tee -a "$run_dir/failures.log"
            failed=1
            continue
        fi

        if ! nsys stats \
            --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
            "$report_dir/trace.nsys-rep" \
            2>&1 | tee "$report_dir/stats.txt"; then
            echo "FAILED stats batch_size=$batch_size" | tee -a "$run_dir/failures.log"
            failed=1
        fi
    elif ! "${benchmark_command[@]}" \
        2>&1 | tee "$report_dir/run.log"; then
        echo "FAILED batch_size=$batch_size" | tee -a "$run_dir/failures.log"
        failed=1
    fi
done

echo "Metrics: $csv_path"
if ((profile)); then
    echo "Nsight reports: $run_dir/bs*/trace.nsys-rep"
fi
exit "$failed"
