#!/usr/bin/env bash
set -uo pipefail

source /root/autodl-tmp/activate_gpu_driver.sh || exit 1
cd /root/autodl-tmp/nano-vllm || exit 1

repeats=${REPEATS:-3}
spec_tokens=${SPEC_TOKENS:-5}
batch_sizes=${BATCH_SIZES:-"1 2 4"}
include_eager=${INCLUDE_EAGER:-0}
seed=${SEED:-0}

result_root=/root/autodl-tmp/results/qwen3_8b_speculative_experiments
mkdir -p "$result_root" || exit 1
run_dir=$(mktemp -d "$result_root/matrix_XXXXXX") || exit 1
manifest="$run_dir/manifest.csv"
master_log="$run_dir/run.log"
environment="$run_dir/environment.txt"

modes='baseline_graph|speculative_graph'
if [[ $include_eager == 1 ]]; then
    modes+='|baseline_eager|speculative_eager'
fi

printf '%s\n' \
    'case,input_len,output_len,batch_sizes,repeats,spec_tokens,seed,modes,mode_order,ignore_eos,profile' \
    "short,128,128,${batch_sizes// /|},$repeats,$spec_tokens,$seed,$modes,baseline_first,true,false" \
    "prefill_heavy,1024,128,${batch_sizes// /|},$repeats,$spec_tokens,$seed,$modes,speculative_first,true,false" \
    "decode_heavy,128,1024,${batch_sizes// /|},$repeats,$spec_tokens,$seed,$modes,baseline_first,true,false" \
    "long_context,1024,1024,${batch_sizes// /|},$repeats,$spec_tokens,$seed,$modes,speculative_first,true,false" \
    > "$manifest"

{
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    printf 'git_status_begin\n'
    git status --short 2>/dev/null || true
    printf 'git_status_end\n'
    printf 'target_model=%s\n' '/root/autodl-tmp/models/models/Qwen--Qwen3-8B/snapshots/master'
    printf 'draft_model=%s\n' '/root/autodl-tmp/models/models/Qwen--Qwen3-0.6B/snapshots/master'
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
        --format=csv,noheader 2>/dev/null || true
    /root/miniconda3/bin/python -c \
        'import flash_attn, torch, transformers, triton; print(f"python_stack=torch:{torch.__version__},triton:{triton.__version__},transformers:{transformers.__version__},flash_attn:{flash_attn.__version__}")' \
        2>/dev/null || true
} > "$environment"

echo "Experiment results: $run_dir" | tee "$master_log"
echo "SPEC_TOKENS=$spec_tokens REPEATS=$repeats BATCH_SIZES=$batch_sizes SEED=$seed" \
    | tee -a "$master_log"

cases=(
    "short:128:128"
    "prefill_heavy:1024:128"
    "decode_heavy:128:1024"
    "long_context:1024:1024"
)

failed=0
run_case() {
    local case_name=$1
    local input_len=$2
    local output_len=$3
    local mode=$4
    local execution=$5
    local combination_dir="$run_dir/$case_name/${mode}_${execution}"

    echo \
        "=== case=$case_name input=$input_len output=$output_len mode=$mode execution=$execution ===" \
        | tee -a "$master_log"

    local args=(--no-profile)
    if [[ $mode == baseline ]]; then
        args+=(--baseline)
    else
        args+=(--speculative)
    fi
    if [[ $execution == eager ]]; then
        args+=(--eager)
    else
        args+=(--graph)
    fi

    if ! BATCH_SIZES="$batch_sizes" \
        INPUT_LEN="$input_len" \
        OUTPUT_LEN="$output_len" \
        REPEATS="$repeats" \
        SPEC_TOKENS="$spec_tokens" \
        SEED="$seed" \
        RESULT_ROOT="$combination_dir" \
        bash scripts/run_speculative_sweep.sh "${args[@]}" \
        2>&1 | tee -a "$master_log"; then
        printf 'FAILED case=%s mode=%s execution=%s\n' \
            "$case_name" "$mode" "$execution" \
            | tee -a "$run_dir/failures.log" "$master_log"
        failed=1
    fi
}

case_index=0
for case_spec in "${cases[@]}"; do
    IFS=: read -r case_name input_len output_len <<< "$case_spec"
    if ((case_index % 2 == 0)); then
        run_case "$case_name" "$input_len" "$output_len" baseline graph
        run_case "$case_name" "$input_len" "$output_len" speculative graph
    else
        run_case "$case_name" "$input_len" "$output_len" speculative graph
        run_case "$case_name" "$input_len" "$output_len" baseline graph
    fi

    if [[ $include_eager == 1 ]]; then
        if ((case_index % 2 == 0)); then
            run_case "$case_name" "$input_len" "$output_len" baseline eager
            run_case "$case_name" "$input_len" "$output_len" speculative eager
        else
            run_case "$case_name" "$input_len" "$output_len" speculative eager
            run_case "$case_name" "$input_len" "$output_len" baseline eager
        fi
    fi
    ((case_index += 1))
done

echo "Manifest: $manifest" | tee -a "$master_log"
echo "Environment: $environment" | tee -a "$master_log"
echo "Master log: $master_log" | tee -a "$master_log"
exit "$failed"
