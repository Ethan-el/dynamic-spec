import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.profiling import nvtx_range


TARGET_MODEL = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-8B/snapshots/master"
)
DRAFT_MODEL = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-0.6B/snapshots/master"
)

METRICS = (
    "input_tokens",
    "output_tokens",
    "elapsed_s",
    "e2e_mean_ms",
    "output_throughput_tok_s",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "acceptance_rate",
    "mean_accepted_per_verify",
    "verify_iterations",
    "decode_steps",
    "peak_allocated_mib",
    "peak_reserved_mib",
)

SUMMARY_DISTRIBUTIONS = (
    "elapsed_s",
    "e2e_mean_ms",
    "output_throughput_tok_s",
    "ttft_mean_ms",
    "tpot_mean_ms",
    "acceptance_rate",
    "mean_accepted_per_verify",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-8B + Qwen3-0.6B speculative decoding."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--dynamic-speculative-schedule",
        type=json.loads,
        help='JSON ranges, for example "[[1,3,3],[4,8,0]]".',
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graph replay for target, draft, and verify decode.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run ordinary target-only autoregressive decoding.",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def measure_repeat(
    llm: LLM,
    prompts: list[list[int]],
    sampling: SamplingParams,
    repeat: int,
    num_speculative_tokens: int,
) -> dict:
    request_seqs = []
    for prompt in prompts:
        llm.add_request(prompt, sampling)
        request_seqs.append(llm.scheduler.waiting[-1])

    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    accepted_draft_tokens = 0
    attempted_draft_tokens = 0
    verify_iterations = 0
    decode_steps = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    with nvtx_range(f"nano.benchmark.repeat_{repeat}"):
        while not llm.is_finished():
            outputs, _ = llm.step()
            now = perf_counter()
            stats = llm.last_step_stats
            if stats and not stats["is_prefill"]:
                decode_steps += 1
                accepted_draft_tokens += stats["accepted_draft_tokens"]
                attempted_draft_tokens += stats["attempted_draft_tokens"]
                if stats["attempted_draft_tokens"]:
                    verify_iterations += len(stats["seq_ids"])
            for seq in request_seqs:
                if seq.num_completion_tokens and seq.seq_id not in first_token_at:
                    first_token_at[seq.seq_id] = now
            for seq_id, _ in outputs:
                finished_at[seq_id] = now

    torch.cuda.synchronize()
    elapsed = perf_counter() - start

    expected_tokens = len(prompts) * sampling.max_tokens
    actual_tokens = sum(seq.num_completion_tokens for seq in request_seqs)
    if actual_tokens != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} output tokens, got {actual_tokens}"
        )
    missing_first = [seq.seq_id for seq in request_seqs if seq.seq_id not in first_token_at]
    missing_finish = [seq.seq_id for seq in request_seqs if seq.seq_id not in finished_at]
    if missing_first or missing_finish:
        raise RuntimeError(
            f"Missing timing events: first={missing_first}, finish={missing_finish}"
        )

    ttft = [first_token_at[seq.seq_id] - start for seq in request_seqs]
    e2e = [finished_at[seq.seq_id] - start for seq in request_seqs]
    if sampling.max_tokens > 1:
        tpot = [
            (finished_at[seq.seq_id] - first_token_at[seq.seq_id])
            / (sampling.max_tokens - 1)
            for seq in request_seqs
        ]
    else:
        tpot = [0.0 for _ in request_seqs]

    return {
        "input_tokens": sum(len(prompt) for prompt in prompts),
        "output_tokens": actual_tokens,
        "num_speculative_tokens": num_speculative_tokens,
        "elapsed_s": elapsed,
        "e2e_mean_ms": mean(e2e) * 1000,
        "output_throughput_tok_s": actual_tokens / elapsed,
        "ttft_mean_ms": mean(ttft) * 1000,
        "ttft_p50_ms": percentile(ttft, 0.50) * 1000,
        "ttft_p99_ms": percentile(ttft, 0.99) * 1000,
        "tpot_mean_ms": mean(tpot) * 1000,
        "acceptance_rate": (
            accepted_draft_tokens / attempted_draft_tokens
            if attempted_draft_tokens
            else 0.0
        ),
        "mean_accepted_per_verify": (
            accepted_draft_tokens / verify_iterations
            if verify_iterations
            else 0.0
        ),
        "verify_iterations": verify_iterations,
        "decode_steps": decode_steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {name: mean(row[name] for row in rows) for name in METRICS}
    for name in SUMMARY_DISTRIBUTIONS:
        values = [row[name] for row in rows]
        summary[f"{name}_median"] = median(values)
        summary[f"{name}_stddev"] = pstdev(values)
    return summary


def make_prompts(batch_size: int, input_len: int, seed: int) -> list[list[int]]:
    generator = random.Random(seed)
    return [
        [generator.randint(100, 10_000) for _ in range(input_len)]
        for _ in range(batch_size)
    ]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    positive = (
        args.batch_size,
        args.input_len,
        args.output_len,
        args.repeats,
    )
    if min(positive) < 1:
        raise ValueError("Batch, lengths, and repeats must be positive")
    if not args.baseline and args.num_speculative_tokens < 1:
        raise ValueError("K must be positive for speculative decoding")

    effective_speculative_tokens = (
        0 if args.baseline else args.num_speculative_tokens
    )
    max_model_len = (
        args.input_len + args.output_len + effective_speculative_tokens + 8
    )
    engine_options = dict(
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=args.batch_size * args.input_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if not args.baseline:
        engine_options.update(
            speculative_model=DRAFT_MODEL,
            num_speculative_tokens=args.num_speculative_tokens,
            num_speculative_tokens_per_batch_size=(
                args.dynamic_speculative_schedule
            ),
        )
    llm = LLM(TARGET_MODEL, **engine_options)
    try:
        config = llm.model_runner.config
        block_manager = llm.scheduler.block_manager
        if not args.baseline and not block_manager.has_draft:
            raise RuntimeError("Scheduler did not initialize the draft KV pool")
        if args.baseline:
            print(
                "mode=baseline "
                f"execution={'eager' if args.enforce_eager else 'cudagraph'} "
                f"kv_blocks target={config.num_kvcache_blocks}"
            )
        else:
            print(
                "mode=speculative kv_blocks "
                f"execution={'eager' if args.enforce_eager else 'cudagraph'} "
                f"target={config.num_kvcache_blocks} "
                f"draft={config.num_draft_kvcache_blocks}"
            )

        if args.enforce_eager:
            print("graph_status enabled=false stages=none")
        else:
            graph_stages = ["target_decode"]
            if not args.baseline:
                graph_stages.extend(("draft_decode", "target_verify"))
            graph_buckets = ",".join(
                str(size) for size in llm.model_runner.graph_bs
            )
            print(
                "graph_status enabled=true "
                f"stages={','.join(graph_stages)} "
                f"buckets={graph_buckets} "
                "eager_fallback_if_input_tokens_gt=512"
            )

        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.output_len,
            ignore_eos=True,
        )
        warmup_prompts = make_prompts(
            args.batch_size,
            args.input_len,
            args.seed - 1,
        )

        with nvtx_range("nano.benchmark.warmup"):
            llm.generate(
                warmup_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=min(args.output_len, 8),
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )

        rows = []
        metadata = {
            "mode": "baseline" if args.baseline else "speculative",
            "execution": "eager" if args.enforce_eager else "cudagraph",
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_speculative_tokens": effective_speculative_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "ignore_eos": True,
        }
        for repeat in range(1, args.repeats + 1):
            # Deterministic but distinct prompts prevent prefix-cache reuse
            # across repeats and are identical between baseline/spec runs.
            prompts = make_prompts(
                args.batch_size,
                args.input_len,
                args.seed + repeat,
            )
            torch.manual_seed(args.seed + repeat)
            row = {
                **metadata,
                "repeat_seed": args.seed + repeat,
                **measure_repeat(
                    llm,
                    prompts,
                    sampling,
                    repeat,
                    effective_speculative_tokens,
                ),
            }
            rows.append(row)
            print("RESULT_JSON=" + json.dumps(row, sort_keys=True))

        summary = {**metadata, **summarize(rows)}
        print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
        if args.csv:
            csv_rows = [
                {"record_type": "repeat", "repeat": index, **row}
                for index, row in enumerate(rows, start=1)
            ]
            csv_rows.append(
                {"record_type": "summary", "repeat": 0, **summary}
            )
            append_csv(args.csv, csv_rows)
            print(f"csv={args.csv}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
