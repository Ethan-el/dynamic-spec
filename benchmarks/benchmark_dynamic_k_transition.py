"""Exercise one in-process BS=8 -> 1 dynamic speculative transition.

The eight requests share the same prompt length but use deliberately staggered
output limits.  Every decode step records the actual runnable batch size and
the K selected by Scheduler, both globally and per request.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


TARGET_MODEL = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-8B/snapshots/master"
)
DRAFT_MODEL = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-0.6B/snapshots/master"
)
DEFAULT_MAX_TOKENS = (80, 140, 200, 260, 320, 400, 460, 512)
DYNAMIC_SCHEDULE = [(1, 3, 5), (4, 7, 3), (8, 8, 0)]


def parse_max_tokens(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--max-tokens must be a comma-separated list of integers"
        ) from error
    if len(lengths) != 8:
        raise argparse.ArgumentTypeError("--max-tokens must contain 8 values")
    if any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("every max_tokens value must be positive")
    if tuple(sorted(lengths)) != lengths or len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError(
            "--max-tokens must be strictly increasing"
        )
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify BS-based K=0/3/5 switching in one generation run."
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument(
        "--max-tokens",
        type=parse_max_tokens,
        default=DEFAULT_MAX_TOKENS,
        help="Eight increasing output limits (default: 80,140,...,512).",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "/root/autodl-tmp/results/dynamic_k_transition_bs8.json"
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graph; useful only for correctness comparison.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the short K=5/K=3/K=0 warmup runs.",
    )
    parser.add_argument(
        "--skip-k0-draft-sync",
        action="store_true",
        help="Skip Draft KV work at K=0 and catch up when K becomes positive.",
    )
    return parser.parse_args()


def make_prompts(batch_size: int, input_len: int, seed: int) -> list[list[int]]:
    generator = random.Random(seed)
    return [
        [generator.randint(100, 10_000) for _ in range(input_len)]
        for _ in range(batch_size)
    ]


def expected_k(batch_size: int) -> int:
    if batch_size <= 3:
        return 5
    if batch_size <= 7:
        return 3
    return 0


def sorted_counter(counter: Counter) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter)
    }


def warmup(llm: LLM, input_len: int, temperature: float, seed: int) -> None:
    # Warm one representative batch for every execution branch.  These
    # requests finish before measurement and use different prompts.
    for offset, batch_size in enumerate((3, 5, 8), start=1):
        torch.manual_seed(seed - offset)
        llm.generate(
            make_prompts(batch_size, input_len, seed - 100 - offset),
            SamplingParams(
                temperature=temperature,
                max_tokens=4,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )


def main() -> None:
    args = parse_args()
    if args.input_len < 1:
        raise ValueError("--input-len must be positive")

    max_tokens = tuple(args.max_tokens)
    num_requests = len(max_tokens)
    max_k = 5
    max_model_len = args.input_len + max(max_tokens) + max_k + 8
    llm = LLM(
        TARGET_MODEL,
        speculative_model=DRAFT_MODEL,
        num_speculative_tokens=max_k,
        num_speculative_tokens_per_batch_size=DYNAMIC_SCHEDULE,
        skip_draft_kv_on_k0=args.skip_k0_draft_sync,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=num_requests * args.input_len,
        max_num_seqs=num_requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        verify_graph_keys = []
        if not args.enforce_eager:
            verify_graph_keys = sorted(llm.model_runner.verify_graphs)
            required_verify_graphs = {
                (3, 4),
                (3, 8),
                (5, 1),
                (5, 2),
                (5, 4),
            }
            missing = required_verify_graphs.difference(verify_graph_keys)
            if missing:
                raise RuntimeError(
                    f"missing dedicated Verify CUDA Graphs: {sorted(missing)}"
                )
            print(
                "VERIFY_GRAPHS_JSON="
                + json.dumps(verify_graph_keys)
            )

        if not args.skip_warmup:
            warmup(llm, args.input_len, args.temperature, args.seed)

        prompts = make_prompts(num_requests, args.input_len, args.seed)
        torch.manual_seed(args.seed)
        requests = []
        for request_index, (prompt, output_limit) in enumerate(
            zip(prompts, max_tokens),
            start=1,
        ):
            llm.add_request(
                prompt,
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=output_limit,
                    ignore_eos=True,
                ),
            )
            sequence = llm.scheduler.waiting[-1]
            requests.append(
                {
                    "request_index": request_index,
                    "seq_id": sequence.seq_id,
                    "max_tokens": output_limit,
                    "sequence": sequence,
                    "decode_steps_by_k": Counter(),
                    "output_tokens_by_k": Counter(),
                    "finish_decode_step": None,
                }
            )

        by_seq_id = {request["seq_id"]: request for request in requests}
        decode_step = 0
        step_count_by_k = Counter()
        request_iterations_by_k = Counter()
        output_tokens_by_k = Counter()
        elapsed_by_k = defaultdict(float)
        accepted_draft_tokens_by_k = Counter()
        attempted_draft_tokens_by_k = Counter()
        draft_sync_steps_by_k = Counter()
        draft_sync_skipped_steps_by_k = Counter()
        transition_events = []
        previous_state = None

        torch.cuda.synchronize()
        start = perf_counter()
        while not llm.is_finished():
            completion_before = {
                request["seq_id"]: request["sequence"].num_completion_tokens
                for request in requests
            }
            step_start = perf_counter()
            outputs, _ = llm.step()
            torch.cuda.synchronize()
            step_elapsed = perf_counter() - step_start
            stats = llm.last_step_stats
            if stats is None or stats["is_prefill"]:
                continue

            decode_step += 1
            batch_size = stats["batch_size"]
            selected_k = stats["num_speculative_tokens"]
            required_k = expected_k(batch_size)
            if selected_k != required_k:
                raise RuntimeError(
                    f"step {decode_step}: BS={batch_size} selected K={selected_k}, "
                    f"expected K={required_k}"
                )

            active_request_indices = []
            step_count_by_k[selected_k] += 1
            request_iterations_by_k[selected_k] += batch_size
            elapsed_by_k[selected_k] += step_elapsed
            accepted_draft_tokens_by_k[selected_k] += stats[
                "accepted_draft_tokens"
            ]
            attempted_draft_tokens_by_k[selected_k] += stats[
                "attempted_draft_tokens"
            ]
            draft_sync_steps_by_k[selected_k] += int(stats["draft_sync"])
            draft_sync_skipped_steps_by_k[selected_k] += int(
                stats["draft_sync_skipped"]
            )
            for seq_id in stats["seq_ids"]:
                request = by_seq_id[seq_id]
                sequence = request["sequence"]
                kept_tokens = (
                    sequence.num_completion_tokens - completion_before[seq_id]
                )
                request["decode_steps_by_k"][selected_k] += 1
                request["output_tokens_by_k"][selected_k] += kept_tokens
                output_tokens_by_k[selected_k] += kept_tokens
                active_request_indices.append(request["request_index"])

            state = (batch_size, selected_k)
            if state != previous_state:
                event = {
                    "decode_step": decode_step,
                    "batch_size": batch_size,
                    "k": selected_k,
                    "active_requests": active_request_indices,
                }
                transition_events.append(event)
                print("TRANSITION_JSON=" + json.dumps(event, sort_keys=True))
                previous_state = state

            for seq_id, _ in outputs:
                request = by_seq_id[seq_id]
                request["finish_decode_step"] = decode_step

        torch.cuda.synchronize()
        elapsed = perf_counter() - start

        request_results = []
        for request in requests:
            sequence = request["sequence"]
            if sequence.num_completion_tokens != request["max_tokens"]:
                raise RuntimeError(
                    f"request {request['request_index']} expected "
                    f"{request['max_tokens']} tokens, got "
                    f"{sequence.num_completion_tokens}"
                )
            result = {
                "request_index": request["request_index"],
                "seq_id": request["seq_id"],
                "max_tokens": request["max_tokens"],
                "observed_k_values": sorted(request["decode_steps_by_k"]),
                "decode_steps_by_k": sorted_counter(
                    request["decode_steps_by_k"]
                ),
                "output_tokens_by_k": sorted_counter(
                    request["output_tokens_by_k"]
                ),
                "prefill_output_tokens": 1,
                "total_output_tokens": sequence.num_completion_tokens,
                "finish_decode_step": request["finish_decode_step"],
            }
            request_results.append(result)
            print("REQUEST_JSON=" + json.dumps(result, sort_keys=True))

        observed_k_values = sorted(step_count_by_k)
        if observed_k_values != [0, 3, 5]:
            raise RuntimeError(
                f"expected to exercise K=0,3,5; observed {observed_k_values}"
            )

        per_k_metrics = {}
        for k in observed_k_values:
            attempted = attempted_draft_tokens_by_k[k]
            metrics = {
                "decode_steps": step_count_by_k[k],
                "elapsed_s": elapsed_by_k[k],
                "mean_step_ms": (
                    elapsed_by_k[k] / step_count_by_k[k] * 1000
                ),
                "request_iterations": request_iterations_by_k[k],
                "output_tokens": output_tokens_by_k[k],
                "accepted_draft_tokens": accepted_draft_tokens_by_k[k],
                "attempted_draft_tokens": attempted,
                "acceptance_rate": (
                    accepted_draft_tokens_by_k[k] / attempted
                    if attempted
                    else None
                ),
                "draft_sync_steps": draft_sync_steps_by_k[k],
                "draft_sync_skipped_steps": (
                    draft_sync_skipped_steps_by_k[k]
                ),
            }
            per_k_metrics[str(k)] = metrics
            print(
                "K_METRICS_JSON="
                + json.dumps({"k": k, **metrics}, sort_keys=True)
            )

        total_output_tokens = sum(max_tokens)
        summary = {
            "execution": "eager" if args.enforce_eager else "cudagraph",
            "input_len": args.input_len,
            "max_tokens": list(max_tokens),
            "dynamic_schedule": [list(entry) for entry in DYNAMIC_SCHEDULE],
            "schedule_tail": "K=0 for BS>=8",
            "skip_k0_draft_sync": args.skip_k0_draft_sync,
            "elapsed_s": elapsed,
            "total_output_tokens": total_output_tokens,
            "output_throughput_tok_s": total_output_tokens / elapsed,
            "decode_steps": decode_step,
            "observed_k_values": observed_k_values,
            "decode_step_count_by_k": sorted_counter(step_count_by_k),
            "request_iterations_by_k": sorted_counter(
                request_iterations_by_k
            ),
            "output_tokens_by_k": sorted_counter(output_tokens_by_k),
            "per_k_metrics": per_k_metrics,
            "verify_graph_keys": [list(key) for key in verify_graph_keys],
            "transition_events": transition_events,
            "requests": request_results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
        print(f"result_json={args.output_json}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
