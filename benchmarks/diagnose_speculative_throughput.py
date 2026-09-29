import argparse
import os
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--speculative-model", required=True)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--engine-max-num-seqs", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--no-reset-torch-seed", action="store_true")
    args = parser.parse_args()

    seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    llm = LLM(
        os.path.expanduser(args.model),
        speculative_model=os.path.expanduser(args.speculative_model),
        num_speculative_tokens=args.num_speculative_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.engine_max_num_seqs,
    )
    try:
        config = llm.model_runner.config
        print(
            "KV_BLOCKS "
            f"target={config.num_kvcache_blocks} "
            f"draft={config.num_draft_kvcache_blocks}",
            flush=True,
        )

        llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

        counters = {
            "preemptions": 0,
            "decode_steps": 0,
            "active_sequences": 0,
            "accepted": 0,
            "attempted": 0,
            "decode_graph_replays": 0,
            "verify_graph_replays": 0,
        }

        original_preempt = llm.scheduler.preempt
        original_schedule = llm.scheduler.schedule
        original_postprocess = llm.scheduler.postprocess
        original_decode_graph = llm.model_runner._run_decode_graph
        original_verify_graph = llm.model_runner._run_verify_graph

        def tracked_preempt(sequence):
            counters["preemptions"] += 1
            return original_preempt(sequence)

        def tracked_schedule():
            sequences, is_prefill = original_schedule()
            if not is_prefill:
                counters["decode_steps"] += 1
                counters["active_sequences"] += len(sequences)
            return sequences, is_prefill

        def tracked_postprocess(sequences, token_ids, is_prefill):
            if token_ids and isinstance(token_ids[0], list):
                counters["accepted"] += sum(
                    max(0, len(ids) - 1) for ids in token_ids
                )
                counters["attempted"] += (
                    len(sequences) * args.num_speculative_tokens
                )
            return original_postprocess(sequences, token_ids, is_prefill)

        def tracked_decode_graph(*graph_args, **graph_kwargs):
            counters["decode_graph_replays"] += 1
            return original_decode_graph(*graph_args, **graph_kwargs)

        def tracked_verify_graph(*graph_args, **graph_kwargs):
            counters["verify_graph_replays"] += 1
            return original_verify_graph(*graph_args, **graph_kwargs)

        llm.scheduler.preempt = tracked_preempt
        llm.scheduler.schedule = tracked_schedule
        llm.scheduler.postprocess = tracked_postprocess
        llm.model_runner._run_decode_graph = tracked_decode_graph
        llm.model_runner._run_verify_graph = tracked_verify_graph

        if not args.no_reset_torch_seed:
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.generate(
            prompt_token_ids,
            sampling_params,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        requested_tokens = sum(sp.max_tokens for sp in sampling_params)
        actual_tokens = sum(len(output["token_ids"]) for output in outputs)
        average_active_batch = (
            counters["active_sequences"] / counters["decode_steps"]
            if counters["decode_steps"]
            else 0.0
        )
        acceptance_rate = (
            counters["accepted"] / counters["attempted"]
            if counters["attempted"]
            else 0.0
        )

        print(
            "DIAGNOSTIC "
            f"requested_tokens={requested_tokens} "
            f"actual_tokens={actual_tokens} "
            f"elapsed_s={elapsed:.6f} "
            f"throughput_tok_s={actual_tokens / elapsed:.6f} "
            f"preemptions={counters['preemptions']} "
            f"decode_steps={counters['decode_steps']} "
            f"average_active_batch={average_active_batch:.6f} "
            f"acceptance_rate={acceptance_rate:.6f} "
            f"decode_graph_replays={counters['decode_graph_replays']} "
            f"verify_graph_replays={counters['verify_graph_replays']}",
            flush=True,
        )
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
