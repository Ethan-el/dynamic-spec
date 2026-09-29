import argparse
from statistics import mean
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.profiling import nvtx_range


MODEL_PATH = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-1.7B/snapshots/master"
)
DRAFT_PATH = (
    "/root/autodl-tmp/models/models/Qwen--Qwen3-0.6B/snapshots/master"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable the ordinary one-token CUDA graph decode path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.input_len, args.output_len, args.repeats) < 1:
        raise ValueError("All numeric arguments must be positive")

    max_model_len = args.input_len + args.output_len + 8
    
    llm = LLM(
        MODEL_PATH,
        speculative_model=DRAFT_PATH,      # 改成你的 draft 路径
        num_speculative_tokens=3,
        tensor_parallel_size=1,
        # enforce_eager=not args.cuda_graph,
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=args.batch_size * args.input_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=0.8,
    )
    cfg = llm.model_runner.config
    bm = llm.scheduler.block_manager
    print("num_draft_kvcache_blocks", cfg.num_draft_kvcache_blocks)
    print("has_draft", bm.has_draft)
    print("draft_pool", None if bm.free_draft_block_ids is None else len(bm.free_draft_block_ids))

    sampling = SamplingParams(
        temperature=0.8,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    prompts = [[100] * args.input_len for _ in range(args.batch_size)]

    # Compile kernels for the same batch shape used by the measured run.
    warmup_prompts = [
        [100] * min(args.input_len, 32)
        for _ in range(args.batch_size)
    ]
    with nvtx_range("nano.benchmark.warmup"):
        llm.generate(
            warmup_prompts,
            SamplingParams(temperature=0.8, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )

    throughputs = []
    latencies = []
    expected_output_tokens = args.batch_size * args.output_len
    for repeat in range(args.repeats):
        torch.cuda.synchronize()
        with nvtx_range(f"nano.benchmark.repeat_{repeat + 1}"):
            start = perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
        actual_output_tokens = sum(len(output["token_ids"]) for output in outputs)
        if actual_output_tokens != expected_output_tokens:
            raise RuntimeError(
                f"Expected {expected_output_tokens} output tokens, "
                f"got {actual_output_tokens}"
            )
        throughput = actual_output_tokens / elapsed
        latencies.append(elapsed)
        throughputs.append(throughput)
        print(
            f"repeat={repeat + 1} elapsed={elapsed:.3f}s "
            f"output_throughput={throughput:.2f} tok/s"
        )

    print("--- summary ---")
    print(f"model={MODEL_PATH}")
    print(f"mode={'cuda_graph' if args.cuda_graph else 'eager'}")
    print(f"batch_size={args.batch_size}")
    print(f"input_len={args.input_len}")
    print(f"output_len={args.output_len}")
    print(f"mean_elapsed={mean(latencies):.3f}s")
    print(f"mean_output_throughput={mean(throughputs):.2f} tok/s")


if __name__ == "__main__":
    main()
