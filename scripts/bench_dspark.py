"""4090 check for DSpark. Local CPU tests do not run this.

Greedy smoke test, then the reference size: 8 sequences, 256 new tokens, eager.

    python scripts/bench_dspark.py \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --draft /root/autodl-tmp/models/dspark_qwen3_8b_block7
"""

import argparse
import gc
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams

PROMPT = "Give me a short introduction to large language models."


def speculative_config(draft: str, block_size: int) -> dict:
    return {"method": "dspark", "model": draft, "num_speculative_tokens": block_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--bench", action="store_true", help="8 sequences x 256 tokens, plus a no-draft baseline")
    args = parser.parse_args()
    common = dict(enforce_eager=True, max_model_len=4096, max_num_batched_tokens=1024, tensor_parallel_size=1)

    llm = LLM(args.model, speculative_config=speculative_config(args.draft, args.block_size), **common)
    output = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=64))
    print(output[0]["text"])
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if not args.bench:
        return

    prompts = [PROMPT] * 8
    params = [SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)] * 8
    for label, spec in (("dspark", speculative_config(args.draft, args.block_size)), ("baseline", None)):
        engine = LLM(args.model, speculative_config=spec, **common)
        for prompt, sampling in zip(prompts, params):
            engine.add_request(prompt, sampling)
        accepted = attempted = 0
        started = perf_counter()
        while not engine.is_finished():
            engine.step()
            stats = engine.last_step_stats or {}
            accepted += stats.get("accepted_draft_tokens", 0)
            attempted += stats.get("attempted_draft_tokens", 0)
        elapsed = perf_counter() - started
        rate = accepted / attempted if attempted else 0
        print(f"{label}: {8 * 256 / elapsed:.2f} tok/s, accept={accepted}/{attempted} ({rate:.3f}), seconds={elapsed:.2f}")
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
