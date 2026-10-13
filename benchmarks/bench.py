import argparse
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--quantization", choices=["fp8", "awq", "gptq"], default=None)
    parser.add_argument("--awq-checkpoint-format", choices=["autoawq", "nano-vllm"], default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-seqs", type=int, default=256)
    args = parser.parse_args()

    seed(0)
    num_seqs = args.num_seqs
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser(args.model)
    llm = LLM(path, enforce_eager=args.enforce_eager, max_model_len=4096, quantization=args.quantization,
              awq_checkpoint_format=args.awq_checkpoint_format)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
