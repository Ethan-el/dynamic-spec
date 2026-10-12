#!/usr/bin/env python3
"""Validate AWQ loading, CUDA numerics and real Qwen3 generation.

Uses fresh test/inference processes like Brassinai/nano-vllm's
scripts/benchmark_quantization_matrix.py at e909386a.
Run: python scripts/test_awq.py --model /path/to/Qwen3-AWQ --disable-compile
For the reference export format, also pass --awq-checkpoint-format nano-vllm.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3 AWQ checkpoint directory.")
    parser.add_argument("--awq-checkpoint-format", choices=("autoawq", "nano-vllm"), default=None)
    parser.add_argument("--disable-compile", action="store_true", help="Explicitly bypass torch.compile before import.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_model_checks(args):
    import torch
    import torch.nn.functional as F
    from nanovllm import LLM, SamplingParams
    from nanovllm.quantization.awq import AWQConfig

    if not torch.cuda.is_available():
        raise RuntimeError("AWQ integration tests require CUDA; CPU-only skips are not a pass.")
    # Limit CPU thread overhead during the checkpoint's one-time packing conversion.
    torch.set_num_threads(4)
    started = perf_counter()
    llm = LLM(
        str(Path(args.model).expanduser().resolve()),
        quantization="awq",
        awq_checkpoint_format=args.awq_checkpoint_format,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.4,
    )
    try:
        config = llm.config.quant_config
        assert isinstance(config, AWQConfig)
        assert llm.config.dtype == torch.float16
        assert llm.model_runner.graphs, "CUDA Graph was not captured"
        print(f"MODEL loaded: format={config.checkpoint_format}, group_size={config.group_size}, "
              f"device={torch.cuda.get_device_name(0)}, seconds={perf_counter() - started:.2f}", flush=True)

        checked = 0
        decoder = llm.model_runner.model.model.layers[0]
        with torch.inference_mode():
            for name in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
                layer = decoder.get_submodule(name)
                if not isinstance(layer.quant_config, AWQConfig):
                    continue
                assert layer.qweight.dtype == torch.int32
                weight = layer.quant_config.dequantize(layer.qweight, layer.qzeros, layer.scales).float()
                max_error = 0.0
                for batch_size in (1, 7):
                    torch.manual_seed(batch_size)
                    x = torch.randn(batch_size, layer.input_size, device="cuda", dtype=torch.float16)
                    bias = None if layer.bias is None else layer.bias.float()
                    expected = F.linear(x.float(), weight, bias)
                    actual = layer(x).float()
                    assert torch.isfinite(actual).all(), f"{name}: non-finite output"
                    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
                    max_error = max(max_error, (actual - expected).abs().max().item())
                checked += 1
                print(f"PASS real-weight projection {name}: max_abs_error={max_error:.6f}", flush=True)
                del weight
        assert checked, "No AWQ projection was tested"

        cases = (("What is 2 + 2? Reply with the number only.", "4"),
                 ("What is 3 + 5? Reply with the number only.", "8"))
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for question, _ in cases
        ]
        token_ids_by_mode = []
        for mode, eager in (("cuda_graph", False), ("eager", True)):
            llm.model_runner.enforce_eager = eager
            torch.manual_seed(0)
            # Near-greedy positive temperature: this fork rejects temperature=0.
            outputs = llm.generate(prompts, SamplingParams(temperature=1e-6, max_tokens=16), use_tqdm=False)
            assert len(outputs) == len(cases)
            for output, (_, expected_text) in zip(outputs, cases):
                text = llm.tokenizer.decode(output["token_ids"], skip_special_tokens=True).strip()
                assert text == expected_text, f"{mode}: expected {expected_text!r}, got {text!r}"
            assert not llm.last_step_stats["is_prefill"], f"{mode}: decode path was not exercised"
            token_ids_by_mode.append([output["token_ids"] for output in outputs])
            print(f"PASS {mode} generation: 2+2=4, 3+5=8", flush=True)
        assert token_ids_by_mode[0] == token_ids_by_mode[1], "CUDA Graph and eager token sequences differ"
        print("PASS CUDA Graph/eager token sequences match", flush=True)
    finally:
        llm.exit()


def main():
    args = parse_args()
    if args.worker:
        run_model_checks(args)
        return

    env = os.environ.copy()
    if args.disable_compile:
        env["TORCH_COMPILE_DISABLE"] = "1"
    if env.get("TORCH_COMPILE_DISABLE") == "1":
        print("torch.compile DISABLED explicitly: activation/RMSNorm/RoPE/sampling lose compilation fusion; "
              "latency or throughput may regress and rounding may differ. "
              "AWQ Triton JIT, FlashAttention and CUDA Graph stay enabled.", flush=True)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_awq*.py", "-v"],
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", args.model],
    ]
    if args.awq_checkpoint_format:
        commands[1].extend(["--awq-checkpoint-format", args.awq_checkpoint_format])
    for label, command in zip(("AWQ unit/CUDA regression", "Real-checkpoint integration"), commands):
        print(f"=== {label} ===", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("AWQ TEST PASSED: loading, CUDA numerics, generation and graph/eager agreement.", flush=True)


if __name__ == "__main__":
    main()
