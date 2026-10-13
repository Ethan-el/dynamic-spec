#!/usr/bin/env python3
"""Paired GSM8K accuracy and reference-answer conditional PPL for dense vs AWQ.

Official data: https://github.com/openai/grade-school-math
Protocol: zero-shot, thinking disabled, FP16 runtime for both models, greedy argmax.
PPL scores the held-out reference solution conditioned on the question; it is not
WikiText perplexity. Calculator annotations <<...>> are removed from references.
"""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
INSTRUCTION = ("Solve this math problem step by step. End your response with a line "
               "in the form '#### number' containing only the final numeric answer.\n\n")


def extract_answer(text, strict=True):
    if strict:
        matches = re.findall(r"####\s*(-?[0-9.,]+)", text)
    else:
        matches = [a or b for a, b in re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)]
    for match in reversed(matches):
        try:
            value = Decimal(match.replace(",", "").replace("$", "").strip())
            if value.is_finite():
                return str(value.normalize())
        except InvalidOperation:
            pass
    return None


def clean_reference(answer):
    return re.sub(r"<<.*?>>", "", answer, flags=re.DOTALL)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(args):
    if digest(args.dataset) != DATA_SHA256:
        raise ValueError("Dataset SHA256 does not match the official GSM8K test file")
    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    if len(cases) != 1319 or any(extract_answer(case["answer"]) is None for case in cases):
        raise ValueError("Invalid official GSM8K test data")
    return cases[:args.limit] if args.limit else cases


def protocol(args, count):
    code = hashlib.sha256()
    for path in sorted((ROOT / "nanovllm").rglob("*.py")):
        code.update(str(path.relative_to(ROOT)).encode())
        code.update(path.read_bytes())
    return {
        "dataset": "GSM8K official test", "dataset_sha256": DATA_SHA256, "samples": count,
        "shot_count": 0, "enable_thinking": False, "decoding": "greedy_argmax",
        "runtime_dtype": "float16", "max_tokens": args.max_tokens,
        "batch_size": args.batch_size, "max_model_len": 2048,
        "ppl_definition": "reference answer conditional PPL; calculator annotations removed",
        "torch_compile_disabled": os.environ.get("TORCH_COMPILE_DISABLE") == "1",
        "script_sha256": digest(__file__), "model_code_sha256": code.hexdigest(),
        "prompt_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
    }


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def conditional_ppl(llm, cases, label):
    import torch
    import torch.nn.functional as F
    from nanovllm.utils.context import set_context, reset_context

    loss_sum, token_count = 0.0, 0
    started = perf_counter()
    with torch.inference_mode():
        for index, case in enumerate(cases):
            prefix = prompt_ids(llm.tokenizer, case["question"])
            target = llm.tokenizer.encode(clean_reference(case["answer"]), add_special_tokens=False)
            ids = prefix + target
            if not target or len(ids) > llm.config.max_model_len:
                raise ValueError(f"Reference {index} is empty or exceeds the context limit")
            tokens = torch.tensor(ids, device="cuda", dtype=torch.long)
            positions = torch.arange(len(ids), device="cuda")
            cu = torch.tensor([0, len(ids)], dtype=torch.int32, device="cuda")
            try:
                set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                            max_seqlen_q=len(ids), max_seqlen_k=len(ids),
                            slot_mapping=torch.full((len(ids),), -1, dtype=torch.int32, device="cuda"))
                hidden = llm.model_runner.model(tokens, positions)
                # Prefill LM head normally keeps only each sequence's last token.
                # Reset that context to score every reference-answer token.
                reset_context()
                for start in range(len(prefix) - 1, len(ids) - 1, 32):
                    end = min(start + 32, len(ids) - 1)
                    logits = llm.model_runner.model.compute_logits(hidden[start:end]).float()
                    nll = F.cross_entropy(logits, tokens[start + 1:end + 1], reduction="sum")
                    if not torch.isfinite(nll):
                        raise RuntimeError(f"Non-finite NLL in reference {index}")
                    loss_sum += nll.item()
                    token_count += end - start
                del hidden
            finally:
                reset_context()
            if (index + 1) % 128 == 0 or index + 1 == len(cases):
                print(f"{label} PPL {index + 1}/{len(cases)}: "
                      f"NLL={loss_sum / token_count:.6f}, seconds={perf_counter() - started:.1f}", flush=True)
    return {"answer_tokens": token_count, "nll": loss_sum / token_count,
            "ppl": math.exp(loss_sum / token_count), "seconds": perf_counter() - started}


def run_worker(args, cases):
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    # Evaluation-only override: Config currently overwrites its dtype argument.
    # Keep both models' activations and KV cache FP16 without changing product code.
    original_init = ModelRunner.__init__
    def fp16_init(self, config, rank, event):
        config.dtype = torch.float16
        original_init(self, config, rank, event)
    ModelRunner.__init__ = fp16_init

    label = args.worker
    model_path = args.dense_model if label == "dense" else args.awq_model
    started = perf_counter()
    llm = LLM(model_path, quantization="awq" if label == "awq" else None,
              tensor_parallel_size=1, enforce_eager=False,
              max_model_len=2048, max_num_batched_tokens=1024,
              max_num_seqs=args.batch_size, gpu_memory_utilization=0.9)
    try:
        assert llm.config.dtype == torch.float16
        assert llm.model_runner.graphs
        # This is true greedy decoding; the engine otherwise uses stochastic sampling.
        llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
        print(f"{label} loaded: runtime=float16, checkpoint={model_path}, "
              f"seconds={perf_counter() - started:.1f}", flush=True)
        ppl = conditional_ppl(llm, cases, label)
        pending = {}
        for index, case in enumerate(cases):
            tokens = prompt_ids(llm.tokenizer, case["question"])
            if len(tokens) + args.max_tokens > llm.config.max_model_len:
                raise ValueError(f"Question {index} plus generation budget exceeds context")
            llm.add_request(tokens, SamplingParams(temperature=1.0, max_tokens=args.max_tokens))
            pending[llm.scheduler.waiting[-1].seq_id] = index

        predictions = []
        generation_started = last_progress = perf_counter()
        with tempfile.NamedTemporaryFile(mode="w", prefix=f"{label}_predictions.",
                                         suffix=".jsonl", dir=args.output_dir, delete=False) as output:
            prediction_path = output.name
            while not llm.is_finished():
                finished, _ = llm.step()
                for seq_id, token_ids in finished:
                    index = pending.pop(seq_id)
                    case = cases[index]
                    text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
                    gold = extract_answer(case["answer"])
                    strict = extract_answer(text)
                    flexible = strict if strict is not None else extract_answer(text, False)
                    row = {
                        "id": index, "gold": gold, "strict_prediction": strict,
                        "flexible_prediction": flexible, "strict_correct": strict == gold,
                        "flexible_correct": flexible == gold, "output_tokens": len(token_ids),
                        "truncated": len(token_ids) >= args.max_tokens and token_ids[-1] != llm.config.eos,
                        "text": text,
                    }
                    predictions.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                if finished:
                    output.flush()
                now = perf_counter()
                if now - last_progress >= 20 or llm.is_finished():
                    correct = sum(row["flexible_correct"] for row in predictions)
                    print(f"{label} GEN {len(predictions)}/{len(cases)}: "
                          f"correct={correct}, seconds={now - generation_started:.1f}", flush=True)
                    last_progress = now
        if pending or len(predictions) != len(cases):
            raise RuntimeError("Missing generated answers")
        result = {
            "model": label, "model_path": model_path, "protocol": protocol(args, len(cases)),
            "checkpoint_dtype": str(llm.config.hf_config.dtype),
            "gpu": torch.cuda.get_device_name(0), "ppl": ppl,
            "strict_correct": sum(row["strict_correct"] for row in predictions),
            "flexible_correct": sum(row["flexible_correct"] for row in predictions),
            "unparseable": sum(row["flexible_prediction"] is None for row in predictions),
            "truncated": sum(row["truncated"] for row in predictions),
            "generation_seconds": perf_counter() - generation_started,
            "generated_tokens": sum(row["output_tokens"] for row in predictions),
            "predictions_file": prediction_path,
        }
        Path(args.output_dir, f"{label}.json").write_text(json.dumps(result, indent=2))
        print(f"{label} DONE: {result['flexible_correct']}/{len(cases)}, "
              f"PPL={ppl['ppl']:.6f}, truncated={result['truncated']}", flush=True)
    finally:
        llm.exit()


def summarize(args, results):
    dense, awq = results
    if dense["protocol"] != awq["protocol"] or dense["ppl"]["answer_tokens"] != awq["ppl"]["answer_tokens"]:
        raise ValueError("Models were not evaluated with identical protocols/tokens")
    rows = []
    for result in results:
        items = [json.loads(line) for line in Path(result["predictions_file"]).read_text().splitlines()]
        rows.append({item["id"]: item for item in items})
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Question sets differ")
    n = len(rows[0])
    lost = sum(rows[0][i]["flexible_correct"] and not rows[1][i]["flexible_correct"] for i in rows[0])
    gained = sum(not rows[0][i]["flexible_correct"] and rows[1][i]["flexible_correct"] for i in rows[0])
    dense_acc, awq_acc = dense["flexible_correct"] / n, awq["flexible_correct"] / n
    drop = dense_acc - awq_acc
    se = math.sqrt(max(0, ((lost + gained) / n - drop * drop) / (n - 1))) if n > 1 else 0
    summary = {
        "protocol": dense["protocol"],
        "dense_accuracy_percent": 100 * dense_acc, "awq_accuracy_percent": 100 * awq_acc,
        "accuracy_drop_percentage_points": 100 * drop,
        "relative_accuracy_drop_percent": 100 * drop / dense_acc if dense_acc else None,
        "paired_drop_approx_95ci_percentage_points": [100 * (drop - 1.96 * se), 100 * (drop + 1.96 * se)],
        "dense_strict_accuracy_percent": 100 * dense["strict_correct"] / n,
        "awq_strict_accuracy_percent": 100 * awq["strict_correct"] / n,
        "dense_correct_awq_wrong": lost, "dense_wrong_awq_correct": gained,
        "dense_answer_ppl": dense["ppl"]["ppl"], "awq_answer_ppl": awq["ppl"]["ppl"],
        "ppl_increase_percent": 100 * (awq["ppl"]["ppl"] / dense["ppl"]["ppl"] - 1),
        "dense_truncated": dense["truncated"], "awq_truncated": awq["truncated"],
        "dense_unparseable": dense["unparseable"], "awq_unparseable": awq["unparseable"],
    }
    Path(args.output_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print("PAIRED RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "eval_data/gsm8k_test.jsonl"))
    parser.add_argument("--dense-model", default="/root/autodl-tmp/models/Qwen/Qwen3-8B")
    parser.add_argument("--awq-model", default="/root/autodl-tmp/models/Qwen3-8B-AWQ")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all 1319 test questions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--worker", choices=("dense", "awq"), help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert extract_answer("reasoning 100\n#### -1,234.50") == "-1234.5"
        assert extract_answer("10 items. Final answer: $42.", False) == "42"
        assert extract_answer("no numeric answer", False) is None
        assert clean_reference("3*2=<<3*2=6>>6\n#### 6") == "3*2=6\n#### 6"
        print("Parser/annotation self-tests passed")
        return
    if not args.output_dir or args.limit < 0 or args.max_tokens < 1 or args.batch_size < 1:
        parser.error("Provide --output-dir and positive token/batch limits")
    if args.disable_compile:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        print("torch.compile disabled explicitly for BOTH models; CUDA Graph/Triton remain enabled.", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    if args.worker:
        run_worker(args, cases)
        return
    if digest(Path(args.dense_model, "tokenizer.json")) != digest(Path(args.awq_model, "tokenizer.json")):
        raise ValueError("Tokenizers differ")
    if digest(Path(args.dense_model, "tokenizer_config.json")) != digest(Path(args.awq_model, "tokenizer_config.json")):
        raise ValueError("Chat templates/tokenizer configs differ")
    expected = protocol(args, len(cases))
    results = []
    for label in ("dense", "awq"):
        result_path = Path(args.output_dir, f"{label}.json")
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["protocol"] != expected:
                raise ValueError(f"Existing {label} results use a different protocol; choose a new output directory")
            print(f"Reusing completed {label} results", flush=True)
        else:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                            "--worker", label], check=True, cwd=ROOT)
            result = json.loads(result_path.read_text())
        results.append(result)
    summarize(args, results)


if __name__ == "__main__":
    main()
