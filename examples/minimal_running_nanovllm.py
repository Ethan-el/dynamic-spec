from nanovllm import LLM, SamplingParams


MODEL_PATH = (
    "/root/autodl-tmp/models/models/"
    "Qwen--Qwen3-0.6B/snapshots/master"
)


def main() -> None:
    print(f"Loading target model: {MODEL_PATH}", flush=True)
    llm = LLM(
        MODEL_PATH,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.7,
    )

    outputs = llm.generate(
        ["Hello, introduce yourself in one sentence."],
        SamplingParams(
            temperature=0.8,
            max_tokens=8,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )

    output = outputs[0]
    print("token_ids:", output["token_ids"])
    print("text:", repr(output["text"]))
    assert len(output["token_ids"]) == 8
    print("Minimal normal inference passed.")


if __name__ == "__main__":
    main()
