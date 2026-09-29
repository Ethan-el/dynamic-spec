# Dynamic Speculative Decoding V1 (SSH) Technical Report

## 1. Goal and current status

This change implements batch-size-based dynamic speculative decoding in the
SSH repository at `/root/autodl-tmp/nano-vllm`.

By default Draft KV stays synchronized when `K=0`. Setting
`skip_draft_kv_on_k0=True` removes that Draft forward; the next `K>0` step
uses the existing Draft prefill path to catch up the missing committed tokens.

CPU-safe syntax and policy tests pass. Model-level correctness and performance
testing is deferred until the GPU instance is enabled.

## 2. Configuration

`num_speculative_tokens` is the maximum K and the KV-capacity upper bound.
`num_speculative_tokens_per_batch_size` is an inclusive range table:

```python
llm = LLM(
    target_model_path,
    speculative_model=draft_model_path,
    num_speculative_tokens=3,
    num_speculative_tokens_per_batch_size=[
        (1, 3, 3),
        (4, 8, 0),
    ],
    skip_draft_kv_on_k0=True,
    max_num_seqs=8,
)
```

The example means `BS=1..3 -> K=3` and `BS=4..8 -> K=0`. The first range
must start at one, ranges cannot overlap, gaps and the uncovered tail inherit
the previous K, and values above the configured maximum are capped.

If no dynamic table is supplied, the existing static behavior remains: every
decode step uses `num_speculative_tokens`.

## 3. End-to-end control flow

```text
Config
  -> validates/normalizes the dynamic table
  -> Scheduler expands it into O(1) batch_size -> K lookup

Scheduler.schedule()
  -> completes admission and any preemption
  -> obtains the actual runnable decode batch size
  -> selects K for this step

LLMEngine.step()
  -> passes selected K to every ModelRunner rank

ModelRunner.run()
  -> K > 0: Draft proposes K tokens, Target verifies K+1 positions
  -> K = 0: Target decodes one token; Draft KV syncs unless skip is enabled

Scheduler.postprocess()
  -> list[int]: ordinary K=0 result
  -> list[list[int]]: accepted draft prefix plus correction/bonus result
```

K is selected after the decode batch is finalized. Therefore, when short
requests finish and an active batch falls from eight to three, the next decode
step changes from K=0 to K=3.

## 4. K=0 Draft KV invariant

At a normal postprocessed decode boundary:

```text
draft_kv_len == len(sequence) - 1
```

During a K=0 step, Target and Draft both consume the current last committed
token. Draft KV advances to the pre-append sequence length, then Scheduler
appends the newly sampled target token. The invariant is restored for the next
step. When Draft is further behind (for example after preemption), the sync
path uses `prepare_draft_prefill()` to consume every missing committed token.

The K=0 sync deliberately skips the Draft LM head because no proposal logits
are needed. With `skip_draft_kv_on_k0=True`, Draft KV remains stale until a
positive-K step calls `prepare_draft_prefill()` to catch up before proposing.

## 5. CUDA Graph behavior

The SSH version already had Target, Draft, and Verify CUDA Graph support. V1
preserves it as follows:

- K=0 Target decode replays the existing Target decode graph.
- K=0 Draft KV sync replays the Draft decode graph and skips the LM head; the
  optional skip mode does no Draft work.
- Every positive K required by the schedule has a dedicated Verify graph keyed
  by `(K, batch_bucket)`.
- Runtime verification selects that exact key; an uncaptured shape falls back
  to eager verification.

## 6. Files changed and why

- `nanovllm/dynamic_speculative.py`: pure-Python validation and dense lookup.
- `nanovllm/config.py`: public dynamic schedule configuration.
- `nanovllm/engine/scheduler.py`: selects K from the final decode batch size.
- `nanovllm/engine/llm_engine.py`: carries per-step K into ModelRunner and
  exposes K/draft-sync in `last_step_stats`.
- `nanovllm/engine/model_runner.py`: active-K buffers and verification width,
  K=0 target path, and Draft KV-only synchronization.
- `benchmarks/benchmark_speculative.py`: accepts `--dynamic-speculative-schedule`.
- `scripts/run_speculative_sweep.sh`: accepts `DYNAMIC_SPEC_SCHEDULE`.
- `tests/test_dynamic_speculative.py`: validates lookup, gaps, caps, and bad
  tables without importing the CUDA runtime.

Block allocation remains based on maxK, not the current K. This is
conservative but ensures that a later K increase never lacks already-reserved
Target/Draft KV capacity.

## 7. CPU verification

```bash
cd /root/autodl-tmp/nano-vllm
/root/miniconda3/bin/python -m compileall -q \
  nanovllm benchmarks/benchmark_speculative.py tests/test_dynamic_speculative.py
/root/miniconda3/bin/python -m unittest \
  tests.test_dynamic_speculative -v
git diff --check
```

## 8. GPU validation commands

A fixed-BS point test:

```bash
cd /root/autodl-tmp/nano-vllm
source /root/autodl-tmp/activate_gpu_driver.sh
python -u benchmarks/benchmark_speculative.py \
  --batch-size 8 \
  --input-len 128 \
  --output-len 1024 \
  --repeats 1 \
  --num-speculative-tokens 3 \
  --dynamic-speculative-schedule '[[1,3,3],[4,8,0]]'
```

A sweep with the same policy:

```bash
DYNAMIC_SPEC_SCHEDULE='[[1,3,3],[4,8,0]]' \
SPEC_TOKENS=3 \
BATCH_SIZES='1 2 3 4 8' \
INPUT_LEN=128 \
OUTPUT_LEN=1024 \
REPEATS=1 \
bash scripts/run_speculative_sweep.sh --graph --no-profile
```

The existing fixed-length benchmark launches one independent engine per batch
size, so it validates each table point but does not produce an in-process
BS=8 -> BS=3 transition. That transition requires requests with different
output lengths and should be the first dedicated GPU correctness test.

## 9. Recovery

The pre-change remote files were copied to:

```text
/root/autodl-tmp/nano-vllm-backups/dynamic-k-v1-20260927
```
