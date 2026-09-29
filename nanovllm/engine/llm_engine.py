import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.profiling import nvtx_range


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._closed = False
        self.last_step_stats = None
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.last_step_num_speculative_tokens = 0
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        with nvtx_range("nano.scheduler.schedule"):
            seqs, is_prefill = self.scheduler.schedule()
        num_speculative_tokens = self.scheduler.last_num_speculative_tokens
        self.last_step_num_speculative_tokens = num_speculative_tokens
        scheduled_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        with nvtx_range("nano.runner.prefill" if is_prefill else "nano.runner.decode"):
            token_ids = self.model_runner.call(
                "run", seqs, is_prefill, num_speculative_tokens
            )

        if not is_prefill and token_ids and isinstance(token_ids[0], list):
            generated_counts = [len(ids) for ids in token_ids]
            accepted_draft_tokens = sum(max(0, count - 1) for count in generated_counts)
            attempted_draft_tokens = len(seqs) * num_speculative_tokens
        elif not is_prefill:
            generated_counts = [1] * len(seqs)
            accepted_draft_tokens = 0
            attempted_draft_tokens = 0
        else:
            generated_counts = [0] * len(seqs)
            accepted_draft_tokens = 0
            attempted_draft_tokens = 0

        self.last_step_stats = {
            "is_prefill": is_prefill,
            "batch_size": len(seqs),
            "seq_ids": [seq.seq_id for seq in seqs],
            "generated_counts": generated_counts,
            "accepted_draft_tokens": accepted_draft_tokens,
            "attempted_draft_tokens": attempted_draft_tokens,
            "num_speculative_tokens": num_speculative_tokens,
            "draft_sync": bool(
                not is_prefill
                and self.config.speculative_model
                and num_speculative_tokens == 0
                and not self.config.skip_draft_kv_on_k0
            ),
            "draft_sync_skipped": bool(
                not is_prefill
                and self.config.speculative_model
                and num_speculative_tokens == 0
                and self.config.skip_draft_kv_on_k0
            ),
        }
        num_tokens = scheduled_tokens if is_prefill else -sum(generated_counts)

        with nvtx_range("nano.scheduler.postprocess"):
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
