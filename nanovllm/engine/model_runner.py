import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.dynamic_speculative import build_dynamic_speculative_lookup
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.speculative import rejection_sample
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.profiling import nvtx_range


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # Capacity and captured Verify Graph upper bound. Scheduler supplies
        # the active K independently on every decode step.
        self.max_num_speculative_tokens = config.num_speculative_tokens
        if config.speculative_model is not None and self.max_num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be positive when a draft model is configured")

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config, config.quant_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.draft_hf_config = None
        self.draft_model = None
        if config.speculative_model is not None:
            self.draft_hf_config = AutoConfig.from_pretrained(config.speculative_model)
            assert self.draft_hf_config.vocab_size == hf_config.vocab_size, "target and draft vocab sizes must match"
            assert self.draft_hf_config.dtype == hf_config.dtype, "target and draft dtypes must match in the MVP"
            assert self.draft_hf_config.max_position_embeddings >= config.max_model_len
            self.draft_model = Qwen3ForCausalLM(self.draft_hf_config)
            load_model(self.draft_model, config.speculative_model)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.draft_model is not None:
                self.capture_draft_cudagraph()
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        if hasattr(self, "draft_graphs"):
            del self.draft_graphs, self.draft_graph_pool
        if hasattr(self, "verify_graphs"):
            del self.verify_graphs, self.verify_graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if self.draft_model is not None:
            # Include the draft forward's temporary allocations in the peak
            # used to size the KV caches. Warmup sequences have no cache blocks.
            try:
                input_ids, positions = self.prepare_prefill(seqs)
                logits = self.run_draft_model(input_ids, positions, True)
                if self.rank == 0:
                    self.sampler(logits, self.prepare_sample(seqs))
                del logits
            finally:
                reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        available = int(total * config.gpu_memory_utilization - used - peak + current)
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if self.draft_model is not None:
            draft_config = self.draft_hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(
                draft_config,
                "head_dim",
                draft_config.hidden_size // draft_config.num_attention_heads,
            )
            draft_block_bytes = (
                2
                * draft_config.num_hidden_layers
                * self.block_size
                * draft_num_kv_heads
                * draft_head_dim
                * draft_config.dtype.itemsize
            )
            # Use a common logical capacity; independently rounded budgets can
            # allocate extra draft blocks that have no matching target capacity.
            num_blocks = available // (block_bytes + draft_block_bytes)
        else:
            num_blocks = available // block_bytes

        config.num_kvcache_blocks = num_blocks
        if num_blocks <= 0:
            raise RuntimeError("Insufficient GPU memory for a KV cache block")
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self._bind_kv_cache(self.model, self.kv_cache)

        config.num_draft_kvcache_blocks = 0
        if self.draft_model is not None:
            config.num_draft_kvcache_blocks = num_blocks
            self.draft_kv_cache = torch.empty(
                2,
                draft_config.num_hidden_layers,
                config.num_draft_kvcache_blocks,
                self.block_size,
                draft_num_kv_heads,
                draft_head_dim,
            )
            self._bind_kv_cache(self.draft_model, self.draft_kv_cache)

    @staticmethod
    def _bind_kv_cache(model, kv_cache):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence], use_draft: bool = False):
        tables = [seq.draft_block_table if use_draft else seq.block_table for seq in seqs]
        max_len = max(len(table) for table in tables)
        block_tables = [table + [-1] * (max_len - len(table)) for table in tables]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        #prefix cache 命中时，cached block 已经有 KV；
        # new block 还没有 KV，但已经被分配地址，
        # 本轮 forward 会把 KV 写进去。block_tables 必须描述两部分。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs) #block_tables，告诉 attention 去哪里读 cached KV
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            position = len(seq) - 1
            positions.append(position)
            context_lens.append(len(seq))
            slot_mapping.append(self._slot_for(seq.block_table, position))
        #list → CPU tensor → GPU tensor。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _slot_for(self, block_table: list[int], position: int) -> int:
        if position < 0:
            raise ValueError("KV cache positions must be non-negative")
        block_index, block_offset = divmod(position, self.block_size)
        if block_index >= len(block_table) or block_table[block_index] < 0:
            raise ValueError(
                f"No KV cache block allocated for position {position}; "
                "the scheduler must reserve blocks for speculative tokens"
            )
        return block_table[block_index] * self.block_size + block_offset

    def prepare_draft_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        sample_indices = []
        for seq in seqs:
            start = seq.draft_kv_len
            end = len(seq)
            if not 0 <= start < end:
                raise ValueError("draft_kv_len must leave at least one uncached token")
            seqlen_q = end - start
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, end)
            slot_mapping.extend(self._slot_for(seq.draft_block_table, position) for position in range(start, end))
            sample_indices.append(cu_seqlens_q[-1] - 1)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs, use_draft=True)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sample_indices = torch.tensor(sample_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions, sample_indices

    def prepare_draft_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            position = len(seq) - 1
            input_ids.append(seq.last_token)
            positions.append(position)
            context_lens.append(len(seq))
            slot_mapping.append(self._slot_for(seq.draft_block_table, position))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs, use_draft=True)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify(
        self,
        seqs: list[Sequence],
        num_speculative_tokens: int,
    ):
        verify_width = num_speculative_tokens + 1
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            start = len(seq) - verify_width
            for position in range(start, len(seq)):
                input_ids.append(seq[position])
                positions.append(position)
                slot_mapping.append(self._slot_for(seq.block_table, position))
            context_lens.append(len(seq))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            num_speculative_tokens=num_speculative_tokens,
        )
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        is_verify = getattr(context, "num_speculative_tokens", 0) > 0
        verify_graph_key = (
            self._get_verify_graph_key(
                input_ids.size(0),
                context.num_speculative_tokens,
            )
            if is_verify and hasattr(self, "verify_graphs")
            else None
        )
        if (
            is_prefill or self.enforce_eager or input_ids.size(0) > 512
            or (is_verify and verify_graph_key is None)
        ):
            phase = "prefill" if is_prefill else ("verify_eager" if is_verify else "decode_eager")
            with nvtx_range(f"nano.model.{phase}"):
                hidden_states = self.model(input_ids, positions)
            with nvtx_range("nano.model.lm_head"):
                return self.model.compute_logits(hidden_states)

        if is_verify:
            return self._run_verify_graph(
                input_ids,
                positions,
                context,
                verify_graph_key,
            )
        return self._run_decode_graph(
            input_ids,
            positions,
            context,
            self.graphs,
            self.graph_vars,
            self.model,
            "target",
        )

    @torch.inference_mode()
    def run_draft_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ):
        assert self.draft_model is not None
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            phase = "prefill" if is_prefill else "decode_eager"
            with nvtx_range(f"nano.speculative.draft_{phase}"):
                hidden_states = self.draft_model(input_ids, positions)
            with nvtx_range("nano.speculative.draft_lm_head"):
                return self.draft_model.compute_logits(hidden_states)

        return self._run_decode_graph(
            input_ids,
            positions,
            get_context(),
            self.draft_graphs,
            self.draft_graph_vars,
            self.draft_model,
            "draft",
        )

    @torch.inference_mode()
    def run_draft_kv_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> None:
        """Advance Draft KV without computing unused LM-head logits."""
        assert self.draft_model is not None
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            phase = "prefill" if is_prefill else "decode_eager"
            with nvtx_range(f"nano.speculative.draft_sync_{phase}"):
                self.draft_model(input_ids, positions)
            return
        self._run_decode_graph(
            input_ids,
            positions,
            get_context(),
            self.draft_graphs,
            self.draft_graph_vars,
            self.draft_model,
            "draft_sync",
            compute_logits=False,
        )

    def _run_decode_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context,
        graphs: dict[int, torch.cuda.CUDAGraph],
        graph_vars: dict[str, torch.Tensor],
        model: Qwen3ForCausalLM,
        graph_name: str,
        compute_logits: bool = True,
    ):
        batch_size = input_ids.size(0)
        graph_batch_size = next(
            (size for size in self.graph_bs if size >= batch_size),
            None,
        )
        if graph_batch_size is None:
            raise ValueError(f"No CUDA graph bucket for batch size {batch_size}")

        graph_vars["input_ids"][:batch_size] = input_ids
        graph_vars["positions"][:batch_size] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:batch_size] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:batch_size] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][
            :batch_size, :context.block_tables.size(1)
        ] = context.block_tables

        with nvtx_range(f"nano.model.{graph_name}_decode_cudagraph_replay"):
            graphs[graph_batch_size].replay()
        if not compute_logits:
            return None
        with nvtx_range(f"nano.model.{graph_name}_lm_head"):
            return model.compute_logits(graph_vars["outputs"][:batch_size])

    def _get_verify_graph_key(
        self,
        num_tokens: int,
        num_speculative_tokens: int,
    ) -> tuple[int, int] | None:
        """Return the captured ``(K, batch bucket)`` for this verify call."""
        if num_speculative_tokens <= 0:
            return None
        verify_width = num_speculative_tokens + 1
        if num_tokens % verify_width:
            return None
        batch_size = num_tokens // verify_width
        graph_batch_size = next(
            (size for size in self.graph_bs if size >= batch_size),
            None,
        )
        if graph_batch_size is None:
            return None
        key = (num_speculative_tokens, graph_batch_size)
        return key if key in self.verify_graphs else None

    def _run_verify_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context,
        graph_key: tuple[int, int] | None = None,
    ):
        verify_width = context.num_speculative_tokens + 1
        num_tokens = input_ids.size(0)
        if num_tokens % verify_width:
            raise ValueError("Verify token count must be divisible by K+1")
        batch_size = num_tokens // verify_width
        if graph_key is None:
            graph_key = self._get_verify_graph_key(
                num_tokens,
                context.num_speculative_tokens,
            )
        if graph_key is None:
            raise ValueError(
                "No verify CUDA graph for "
                f"K={context.num_speculative_tokens}, batch_size={batch_size}"
            )

        graph_vars = self.verify_graph_vars
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:batch_size] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][
            :batch_size, :context.block_tables.size(1)
        ] = context.block_tables

        with nvtx_range("nano.model.verify_cudagraph_replay"):
            self.verify_graphs[graph_key].replay()
        with nvtx_range("nano.model.verify_lm_head"):
            return self.model.compute_logits(
                graph_vars["outputs"][:num_tokens]
            )
            hidden_states = self.draft_model(input_ids, positions)
            return self.draft_model.compute_logits(hidden_states)

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        num_speculative_tokens: int | None = None,
    ) -> list[int] | list[list[int]] | None:
        try:
            return self._run(seqs, is_prefill, num_speculative_tokens)
        finally:
            reset_context()

    def _run_target_decode(
        self,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
    ) -> list[int] | None:
        with nvtx_range("nano.prepare.decode"):
            input_ids, positions = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, positions, False)
        if self.rank == 0:
            with nvtx_range("nano.sampler.gpu"):
                sampled = self.sampler(logits, temperatures)
            with nvtx_range("nano.sampler.to_cpu"):
                token_ids = sampled.tolist()
        else:
            token_ids = None
        del logits
        reset_context()
        return token_ids

    def _sync_draft_kv(self, seqs: list[Sequence]) -> None:
        assert self.draft_model is not None
        for seq in seqs:
            if seq.draft_kv_len > len(seq) - 1:
                raise RuntimeError("draft KV advanced past the committed boundary")

        with nvtx_range("nano.speculative.k0_draft_sync"):
            needs_prefill = any(
                seq.draft_kv_len < len(seq) - 1 for seq in seqs
            )
            if needs_prefill:
                input_ids, positions, _ = self.prepare_draft_prefill(seqs)
            else:
                input_ids, positions = self.prepare_draft_decode(seqs)
            self.run_draft_kv_only(input_ids, positions, needs_prefill)
            reset_context()
        for seq in seqs:
            seq.draft_kv_len = len(seq)

    def _run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        num_speculative_tokens: int | None = None,
    ) -> list[int] | list[list[int]] | None:
        active_k = (
            self.max_num_speculative_tokens
            if num_speculative_tokens is None
            else num_speculative_tokens
        )
        if not 0 <= active_k <= self.max_num_speculative_tokens:
            raise ValueError(
                f"active K must be in [0, {self.max_num_speculative_tokens}]"
            )
        if not seqs:
            return [] if self.rank == 0 else None
        if not is_prefill and self.draft_model is not None:
            temperatures = self.prepare_sample(seqs)
            if active_k == 0:
                token_ids = self._run_target_decode(seqs, temperatures)
                if not self.config.skip_draft_kv_on_k0:
                    self._sync_draft_kv(seqs)
                return token_ids
            return self.run_speculative_decode(seqs, temperatures, active_k)

        phase = "prefill" if is_prefill else "decode"
        with nvtx_range(f"nano.prepare.{phase}"):
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            with nvtx_range("nano.sampler.gpu"):
                sampled = self.sampler(logits, temperatures)
            with nvtx_range("nano.sampler.to_cpu"):
                token_ids = sampled.tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_speculative_decode(
        self,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
        num_draft_tokens: int,
    ) -> list[list[int]] | None:
        if not seqs:
            return [] if self.rank == 0 else None
        original_state = [(len(seq), seq.draft_kv_len) for seq in seqs]
        try:
            return self._run_speculative_decode(seqs, temperatures, num_draft_tokens)
        except Exception:
            # A failed forward or sampler must not leave unverified proposals
            # in the sequence. KV beyond the restored lengths is overwritten.
            for seq, (length, draft_kv_len) in zip(seqs, original_state):
                num_extra = len(seq) - length
                if num_extra > 0:
                    seq.pop_last_n_tokens(num_extra)
                seq.draft_kv_len = draft_kv_len
            raise
        finally:
            reset_context()

    def _run_speculative_decode(
        self,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
        num_draft_tokens: int,
    ) -> list[list[int]] | None:
        if not 1 <= num_draft_tokens <= self.max_num_speculative_tokens:
            raise ValueError("speculative decode requires 1 <= K <= max K")
        batch_size = len(seqs)
        vocab_size = self.config.hf_config.vocab_size
        device = temperatures.device
        draft_token_ids = torch.empty(
            batch_size,
            num_draft_tokens,
            dtype=torch.int64,
            device=device,
        )
        draft_probs = torch.empty(
            batch_size,
            num_draft_tokens,
            vocab_size,
            dtype=torch.float32,
            device=device,
        ) if self.rank == 0 else None

        with nvtx_range("nano.speculative.draft"):
            for step in range(num_draft_tokens):
                needs_prefill = step == 0 and any(seq.draft_kv_len < len(seq) - 1 for seq in seqs)
                if needs_prefill:
                    input_ids, positions, sample_indices = self.prepare_draft_prefill(seqs)
                    logits = self.run_draft_model(input_ids, positions, True)
                    if self.rank == 0:
                        # Upstream ParallelLMHead already selects the last row
                        # per sequence during prefill. Also accept a head that
                        # returns logits for every input token.
                        if logits.size(0) == input_ids.size(0):
                            logits = logits[sample_indices]
                        elif logits.size(0) != batch_size:
                            raise ValueError("Unexpected draft prefill logits shape")
                else:
                    input_ids, positions = self.prepare_draft_decode(seqs)
                    logits = self.run_draft_model(input_ids, positions, False)
                reset_context()

                if self.rank == 0:
                    tokens, probs = self.sampler.sample_with_probs(logits, temperatures)
                    draft_probs[:, step] = probs
                    del probs
                else:
                    tokens = torch.empty(batch_size, dtype=torch.int64, device=device)
                del logits
                if self.world_size > 1:
                    # Only rank 0 has gathered logits and makes random draws.
                    dist.broadcast(tokens, src=0)
                draft_token_ids[:, step] = tokens
                for sequence, token_id in zip(seqs, tokens.tolist()):
                    sequence.draft_kv_len = len(sequence)
                    sequence.append_token(token_id)

        with nvtx_range("nano.speculative.verify"):
            input_ids, positions = self.prepare_verify(seqs, num_draft_tokens)
            logits = self.run_model(input_ids, positions, False)
            reset_context()
            if self.rank == 0:
                if logits.size(0) != batch_size * (num_draft_tokens + 1):
                    raise ValueError("Target verification must return logits for every verification token")
                expanded_temperatures = temperatures.repeat_interleave(num_draft_tokens + 1)
                target_probs = self.sampler.compute_probs(logits, expanded_temperatures)
                target_probs = target_probs.reshape(batch_size, num_draft_tokens + 1, vocab_size)
            del logits

        with nvtx_range("nano.speculative.accept"):
            if self.rank == 0:
                sample = rejection_sample(target_probs, draft_probs, draft_token_ids)
                decisions = torch.stack((sample.num_accepted, sample.next_token_ids), dim=1).to(
                    device=device, dtype=torch.int64
                )
            else:
                decisions = torch.empty(batch_size, 2, dtype=torch.int64, device=device)
            if self.world_size > 1:
                dist.broadcast(decisions, src=0)

        results = []
        accepted_counts = decisions[:, 0].tolist()
        correction_ids = decisions[:, 1].tolist()
        for sequence, proposed, num_accepted, correction_id in zip(
            seqs,
            draft_token_ids.tolist(),
            accepted_counts,
            correction_ids,
        ):
            base_length = len(sequence) - num_draft_tokens
            num_rejected = num_draft_tokens - num_accepted
            if num_rejected:
                sequence.pop_last_n_tokens(num_rejected)

            # The correction/bonus token has not passed through the draft model.
            valid_draft_kv_len = base_length + num_accepted
            sequence.draft_kv_len = min(sequence.draft_kv_len, valid_draft_kv_len)
            sequence.append_token(correction_id)
            results.append(proposed[:num_accepted] + [correction_id])

        return results if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        if max_bs < 1:
            raise ValueError("max_num_seqs must be positive")
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted(
            {bs for bs in (1, 2, 4, 8) if bs <= max_bs}
            | set(range(16, max_bs + 1, 16))
            | {max_bs}
        )
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        assert self.draft_model is not None
        assert self.draft_hf_config is not None
        config = self.config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
        )
        outputs = torch.zeros(max_bs, self.draft_hf_config.hidden_size)
        self.draft_graphs = {}
        self.draft_graph_pool = None

        for batch_size in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.draft_model(
                input_ids[:batch_size],
                positions[:batch_size],
            )
            with torch.cuda.graph(graph, self.draft_graph_pool):
                outputs[:batch_size] = self.draft_model(
                    input_ids[:batch_size],
                    positions[:batch_size],
                )
            if self.draft_graph_pool is None:
                self.draft_graph_pool = graph.pool()
            self.draft_graphs[batch_size] = graph
            torch.cuda.synchronize()
            reset_context()

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_tokens = max_bs * (self.max_num_speculative_tokens + 1)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
        )
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)
        self.verify_graphs = {}
        self.verify_graph_pool = None

        dynamic_schedule = getattr(
            config,
            "num_speculative_tokens_per_batch_size",
            None,
        )
        if dynamic_schedule is None:
            verify_graph_specs = {
                (self.max_num_speculative_tokens, batch_size)
                for batch_size in self.graph_bs
            }
        else:
            dynamic_lookup = build_dynamic_speculative_lookup(
                dynamic_schedule,
                max_bs,
                self.max_num_speculative_tokens,
            )
            verify_graph_specs = set()
            for active_batch_size in range(1, max_bs + 1):
                active_k = dynamic_lookup[active_batch_size]
                if active_k == 0:
                    continue
                graph_batch_size = next(
                    size
                    for size in self.graph_bs
                    if size >= active_batch_size
                )
                verify_graph_specs.add((active_k, graph_batch_size))

        # Capture the largest graph first so all variants can share its pool.
        ordered_specs = sorted(
            verify_graph_specs,
            key=lambda spec: spec[1] * (spec[0] + 1),
            reverse=True,
        )
        for active_k, batch_size in ordered_specs:
            verify_width = active_k + 1
            num_tokens = batch_size * verify_width
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                num_speculative_tokens=active_k,
            )
            outputs[:num_tokens] = self.model(
                input_ids[:num_tokens],
                positions[:num_tokens],
            )
            with torch.cuda.graph(graph, self.verify_graph_pool):
                outputs[:num_tokens] = self.model(
                    input_ids[:num_tokens],
                    positions[:num_tokens],
                )
            if self.verify_graph_pool is None:
                self.verify_graph_pool = graph.pool()
            self.verify_graphs[(active_k, batch_size)] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
