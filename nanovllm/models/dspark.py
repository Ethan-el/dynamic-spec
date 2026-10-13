"""Qwen3 DSpark draft: non-causal block attention plus a Markov head.

The confidence head is constructed only so its checkpoint weights load.
Confidence-based scheduling is not used.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from flash_attn import flash_attn_with_kvcache
from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import apply_rotary_emb, get_rope
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3MLP


@dataclass
class DSparkKVCache:
    """Paged K/V for every DSpark layer of one batch."""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor | None
    context_lens: torch.Tensor | None

    def for_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class DSparkMarkovHead(nn.Module):

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.embed(token_ids))


class DSparkConfidenceHead(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1)


class Qwen3DSparkAttention(nn.Module):
    """Attention over the prefix context and one non-causal draft block."""

    def __init__(self, config):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // self.total_num_heads
        )
        self.scaling = self.head_dim ** -0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
        )

    def _apply_rope(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_emb(hidden_states, cos, sin)

    def _split_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(hidden_states)
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def project_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, _ = target_hidden_states.shape
        _, key, value = self._split_qkv(target_hidden_states)
        key = self.k_norm(key.view(num_tokens, self.num_kv_heads, self.head_dim))
        value = value.view(num_tokens, self.num_kv_heads, self.head_dim)
        return self._apply_rope(key, context_positions), value

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: DSparkKVCache,
        layer_idx: int,
        draft_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Append this draft block and attend to the context plus the block."""
        batch_size, draft_length, _ = hidden_states.shape
        q, key, value = self._split_qkv(hidden_states)
        q = self.q_norm(q.view(batch_size, draft_length, self.num_heads, self.head_dim))
        key = self.k_norm(key.view(batch_size, draft_length, self.num_kv_heads, self.head_dim))
        value = value.view(batch_size, draft_length, self.num_kv_heads, self.head_dim)
        q = self._apply_rope(q, draft_positions)
        key = self._apply_rope(key, draft_positions)
        key_cache, value_cache = kv_cache.for_layer(layer_idx)
        output = flash_attn_with_kvcache(
            q,
            key_cache,
            value_cache,
            k=key,
            v=value,
            cache_seqlens=kv_cache.context_lens,
            block_table=kv_cache.block_table,
            softmax_scale=self.scaling,
            causal=False,
        )
        return self.o_proj(output.reshape(batch_size, draft_length, -1))


class Qwen3DSparkDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3DSparkAttention(config)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: DSparkKVCache,
        layer_idx: int,
        draft_positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_cache, layer_idx, draft_positions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Qwen3DSparkModel(nn.Module):
    """Dense Qwen3 DSpark draft. Its weights are never quantized."""

    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping

    def __init__(self, config):
        super().__init__()
        self.target_layer_ids = tuple(config.target_layer_ids)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DSparkDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = ReplicatedLinear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.markov_head = DSparkMarkovHead(config.vocab_size, config.markov_rank)
        confidence_size = config.hidden_size
        if getattr(config, "confidence_head_with_markov", False):
            confidence_size += config.markov_rank
        self.confidence_head = (
            DSparkConfidenceHead(confidence_size)
            if getattr(config, "enable_confidence_head", False)
            else None
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.embed_tokens.weight.data

    def project_target_hidden_states(self, aux_hidden_states: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(aux_hidden_states, list):
            assert len(aux_hidden_states) == len(self.target_layer_ids)
            aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
        return self.hidden_norm(self.fc(aux_hidden_states))

    def create_draft_embeddings(self, last_token_ids: torch.Tensor) -> torch.Tensor:
        draft_ids = torch.full(
            (last_token_ids.numel(), self.block_size),
            self.mask_token_id,
            device=last_token_ids.device,
            dtype=torch.long,
        )
        draft_ids[:, 0] = last_token_ids
        return self.embed_tokens(draft_ids)

    def draft(
        self,
        last_token_ids: torch.Tensor,
        kv_cache: DSparkKVCache,
        draft_positions: torch.Tensor,
    ) -> torch.Tensor:
        assert last_token_ids.ndim == 1
        hidden_states = self.create_draft_embeddings(last_token_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, kv_cache, layer_idx, draft_positions)
        return self.norm(hidden_states)

    def append_context_kv(
        self,
        target_hidden_states: torch.Tensor | list[torch.Tensor],
        context_positions: torch.Tensor,
        kv_cache: DSparkKVCache,
        slot_mapping: torch.Tensor,
    ) -> None:
        target_hidden_states = self.project_target_hidden_states(target_hidden_states)
        assert target_hidden_states.ndim == 2
        assert context_positions.shape == target_hidden_states.shape[:1]
        assert slot_mapping.shape == context_positions.shape
        for layer_idx, layer in enumerate(self.layers):
            key, value = layer.self_attn.project_context_kv(target_hidden_states, context_positions)
            key_cache, value_cache = kv_cache.for_layer(layer_idx)
            store_kvcache(key, value, key_cache, value_cache, slot_mapping)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Bypass ParallelLMHead: it keeps only the last prefill token.
        logits = F.linear(hidden_states, self.lm_head.weight)
        if dist.get_world_size() == 1:
            return logits
        gathered = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, logits)
        return torch.cat(gathered, dim=-1)

    def sample_draft_logits(
        self,
        base_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        sampler: Sampler,
        temperatures: torch.Tensor,
        greedy: bool,
    ) -> torch.Tensor:
        assert base_logits.ndim == 3
        previous_token_ids = anchor_token_ids
        tokens = []
        for logits in base_logits.unbind(dim=1):
            corrected = logits + self.markov_head.bias(previous_token_ids)
            if greedy:
                previous_token_ids = corrected.argmax(dim=-1)
            else:
                previous_token_ids, _ = sampler.sample_with_probs(corrected, temperatures)
            tokens.append(previous_token_ids)
        return torch.stack(tokens, dim=1)

    def draft_logits_with_markov_bias(
        self,
        base_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Recreate the draft distribution for an already sampled block."""
        assert base_logits.shape[:2] == draft_token_ids.shape
        previous_token_ids = anchor_token_ids
        logits = []
        for base_logits_at_position, draft_token_ids_at_position in zip(
            base_logits.unbind(dim=1),
            draft_token_ids.unbind(dim=1),
        ):
            logits.append(base_logits_at_position + self.markov_head.bias(previous_token_ids))
            previous_token_ids = draft_token_ids_at_position
        return torch.stack(logits, dim=1)
