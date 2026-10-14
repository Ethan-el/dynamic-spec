import json
import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig

from nanovllm.fp8 import Fp8Config
from nanovllm.quantization import AWQConfig, QuantConfig, WNA16Config, resolve_quant_config
from nanovllm.dynamic_speculative import (
    validate_and_normalize_dynamic_speculative_schedule,
)

_DSPARK_CONFIG_KEYS = (
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "markov_rank",
    "enable_confidence_head",
    "confidence_head_with_markov",
)


def load_dspark_hf_config(model_path: str) -> AutoConfig:
    """Load a DSpark draft config, including fields AutoConfig does not model."""
    hf_config = AutoConfig.from_pretrained(model_path)
    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)
    for key in _DSPARK_CONFIG_KEYS:
        if key in raw:
            setattr(hf_config, key, raw[key])
    return hf_config


@dataclass(slots=True)
class SpeculativeConfig:
    method: str
    model: str
    num_speculative_tokens: int

    @classmethod
    def from_dict(cls, config: dict) -> "SpeculativeConfig":
        method = str(config.get("method", "")).lower()
        if method != "dspark":
            raise ValueError("speculative_config method must be 'dspark'")
        model = str(config.get("model", ""))
        if not os.path.isdir(model):
            raise ValueError(f"speculative model path does not exist: {model}")
        num_speculative_tokens = int(config.get("num_speculative_tokens", 0))
        if num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive")
        return cls(method=method, model=model, num_speculative_tokens=num_speculative_tokens)


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: str | None = None
    awq_checkpoint_format: str | None = None
    hf_config: AutoConfig | None = None
    quant_config: QuantConfig | None = None
    dtype: torch.dtype | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_model: str | None = None
    num_speculative_tokens: int = 0
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
    skip_draft_kv_on_k0: bool = False
    num_draft_kvcache_blocks: int = -1
    speculative_config: dict | SpeculativeConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.quant_config = resolve_quant_config(self.hf_config, self.model, self.quantization, self.awq_checkpoint_format)
        if self.quant_config is None:
            self.quantization = None
        elif isinstance(self.quant_config, Fp8Config):
            self.quantization = "fp8"
        else:
            self.quantization = self.quant_config.method
        if self.quant_config is not None and self.tensor_parallel_size != 1 and not isinstance(self.quant_config, AWQConfig):
            raise NotImplementedError(f"{self.quantization} only supports tensor_parallel_size=1")
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig.from_dict(self.speculative_config)
        if self.speculative_model is not None and self.speculative_config is not None:
            raise ValueError("speculative_model and speculative_config cannot both be set")
        is_wna16 = isinstance(self.quant_config, WNA16Config)
        if is_wna16 and (self.speculative_model is not None or self.speculative_config is not None):
            raise NotImplementedError(
                f"{self.quantization} models run without speculative decoding; unset speculative decoding"
            )
        if self.speculative_config is not None:
            if self.num_speculative_tokens_per_batch_size is not None:
                raise ValueError("DSpark uses a fixed block size and does not take a dynamic K schedule")
            self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
            self._validate_dspark()
        # The W4A16 kernel computes in float16 regardless of the checkpoint's recorded dtype.
        hf_dtype = getattr(self.hf_config, "dtype", None) or self.hf_config.torch_dtype
        self.dtype = torch.float16 if is_wna16 else hf_dtype
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_model is not None:
            assert os.path.isdir(self.speculative_model)
            assert self.num_speculative_tokens > 0
            if self.num_speculative_tokens_per_batch_size is not None:
                self.num_speculative_tokens_per_batch_size = (
                    validate_and_normalize_dynamic_speculative_schedule(
                        self.num_speculative_tokens_per_batch_size
                    )
                )
        #     assert self.tensor_parallel_size == 1, "speculative decoding MVP only supports tensor_parallel_size=1"
        #     assert self.enforce_eager, "speculative decoding MVP requires enforce_eager=True"
        # else:
        #     assert self.num_speculative_tokens == 0

    def _validate_dspark(self):
        draft = load_dspark_hf_config(self.speculative_config.model)
        target_layers = tuple(getattr(draft, "target_layer_ids", ()) or ())
        if not target_layers:
            raise ValueError("DSpark draft config must define target_layer_ids")
        if any(layer_id < 0 or layer_id >= self.hf_config.num_hidden_layers for layer_id in target_layers):
            raise ValueError("DSpark target_layer_ids fall outside the target model")
        block_size = getattr(draft, "block_size", None)
        if block_size != self.num_speculative_tokens:
            raise ValueError(
                f"DSpark block_size ({block_size}) must equal num_speculative_tokens ({self.num_speculative_tokens})"
            )
        if draft.vocab_size != self.hf_config.vocab_size:
            raise ValueError("target and DSpark draft vocab sizes must match")
        if getattr(draft, "mask_token_id", None) is None or getattr(draft, "markov_rank", None) is None:
            raise ValueError("DSpark draft config must define mask_token_id and markov_rank")
