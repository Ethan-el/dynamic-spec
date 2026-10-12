import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig

from nanovllm.fp8 import Fp8Config
from nanovllm.quantization import QuantConfig, WNA16Config, resolve_quant_config
from nanovllm.dynamic_speculative import (
    validate_and_normalize_dynamic_speculative_schedule,
)


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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.quant_config = resolve_quant_config(self.hf_config, self.model, self.quantization)
        if self.quant_config is None:
            self.quantization = None
        elif isinstance(self.quant_config, Fp8Config):
            self.quantization = "fp8"
        else:
            self.quantization = self.quant_config.method
        if self.quant_config is not None and self.tensor_parallel_size != 1:
            raise NotImplementedError(f"{self.quantization} only supports tensor_parallel_size=1")
        is_wna16 = isinstance(self.quant_config, WNA16Config)
        if is_wna16 and self.speculative_model is not None:
            raise NotImplementedError(
                f"{self.quantization} models run without speculative decoding; unset speculative_model"
            )
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
