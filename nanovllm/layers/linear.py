import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.fp8 import fp8_linear
from nanovllm.quantization import AWQConfig, QuantConfig
from nanovllm.quantization.base import PACK_FACTOR, WNA16Config

WNA16_PARAMS = ("qweight", "qzeros", "scales")


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    shard_ids: tuple = (None,)

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size = input_size
        self.output_size = output_size
        if isinstance(quant_config, AWQConfig) and not quant_config.is_layer_quantized(prefix):
            quant_config = None
        self.quant_config = quant_config
        if isinstance(quant_config, WNA16Config):
            self.create_wna16_weights(quant_config)
        elif quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.register_parameter("weight_scale_inv", None)
        else:
            block_n, block_k = quant_config.block_size
            if output_size % block_n or input_size % block_k:
                raise ValueError(
                    f"FP8 linear shape {(output_size, input_size)} is not divisible by "
                    f"block size {quant_config.block_size}"
                )
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(output_size // block_n, input_size // block_k, dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self.weight_scale_loader
        if not isinstance(quant_config, WNA16Config):
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size), requires_grad=quant_config is None)
            self.bias.weight_loader = self.bias_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_linear(self, x: torch.Tensor, bias: torch.Tensor | None):
        if self.quant_config is None:
            return F.linear(x, self.weight, bias)
        if isinstance(self.quant_config, WNA16Config):
            return self.quant_config.linear(x, self.qweight, self.qzeros, self.scales, bias)
        return fp8_linear(x, self.weight, self.weight_scale_inv, bias, self.quant_config)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        offset, size = self.output_shard(loaded_shard_id)
        if self.tp_dim == 0:
            loaded_weight = loaded_weight.chunk(self.tp_size)[self.tp_rank]
        param.data.narrow(0, offset, size).copy_(loaded_weight)

    def output_shard(self, loaded_shard_id) -> tuple[int, int]:
        """Return ``(offset, size)`` of a checkpoint shard in this layer's output dimension."""
        return 0, self.output_size

    def create_wna16_weights(self, config: WNA16Config):
        if self.tp_size != 1 and not isinstance(config, AWQConfig):
            raise NotImplementedError(f"{config.method} only supports tensor_parallel_size=1")
        if isinstance(config, AWQConfig) and self.tp_size > 1 and self.tp_dim == 1:
            if config.group_size != -1 and self.input_size % config.group_size:
                raise ValueError(
                    f"AWQ row-parallel input shard {self.input_size} must align with group_size {config.group_size}"
                )
        k, n = self.input_size, self.output_size
        num_groups = config.num_groups(k)
        shapes = {
            "qweight": (config.qweight_shape(k, n), torch.int32),
            "qzeros": ((num_groups, n // PACK_FACTOR), torch.int32),
            "scales": ((num_groups, n), torch.get_default_dtype()),
        }
        for name, (shape, dtype) in shapes.items():
            param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            param.wna16_name = name
            param.weight_loader = self.wna16_loader
            self.register_parameter(name, param)
        self.loaded_wna16_shards = set()

    def wna16_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
        name = param.wna16_name
        offset, size = (0, self.output_size) if loaded_shard_id is None else self.output_shard(loaded_shard_id)
        if self.quant_config.packs_output_dim(name):
            if offset % PACK_FACTOR or size % PACK_FACTOR:
                raise ValueError(f"shard {loaded_shard_id!r} width {size} is not divisible by {PACK_FACTOR}")
            offset, size = offset // PACK_FACTOR, size // PACK_FACTOR
        loaded_weight = self.quant_config.convert_checkpoint_tensor(name, loaded_weight)
        target = param.data.narrow(1, offset, size)
        if self.tp_size > 1:
            if self.tp_dim == 0:
                expected = (target.size(0), target.size(1) * self.tp_size)
                if loaded_weight.shape != expected:
                    raise ValueError(f"{name} shard {loaded_shard_id!r}: checkpoint shape {tuple(loaded_weight.shape)} does not match expected {expected}")
                if loaded_shard_id is None and self.shard_ids != (None,):
                    pieces = []
                    for shard_id in self.shard_ids:
                        shard_offset, shard_size = self.output_shard(shard_id)
                        if self.quant_config.packs_output_dim(name):
                            shard_offset, shard_size = shard_offset // PACK_FACTOR, shard_size // PACK_FACTOR
                        pieces.append(loaded_weight.narrow(1, shard_offset * self.tp_size + self.tp_rank * shard_size, shard_size))
                    loaded_weight = torch.cat(pieces, dim=1)
                else:
                    loaded_weight = loaded_weight.narrow(1, self.tp_rank * target.size(1), target.size(1))
            elif self.tp_dim == 1:
                replicate_group = name != "qweight" and self.quant_config.group_size == -1
                expected_rows = target.size(0) if replicate_group else target.size(0) * self.tp_size
                expected = (expected_rows, target.size(1))
                if loaded_weight.shape != expected:
                    raise ValueError(f"{name} shard {loaded_shard_id!r}: checkpoint shape {tuple(loaded_weight.shape)} does not match expected {expected}")
                if not replicate_group:
                    loaded_weight = loaded_weight.narrow(0, self.tp_rank * target.size(0), target.size(0))
        if loaded_weight.shape != target.shape:
            raise ValueError(
                f"{name} shard {loaded_shard_id!r}: checkpoint shape {tuple(loaded_weight.shape)} "
                f"does not match expected {tuple(target.shape)}"
            )
        key = (name, loaded_shard_id)
        if key in self.loaded_wna16_shards or (name, None) in self.loaded_wna16_shards or (
            loaded_shard_id is None and any(loaded_name == name for loaded_name, _ in self.loaded_wna16_shards)
        ):
            raise ValueError(f"{name} shard {loaded_shard_id!r} was loaded twice")
        target.copy_(loaded_weight)
        self.loaded_wna16_shards.add(key)

    def check_wna16_loaded(self, prefix: str):
        expected = {
            (name, shard_id) for name in WNA16_PARAMS for shard_id in self.shard_ids
            if (name, None) not in self.loaded_wna16_shards
        }
        missing = expected - self.loaded_wna16_shards
        if missing:
            raise ValueError(f"{prefix}: checkpoint is missing quantized tensors {sorted(map(str, missing))}")


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config, prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        self.shard_ids = tuple(range(len(output_sizes)))
        super().__init__(input_size, sum(output_sizes), bias, quant_config, prefix)

    def output_shard(self, loaded_shard_id: int) -> tuple[int, int]:
        offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        return offset, self.output_sizes[loaded_shard_id] // self.tp_size

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        block_n = self.quant_config.block_size[0]
        sizes = [size // block_n for size in self.output_sizes]
        shard_offset = sum(sizes[:loaded_shard_id]) // self.tp_size
        shard_size = sizes[loaded_shard_id] // self.tp_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    shard_ids = ("q", "k", "v")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config, prefix)

    def output_shard(self, loaded_shard_id: str) -> tuple[int, int]:
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        return {"q": (0, q_size), "k": (q_size, kv_size), "v": (q_size + kv_size, kv_size)}[loaded_shard_id]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        assert loaded_shard_id in ["q", "k", "v"]
        block_n = self.quant_config.block_size[0]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size // block_n
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size // block_n
            shard_offset = self.num_heads * self.head_size // block_n
        else:
            shard_size = self.num_kv_heads * self.head_size // block_n
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size // block_n
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quant_config, prefix)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def weight_scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        self.weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.apply_linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
