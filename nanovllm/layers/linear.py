import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.fp8 import Fp8Config, fp8_linear


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: Fp8Config | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_config = quant_config
        if quant_config is None:
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
        return fp8_linear(x, self.weight, self.weight_scale_inv, bias, self.quant_config)

    def bias_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Fp8Config | None = None,
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
        quant_config: Fp8Config | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config)

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
        quant_config: Fp8Config | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quant_config)

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

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Fp8Config | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config)

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
        quant_config: Fp8Config | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quant_config)

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
