"""Shared interface and INT4 helpers for weight-only (W4A16) quantization.

Adapted from Brassinai/nano-vllm@e909386a (nanovllm/quantization/base.py).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
import triton

PACK_FACTOR = 8
_SHIFTS = tuple(range(0, 32, 4))


def unpack_int4(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand int32 words along ``dim`` into eight values in nibble order."""
    x = packed.movedim(dim, -1).unsqueeze(-1)
    shifts = torch.tensor(_SHIFTS, dtype=torch.int32, device=packed.device)
    values = (x >> shifts) & 0xF
    return values.flatten(-2).movedim(-1, dim)


def pack_int4(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack groups of eight 0..15 values along ``dim`` into int32 words."""
    x = values.movedim(dim, -1).to(torch.int64).unflatten(-1, (-1, PACK_FACTOR))
    shifts = torch.tensor(_SHIFTS, dtype=torch.int64, device=values.device)
    packed = (x << shifts).sum(-1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32).movedim(-1, dim)


def check_group_size(group_size: int) -> int:
    if group_size != -1 and group_size <= 0:
        raise ValueError(f"invalid quantization group_size {group_size}")
    return group_size


@dataclass(frozen=True)
class WNA16Config(ABC):
    """A 4-bit checkpoint format with ``qweight``, ``qzeros`` and ``scales`` per linear layer."""

    method: ClassVar[str]
    group_size: int

    @classmethod
    @abstractmethod
    def from_config(cls, raw: dict) -> "WNA16Config":
        raise NotImplementedError

    @abstractmethod
    def qweight_shape(self, input_size: int, output_size: int) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def unpack_qweight(self, qweight: torch.Tensor) -> torch.Tensor:
        """Return the ``[input_size, output_size]`` integer weight."""
        raise NotImplementedError

    @abstractmethod
    def unpack_qzeros(self, qzeros: torch.Tensor) -> torch.Tensor:
        """Return the ``[num_groups, output_size]`` zero points."""
        raise NotImplementedError

    @abstractmethod
    def matmul(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def packs_output_dim(self, name: str) -> bool:
        """Whether checkpoint tensor ``name`` packs eight output channels per int32."""
        return name == "qzeros"

    def convert_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def validate_g_idx(self, g_idx: torch.Tensor, input_size: int):
        raise ValueError(f"{self.method} checkpoints do not contain g_idx tensors")

    def effective_group_size(self, input_size: int) -> int:
        return input_size if self.group_size == -1 else self.group_size

    def num_groups(self, input_size: int) -> int:
        return -(-input_size // self.effective_group_size(input_size))

    def dequantize(self, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Return the dense ``[output_size, input_size]`` weight in ``scales.dtype``."""
        q = self.unpack_qweight(qweight)
        input_size = q.shape[0]
        groups = torch.arange(input_size, device=q.device) // self.effective_group_size(input_size)
        weight = (q - self.unpack_qzeros(qzeros)[groups]).float() * scales[groups].float()
        return weight.t().to(scales.dtype)

    def linear(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            return F.linear(x, self.dequantize(qweight, qzeros, scales).to(x.dtype), bias)
        if x.dtype != torch.float16:
            raise TypeError(f"{self.method} linear expects float16 activations, got {x.dtype}")
        return self.matmul(x, qweight, qzeros, scales, bias)


def launch_wna16_kernel(
    kernel,
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    **constexprs,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    m, k = x_2d.shape
    n = scales.shape[1]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m = min(max(triton.next_power_of_2(m), 16), 64)
    block_n = 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    kernel[grid](
        x_2d,
        qweight,
        qzeros,
        scales,
        bias if bias is not None else out,
        out,
        m,
        n,
        k,
        group_size,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
        **constexprs,
    )
    return out.reshape(x.shape[:-1] + (n,))
