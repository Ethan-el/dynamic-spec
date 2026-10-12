"""AutoAWQ GEMM checkpoints: 4-bit weights packed along the output dimension.

Adapted from Brassinai/nano-vllm@e909386a (nanovllm/quantization/awq.py).
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from nanovllm.quantization.base import (
    PACK_FACTOR,
    WNA16Config,
    check_group_size,
    launch_wna16_kernel,
    pack_int4,
    unpack_int4,
)

# AutoAWQ stores logical column AWQ_PACK_ORDER[i] in nibble i of each int32.
AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_AWQ_UNPACK_ORDER = [AWQ_PACK_ORDER.index(i) for i in range(PACK_FACTOR)]


def awq_to_natural(packed: torch.Tensor) -> torch.Tensor:
    """Reorder AutoAWQ-packed columns (dim 1) to natural nibble order."""
    nibbles = unpack_int4(packed, 1).unflatten(1, (-1, PACK_FACTOR))
    return pack_int4(nibbles[..., _AWQ_UNPACK_ORDER].flatten(1), 1)


@triton.jit
def _awq_matmul_kernel(
    x_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    bias_ptr,
    out_ptr,
    m,
    n,
    k,
    group_size,
    stride_xm,
    stride_xk,
    stride_qwk,
    stride_qwn,
    stride_qzg,
    stride_qzn,
    stride_sg,
    stride_sn,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_group) % group_size_m
    pid_n = (pid % num_pid_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < m
    n_mask = offs_n < n
    n_shifts = (offs_n % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        offs = k_start + offs_k
        k_mask = offs < k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        packed = tl.load(
            qweight_ptr + offs[:, None] * stride_qwk + (offs_n[None, :] // 8) * stride_qwn,
            mask=w_mask,
            other=0,
        )
        q = (packed >> n_shifts[None, :]) & 0xF
        groups = offs // group_size
        packed_zeros = tl.load(
            qzeros_ptr + groups[:, None] * stride_qzg + (offs_n[None, :] // 8) * stride_qzn,
            mask=w_mask,
            other=0,
        )
        zeros = (packed_zeros >> n_shifts[None, :]) & 0xF
        scales = tl.load(
            scales_ptr + groups[:, None] * stride_sg + offs_n[None, :] * stride_sn,
            mask=w_mask,
            other=0.0,
        )
        w = ((q - zeros).to(tl.float32) * scales.to(tl.float32)).to(tl.float16)
        acc += tl.dot(x, w)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@dataclass(frozen=True)
class AWQConfig(WNA16Config):
    method = "awq"

    @classmethod
    def from_config(cls, raw: dict) -> "AWQConfig":
        bits = int(raw.get("bits", raw.get("w_bit", 0)))
        group_size = int(raw.get("group_size", raw.get("q_group_size", 0)))
        version = str(raw.get("version") or "gemm").lower()
        if bits != 4:
            raise NotImplementedError(f"only 4-bit AWQ checkpoints are supported, got bits={bits}")
        if version != "gemm":
            raise NotImplementedError(f"only AWQ GEMM checkpoints are supported, got version={version!r}")
        if not raw.get("zero_point", True):
            raise NotImplementedError("AWQ checkpoints without zero points are not supported")
        if raw.get("modules_to_not_convert"):
            raise NotImplementedError("partially quantized AWQ checkpoints are not supported")
        return cls(check_group_size(group_size))

    def qweight_shape(self, input_size: int, output_size: int) -> tuple[int, int]:
        if output_size % PACK_FACTOR:
            raise ValueError(f"AWQ output size {output_size} is not divisible by {PACK_FACTOR}")
        return input_size, output_size // PACK_FACTOR

    def packs_output_dim(self, name: str) -> bool:
        return name in ("qweight", "qzeros")

    def convert_checkpoint_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return awq_to_natural(tensor) if name in ("qweight", "qzeros") else tensor

    def unpack_qweight(self, qweight: torch.Tensor) -> torch.Tensor:
        return unpack_int4(qweight, 1)

    def unpack_qzeros(self, qzeros: torch.Tensor) -> torch.Tensor:
        return unpack_int4(qzeros, 1)

    def matmul(self, x, qweight, qzeros, scales, bias):
        return launch_wna16_kernel(
            _awq_matmul_kernel,
            x,
            qweight,
            qzeros,
            scales,
            bias,
            self.effective_group_size(x.shape[-1]),
        )
