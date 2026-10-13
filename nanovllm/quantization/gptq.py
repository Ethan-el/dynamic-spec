"""GPTQ checkpoints: 4-bit weights packed along the input dimension.

Adapted from Brassinai/nano-vllm@e909386a (nanovllm/quantization/gptq.py).
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
    unpack_int4,
)


@triton.jit
def _gptq_matmul_kernel(
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
    ZERO_OFFSET: tl.constexpr,
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
            qweight_ptr + (offs[:, None] // 8) * stride_qwk + offs_n[None, :] * stride_qwn,
            mask=w_mask,
            other=0,
        )
        q = (packed >> ((offs[:, None] % 8) * 4)) & 0xF
        groups = offs // group_size
        packed_zeros = tl.load(
            qzeros_ptr + groups[:, None] * stride_qzg + (offs_n[None, :] // 8) * stride_qzn,
            mask=w_mask,
            other=0,
        )
        zeros = (((packed_zeros >> n_shifts[None, :]) & 0xF) + ZERO_OFFSET) & 0xF
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
class GPTQConfig(WNA16Config):
    method = "gptq"
    # Legacy GPTQ stores zero - 1; gptq_v2 stores the actual zero point.
    zero_offset: int = 1

    @classmethod
    def from_config(cls, raw: dict) -> "GPTQConfig":
        bits = int(raw.get("bits", 0))
        group_size = int(raw.get("group_size", 0))
        checkpoint_format = str(raw.get("checkpoint_format") or raw.get("format") or "gptq").lower()
        if bits != 4:
            raise NotImplementedError(f"only 4-bit GPTQ checkpoints are supported, got bits={bits}")
        if raw.get("desc_act", False):
            raise NotImplementedError("GPTQ desc_act=True (activation order) is not supported")
        if raw.get("lm_head", False):
            raise NotImplementedError("GPTQ checkpoints with a quantized lm_head are not supported")
        if raw.get("dynamic"):
            raise NotImplementedError("GPTQ per-module dynamic overrides are not supported")
        if checkpoint_format not in ("gptq", "gptq_v2"):
            raise NotImplementedError(f"unsupported GPTQ checkpoint_format {checkpoint_format!r}")
        return cls(check_group_size(group_size), 0 if checkpoint_format == "gptq_v2" else 1)

    def qweight_shape(self, input_size: int, output_size: int) -> tuple[int, int]:
        if input_size % PACK_FACTOR or output_size % PACK_FACTOR:
            raise ValueError(f"GPTQ linear shape {(output_size, input_size)} is not divisible by {PACK_FACTOR}")
        return input_size // PACK_FACTOR, output_size

    def validate_g_idx(self, g_idx: torch.Tensor, input_size: int):
        expected = torch.arange(input_size, dtype=torch.int64) // self.effective_group_size(input_size)
        if g_idx.shape != (input_size,) or not torch.equal(g_idx.to(torch.int64), expected):
            raise NotImplementedError("GPTQ g_idx is not sequential; activation-order checkpoints are not supported")

    def unpack_qweight(self, qweight: torch.Tensor) -> torch.Tensor:
        return unpack_int4(qweight, 0)

    def unpack_qzeros(self, qzeros: torch.Tensor) -> torch.Tensor:
        # Wraps like AutoGPTQ's dequantizer: a legacy stored 15 decodes to zero point 0.
        return (unpack_int4(qzeros, 1) + self.zero_offset) & 0xF

    def matmul(self, x, qweight, qzeros, scales, bias):
        return launch_wna16_kernel(
            _gptq_matmul_kernel,
            x,
            qweight,
            qzeros,
            scales,
            bias,
            self.effective_group_size(x.shape[-1]),
            ZERO_OFFSET=self.zero_offset,
        )
