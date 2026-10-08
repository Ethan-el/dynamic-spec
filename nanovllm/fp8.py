from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@dataclass(frozen=True)
class Fp8Config:
    block_size: tuple[int, int]

    @classmethod
    def from_hf_config(cls, hf_config, requested: str | None = None):
        raw = getattr(hf_config, "quantization_config", None)
        if raw is None:
            if requested is not None:
                raise ValueError("quantization='fp8' requires an FP8 checkpoint")
            return None
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        raw = dict(raw)
        method = str(raw.get("quant_method", "")).lower()
        if method != "fp8":
            if requested is None:
                return None
            raise ValueError(f"checkpoint quant_method is {method!r}, not 'fp8'")
        if requested is not None and requested.lower() != "fp8":
            raise ValueError(f"unsupported quantization: {requested!r}")
        if raw.get("activation_scheme", "dynamic") != "dynamic":
            raise NotImplementedError("only dynamic FP8 activation quantization is supported")
        if raw.get("fmt", "e4m3").lower() not in ("e4m3", "e4m3fn"):
            raise NotImplementedError("only E4M3 FP8 checkpoints are supported")
        block_size = tuple(raw.get("weight_block_size") or ())
        if block_size != (128, 128):
            raise NotImplementedError("only FP8 weight_block_size=[128, 128] is supported")
        return cls(block_size)


@triton.jit
def _act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-12)
    tl.store(y_ptr + offsets, (x / scale).to(y_ptr.dtype.element_ty))
    tl.store(s_ptr + pid, scale)


def quantize_activation(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous() or x.shape[-1] % block_size:
        raise ValueError("FP8 activation input must be contiguous and block aligned")
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = x.new_empty(
        x.shape[:-1] + (x.shape[-1] // block_size,), dtype=torch.float32
    )
    _act_quant_kernel[(triton.cdiv(x.numel(), block_size),)](
        x, output, scales, BLOCK_SIZE=block_size
    )
    return output, scales


@triton.jit
def _fp8_matmul_kernel(
    a,
    b,
    c,
    a_scales,
    b_scales,
    m,
    n,
    k,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsk,
    stride_bsn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_group // group_size_m

    offsets_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % m
    offsets_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % n
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = a + offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b + offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
    as_ptrs = a_scales + offsets_am * stride_asm
    bs_ptrs = b_scales + offsets_bn // group_n * stride_bsn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(0, tl.cdiv(k, BLOCK_K)):
        mask_k = offsets_k < k - block * BLOCK_K
        lhs = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        rhs = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        scale_k = block * BLOCK_K // group_k
        lhs_scale = tl.load(as_ptrs + scale_k * stride_ask)
        rhs_scale = tl.load(bs_ptrs + scale_k * stride_bsk)
        accumulator += tl.dot(lhs, rhs) * lhs_scale[:, None] * rhs_scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    output = accumulator.to(c.dtype.element_ty)
    offsets_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=(offsets_cm[:, None] < m) & (offsets_cn[None, :] < n))


def fp8_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    block_n, block_k = block_size
    m = x.numel() // x.shape[-1]
    n, k = weight.shape
    output = x.new_empty(x.shape[:-1] + (n,), dtype=output_dtype)
    block_m = min(max(triton.next_power_of_2(m), 16), 128)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _fp8_matmul_kernel[grid](
        x,
        weight,
        output,
        input_scales,
        weight_scales,
        m,
        n,
        k,
        block_n,
        block_k,
        x.stride(-2),
        x.stride(-1),
        weight.stride(1),
        weight.stride(0),
        output.stride(-2),
        output.stride(-1),
        input_scales.stride(-2),
        input_scales.stride(-1),
        weight_scales.stride(1),
        weight_scales.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
    )
    return output


def dequantize_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    block_n, block_k = block_size
    expanded = scales.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(dtype)


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    config: Fp8Config,
) -> torch.Tensor:
    if not x.is_cuda:
        return F.linear(x, dequantize_weight(weight, scales, config.block_size, x.dtype), bias)
    output_dtype = x.dtype
    x = x.contiguous()
    qinput, input_scales = quantize_activation(x, config.block_size[1])
    output = fp8_matmul(qinput, weight, input_scales, scales, config.block_size, output_dtype)
    return output if bias is None else output + bias
