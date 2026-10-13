"""Reference packing functions, copied from Brassinai/nano-vllm (MIT).

Source: nanovllm/quantization/awq_export.py at e909386a729e5d20113e0b67f7639d5f1b9a6c37.
Unlike AutoAWQ, these pack natural columns and the exporter stores zero - 1.
"""

import torch


def pack_awq_qweight(q_int: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [out_features, in_features] integer weights to AWQ GEMM layout."""
    pack_factor = 32 // bits
    out_features, in_features = q_int.shape
    if out_features % pack_factor:
        raise ValueError(
            f"out_features={out_features} must be divisible by "
            f"pack_factor={pack_factor}."
        )
    transposed = q_int.t().contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (in_features, out_features // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= transposed[:, i::pack_factor] << (i * bits)
    return packed


def pack_awq_qzeros(qzeros: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack [group_count, out_features] AWQ zero-points across output channels."""
    pack_factor = 32 // bits
    group_count, out_features = qzeros.shape
    if out_features % pack_factor:
        raise ValueError(
            f"out_features={out_features} must be divisible by "
            f"pack_factor={pack_factor}."
        )
    values = qzeros.contiguous().cpu().to(torch.int32)
    packed = torch.zeros(
        (group_count, out_features // pack_factor),
        dtype=torch.int32,
    )
    for i in range(pack_factor):
        packed |= values[:, i::pack_factor] << (i * bits)
    return packed
