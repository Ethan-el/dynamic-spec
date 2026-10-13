"""Honor the compile opt-out before importing PyTorch's compiler backend."""

import os
import warnings

import torch

_disabled = os.environ.get("TORCH_COMPILE_DISABLE") == "1"
if _disabled:
    warnings.warn(
        "TORCH_COMPILE_DISABLE=1: nano-vllm is using eager PyTorch for activation, "
        "RMSNorm, RoPE and sampling. Triton kernels and CUDA Graph remain enabled; "
        "latency/throughput may regress and floating-point rounding may differ.",
        RuntimeWarning,
        stacklevel=2,
    )


def compile_if_enabled(fn):
    # torch.compile still imports Inductor with the env opt-out in PyTorch 2.9.1.
    return fn if _disabled else torch.compile(fn)
