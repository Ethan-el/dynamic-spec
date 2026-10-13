"""Stand-ins for CUDA-only packages so nanovllm imports on CPU-only machines.

Kernels decorated with the stub ``triton.jit`` are never launched: every CPU code
path falls back to PyTorch before reaching them.
"""

import sys
import types

import torch
# transformers probes for triton with find_spec at import time; let it see the real state.
import transformers  # noqa: F401


def _unavailable(*args, **kwargs):
    raise RuntimeError("this operation requires a CUDA build of the package")


try:
    import triton  # noqa: F401
except ImportError:
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    # Inductor imports triton when torch.compile decorates a function.
    torch.compile = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)

try:
    import flash_attn  # noqa: F401
except ImportError:
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_varlen_func = _unavailable
    flash_attn.flash_attn_with_kvcache = _unavailable
    sys.modules["flash_attn"] = flash_attn
