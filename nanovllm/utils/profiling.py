"""Lightweight NVTX helpers for Nsight Systems traces.

Profiling is opt-in so normal inference does not pay for NVTX range pushes.
Set ``NANOVLLM_NVTX=1`` before importing nano-vLLM, or call
``set_nvtx_enabled(True)`` from a benchmark driver.
"""

from contextlib import contextmanager
import os

import torch


_enabled = os.environ.get("NANOVLLM_NVTX", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def set_nvtx_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


@contextmanager
def nvtx_range(message: str):
    """Emit an NVTX range when profiling is enabled and CUDA is available."""
    if not _enabled or not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
