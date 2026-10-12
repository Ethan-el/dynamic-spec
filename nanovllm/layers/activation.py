import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.compilation import compile_if_enabled


class SiluAndMul(nn.Module):

    @compile_if_enabled
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
