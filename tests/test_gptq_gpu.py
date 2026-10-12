import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import unittest

import torch

from nanovllm.quantization.base import pack_int4
from nanovllm.quantization.gptq import GPTQConfig


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGPTQKernel(lib.KernelChecks, unittest.TestCase):
    # Legacy zeros cover all 16 stored values, so the kernel's 15 -> 0 wrap is exercised.
    configs = (GPTQConfig(128, 1), GPTQConfig(128, 0), GPTQConfig(-1, 1))

    def random_packed(self, config, k, n, generator):
        q, zeros, scales = self.random_inputs(config, k, n, generator)
        return pack_int4(q, 0), pack_int4(zeros, 1), scales


if __name__ == "__main__":
    unittest.main()
