import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import unittest

import torch

from nanovllm.quantization.awq import AWQConfig
from nanovllm.quantization.base import pack_int4


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestAWQKernel(lib.KernelChecks, unittest.TestCase):
    configs = (AWQConfig(128), AWQConfig(-1))

    def random_packed(self, config, k, n, generator):
        # Runtime layout: qweight and qzeros already converted to natural order along N.
        q, zeros, scales = self.random_inputs(config, k, n, generator)
        return pack_int4(q, 1), pack_int4(zeros, 1), scales


if __name__ == "__main__":
    unittest.main()
