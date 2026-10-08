import unittest
from types import SimpleNamespace

import torch

from nanovllm.fp8 import Fp8Config, dequantize_weight


class TestFp8(unittest.TestCase):
    def test_qwen_block_fp8_config(self):
        hf_config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        )
        self.assertEqual(Fp8Config.from_hf_config(hf_config, "fp8").block_size, (128, 128))

    def test_rejects_unimplemented_format(self):
        hf_config = SimpleNamespace(
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "static",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        )
        with self.assertRaises(NotImplementedError):
            Fp8Config.from_hf_config(hf_config, "fp8")

    def test_cpu_dequantization(self):
        weight = torch.tensor(
            [[1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 2, 2], [3, 3, 4, 4]],
            dtype=torch.float8_e4m3fn,
        )
        scales = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
        result = dequantize_weight(weight, scales, (2, 2), torch.float32)
        expected_scales = scales.repeat_interleave(2, 0).repeat_interleave(2, 1)
        torch.testing.assert_close(result, weight.float() * expected_scales)


if __name__ == "__main__":
    unittest.main()
