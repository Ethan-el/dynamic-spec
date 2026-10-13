"""Engine-level quantization dispatch shared by dense, FP8, AWQ and GPTQ checkpoints."""

import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import tempfile
import json
import os
import unittest

import torch

from nanovllm.config import Config
from nanovllm.layers.linear import QKVParallelLinear
from nanovllm.quantization import resolve_quant_config
from nanovllm.quantization.awq import AWQConfig
from nanovllm.quantization.gptq import GPTQConfig
from safetensors.torch import save_file

AWQ_RAW = {"quant_method": "awq", "bits": 4, "group_size": 32}
GPTQ_RAW = {"quant_method": "gptq", "bits": 4, "group_size": 32}
FP8_RAW = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [128, 128]}


def model_dir(quantization_config=None) -> str:
    path = tempfile.mkdtemp()
    hf_config = lib.tiny_qwen3_config()
    if quantization_config is not None:
        hf_config.quantization_config = quantization_config
    hf_config.save_pretrained(path)
    return path


class TestEngineConfig(unittest.TestCase):
    def test_reference_format_selection_and_checkpoint_scan(self):
        path = model_dir(AWQ_RAW)
        save_file({"model.layers.0.self_attn.q_proj.qweight": torch.zeros(64, 8, dtype=torch.int32)}, os.path.join(path, "model.safetensors"))
        config = Config(path, awq_checkpoint_format="nano-vllm")
        self.assertEqual(config.quant_config.checkpoint_format, "nano-vllm")
        self.assertEqual(config.quant_config.quantized_modules, frozenset({"model.layers.0.self_attn.q_proj"}))
        with self.assertRaisesRegex(ValueError, "requires an AWQ"):
            Config(model_dir(), awq_checkpoint_format="nano-vllm")

    def test_reference_sidecar_metadata(self):
        path = model_dir()
        with open(os.path.join(path, "quantize_config.json"), "w") as f:
            json.dump({"quant_method": "awq", "w_bit": 4, "q_group_size": 32, "modules_to_not_convert": ["lm_head"]}, f)
        config = Config(path, awq_checkpoint_format="nano-vllm")
        self.assertEqual(config.quant_config, AWQConfig(32, "nano-vllm"))

    def test_dispatch_and_runtime_dtype(self):
        for raw, quantization, quant_type, dtype in (
            (None, None, type(None), torch.bfloat16),
            (FP8_RAW, "fp8", object, torch.bfloat16),
            (AWQ_RAW, "awq", AWQConfig, torch.float16),
            (GPTQ_RAW, "gptq", GPTQConfig, torch.float16),
        ):
            with self.subTest(quantization=quantization):
                config = Config(model_dir(raw))
                self.assertEqual(config.quantization, quantization)
                self.assertIsInstance(config.quant_config, quant_type)
                self.assertEqual(config.dtype, dtype)

    def test_awq_and_gptq_reject_speculative_decoding_and_tp(self):
        for raw in (AWQ_RAW, GPTQ_RAW):
            path = model_dir(raw)
            with self.subTest(method=raw["quant_method"]):
                with self.assertRaisesRegex(NotImplementedError, "speculative"):
                    Config(path, speculative_model=path, num_speculative_tokens=3)
                with self.assertRaisesRegex(NotImplementedError, "speculative"):
                    Config(path, speculative_config={"method": "dspark", "model": path, "num_speculative_tokens": 7})
                with self.assertRaisesRegex(NotImplementedError, "tensor_parallel_size"):
                    Config(path, tensor_parallel_size=2)

    def test_dense_checkpoint_rejects_explicit_quantization(self):
        for requested in ("awq", "gptq", "fp8"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                Config(model_dir(), quantization=requested)

    def test_unknown_formats_are_errors(self):
        hf_config = lib.tiny_qwen3_config()
        hf_config.quantization_config = {"quant_method": "bitsandbytes"}
        with self.assertRaises(NotImplementedError):
            resolve_quant_config(hf_config, "/x", None)
        with self.assertRaises(ValueError):
            resolve_quant_config(lib.tiny_qwen3_config(), "/x", "int8")


class TestFusedBias(unittest.TestCase):
    def test_qkv_bias_shards(self):
        lib.init_dist()
        layer = QKVParallelLinear(16, 8, 2, 1, bias=True)
        for shard_id, size, value in (("q", 16, 1.0), ("k", 8, 2.0), ("v", 8, 3.0)):
            layer.bias.weight_loader(layer.bias, torch.full((size,), value), shard_id)
        expected = torch.cat([torch.full((16,), 1.0), torch.full((8,), 2.0), torch.full((8,), 3.0)])
        torch.testing.assert_close(layer.bias.data, expected)


if __name__ == "__main__":
    unittest.main()
