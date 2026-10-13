"""GPTQ checkpoints, checked against the vendored AutoGPTQ packer and forward."""

import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import json
import os
import tempfile
import unittest

import torch

from nanovllm.quantization import resolve_quant_config
from nanovllm.quantization.gptq import GPTQConfig
from upstream import autogptq_quant_linear

QuantLinear = autogptq_quant_linear()
# GPTQModel convert_gptq_v1_to_v2_format_module: 4-bit int32 qzeros += 0b0001...0001.
V1_TO_V2_QZEROS = 0x11111111


def autogptq_checkpoint(group_size: int, seed: int = 0, min_zero: int = 0, nonnegative: bool = False):
    """Quantize every projection with AutoGPTQ; return (tensors, upstream modules)."""
    tensors, modules = lib.unquantized_tensors(), {}
    for name, linear in lib.dense_linears(seed).items():
        if nonnegative:
            linear.weight.data.abs_()
        k, n = linear.in_features, linear.out_features
        export_group = k if group_size == -1 else group_size
        scales, zeros = lib.minmax_params(linear.weight.data, export_group)
        # Integer zeros: pack() casts zeros - 1 to uint32, and a float -1 is undefined (0 on ARM).
        zeros = zeros.clamp(min=min_zero).to(torch.int64)
        module = QuantLinear(4, group_size, k, n, bias=False, weight_dtype=torch.float32)
        module.pack(linear, scales.t(), zeros.t().contiguous(), torch.arange(k) // export_group)
        prefix = f"model.layers.0.{name}"
        tensors[f"{prefix}.qweight"] = module.qweight
        tensors[f"{prefix}.qzeros"] = module.qzeros
        tensors[f"{prefix}.scales"] = module.scales
        tensors[f"{prefix}.g_idx"] = module.g_idx
        modules[name] = module
    return tensors, modules


class TestGPTQConfig(unittest.TestCase):
    def resolve(self, raw, requested=None):
        hf_config = lib.tiny_qwen3_config()
        hf_config.quantization_config = raw
        return resolve_quant_config(hf_config, "/nonexistent", requested)

    def test_checkpoint_formats(self):
        raw = {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False, "sym": True}
        self.assertEqual(self.resolve(raw), GPTQConfig(128, 1))
        self.assertEqual(self.resolve({**raw, "checkpoint_format": "gptq"}), GPTQConfig(128, 1))
        self.assertEqual(self.resolve({**raw, "checkpoint_format": "gptq_v2"}), GPTQConfig(128, 0))
        self.assertEqual(self.resolve({**raw, "format": "gptq_v2"}), GPTQConfig(128, 0))
        self.assertEqual(self.resolve({**raw, "group_size": -1}, "gptq"), GPTQConfig(-1, 1))

    def test_rejects_unsupported_variants(self):
        base = {"quant_method": "gptq", "bits": 4, "group_size": 128}
        for raw in (
            {**base, "desc_act": True},
            {**base, "checkpoint_format": "marlin"},
            {**base, "checkpoint_format": "bitblas"},
            {**base, "bits": 8},
            {**base, "lm_head": True},
            {**base, "dynamic": {"-:.*down_proj": {}}},
        ):
            with self.subTest(raw=raw), self.assertRaises(NotImplementedError):
                self.resolve(raw)
        with self.assertRaises(ValueError):
            self.resolve(base, "awq")

    def test_quantize_config_json(self):
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "quantize_config.json"), "w") as f:
                json.dump({"bits": 4, "group_size": 64, "desc_act": False, "sym": True}, f)
            config = resolve_quant_config(lib.tiny_qwen3_config(), path, None)
        self.assertEqual(config, GPTQConfig(64, 1))


class TestGPTQLoading(unittest.TestCase):
    def test_autogptq_legacy_checkpoint(self):
        for group_size in (32, -1):
            with self.subTest(group_size=group_size):
                tensors, modules = autogptq_checkpoint(group_size)
                model = lib.load_checkpoint(GPTQConfig(group_size, 1), tensors)
                lib.assert_matches_reference(self, model, modules)

    def test_legacy_zero_point_zero(self):
        """AutoGPTQ stores zero 0 as -1; the unpacked nibble 15 must wrap back to 0."""
        tensors, modules = autogptq_checkpoint(32, nonnegative=True)
        self.assertTrue(all((m.qzeros == -1).all() for m in modules.values()))
        model = lib.load_checkpoint(GPTQConfig(32, 1), tensors)
        lib.assert_matches_reference(self, model, modules)

    def test_gptq_v2_checkpoint(self):
        tensors, modules = autogptq_checkpoint(32, min_zero=1)
        for key in [k for k in tensors if k.endswith(".qzeros")]:
            tensors[key] = tensors[key] + V1_TO_V2_QZEROS
        model = lib.load_checkpoint(GPTQConfig(32, 0), tensors)
        lib.assert_matches_reference(self, model, modules)

    def test_zero_bias_is_skipped_and_nonzero_bias_rejected(self):
        tensors, modules = autogptq_checkpoint(32)
        tensors["model.layers.0.self_attn.o_proj.bias"] = torch.zeros(64, dtype=torch.float16)
        lib.assert_matches_reference(self, lib.load_checkpoint(GPTQConfig(32, 1), tensors), modules)
        tensors["model.layers.0.self_attn.o_proj.bias"] = torch.ones(64, dtype=torch.float16)
        with self.assertRaises(AttributeError):
            lib.load_checkpoint(GPTQConfig(32, 1), tensors)

    def test_activation_order_g_idx(self):
        tensors, _ = autogptq_checkpoint(32)
        key = "model.layers.0.mlp.up_proj.g_idx"
        tensors[key] = tensors[key][torch.randperm(64, generator=torch.Generator().manual_seed(0))]
        with self.assertRaises(NotImplementedError):
            lib.load_checkpoint(GPTQConfig(32, 1), tensors)

    def test_missing_tensor(self):
        tensors, _ = autogptq_checkpoint(32)
        del tensors["model.layers.0.mlp.gate_proj.qzeros"]
        with self.assertRaisesRegex(ValueError, "gate_up_proj.*missing"):
            lib.load_checkpoint(GPTQConfig(32, 1), tensors)

    def test_awq_layout_is_rejected(self):
        tensors, _ = autogptq_checkpoint(32)
        key = "model.layers.0.self_attn.o_proj.qweight"
        tensors[key] = tensors[key].reshape(64, 8).contiguous()
        with self.assertRaisesRegex(ValueError, "does not match"):
            lib.load_checkpoint(GPTQConfig(32, 1), tensors)


if __name__ == "__main__":
    unittest.main()
