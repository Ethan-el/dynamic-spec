"""AWQ checkpoints, checked against the vendored AutoAWQ packer and dequantizer."""

import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import unittest

import torch
import torch.nn.functional as F

from nanovllm.quantization import resolve_quant_config
from nanovllm.quantization.awq import AWQConfig, awq_to_natural
from nanovllm.quantization.base import unpack_int4
from upstream import autoawq

WQLinear_GEMM, dequantize_gemm = autoawq()


def to_int32(word: int) -> int:
    return word - (1 << 32) if word >= 1 << 31 else word


def autoawq_checkpoint(group_size: int, seed: int = 0):
    """Quantize every projection with AutoAWQ; return (tensors, upstream reference outputs)."""
    tensors, reference = lib.unquantized_tensors(), {}
    for name, linear in lib.dense_linears(seed).items():
        k = linear.in_features
        export_group = k if group_size == -1 else group_size
        scales, zeros = lib.minmax_params(linear.weight.data, export_group)
        packed = WQLinear_GEMM.from_linear(linear, 4, export_group, scales=scales, zeros=zeros)
        prefix = f"model.layers.0.{name}"
        tensors[f"{prefix}.qweight"] = packed.qweight
        tensors[f"{prefix}.qzeros"] = packed.qzeros
        tensors[f"{prefix}.scales"] = packed.scales
        # float32 scales: the fp16 product rounding is not part of the format.
        weight = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales.float(), 4, export_group)
        reference[name] = lambda x, w=weight: F.linear(x, w.t())
    return tensors, reference


class TestAWQConfig(unittest.TestCase):
    def resolve(self, raw, requested=None):
        hf_config = lib.tiny_qwen3_config()
        hf_config.quantization_config = raw
        return resolve_quant_config(hf_config, "/nonexistent", requested)

    def test_parses_autoawq_metadata(self):
        raw = {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "GEMM", "zero_point": True}
        self.assertEqual(self.resolve(raw), AWQConfig(128))
        self.assertEqual(self.resolve(raw, "awq"), AWQConfig(128))
        legacy_keys = {"quant_method": "awq", "w_bit": 4, "q_group_size": 64}
        self.assertEqual(self.resolve(legacy_keys), AWQConfig(64))

    def test_rejects_unsupported_variants(self):
        base = {"quant_method": "awq", "bits": 4, "group_size": 128}
        for raw, error in (
            ({**base, "version": "gemv"}, NotImplementedError),
            ({**base, "version": "marlin"}, NotImplementedError),
            ({**base, "zero_point": False}, NotImplementedError),
            ({**base, "bits": 8}, NotImplementedError),
            ({**base, "modules_to_not_convert": ["visual"]}, NotImplementedError),
            ({**base, "group_size": 0}, ValueError),
        ):
            with self.subTest(raw=raw), self.assertRaises(error):
                self.resolve(raw)
        with self.assertRaises(ValueError):
            self.resolve(base, "gptq")


class TestAWQPacking(unittest.TestCase):
    def test_reorder_fixture(self):
        packed = torch.tensor([[to_int32(0x75316420)]], dtype=torch.int32)
        self.assertEqual(awq_to_natural(packed).item(), to_int32(0x76543210))

    def test_reorder_negative_word(self):
        packed = torch.tensor([[to_int32(0xFDB9ECA8)]], dtype=torch.int32)
        self.assertEqual(awq_to_natural(packed).item(), to_int32(0xFEDCBA98))

    def test_matches_autoawq_unpacking(self):
        linear = lib.dense_linears()["mlp.gate_proj"]
        scales, zeros = lib.minmax_params(linear.weight.data, 32)
        packed = WQLinear_GEMM.from_linear(linear, 4, 32, scales=scales, zeros=zeros)
        config = AWQConfig(32)
        natural = config.convert_checkpoint_tensor("qweight", packed.qweight)
        upstream = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales, 4, 32)
        ours = config.dequantize(natural, config.convert_checkpoint_tensor("qzeros", packed.qzeros), packed.scales)
        torch.testing.assert_close(ours.t(), upstream, rtol=0, atol=0)

    def test_reference_repo_convention_disagrees_with_autoawq(self):
        """Brassinai/nano-vllm@e909386a unpacks AWQ in natural order and adds 1 to zeros."""
        linear = lib.dense_linears()["mlp.gate_proj"]
        scales, zeros = lib.minmax_params(linear.weight.data, 32)
        packed = WQLinear_GEMM.from_linear(linear, 4, 32, scales=scales, zeros=zeros)
        groups = torch.arange(64) // 32
        q = unpack_int4(packed.qweight, 1)
        z = unpack_int4(packed.qzeros, 1) + 1
        reference_repo = (q - z[groups]).float() * packed.scales[groups].float()
        upstream = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales, 4, 32).float()
        self.assertGreater((reference_repo - upstream).abs().max().item(), 0.01)


class TestAWQLoading(unittest.TestCase):
    def test_autoawq_checkpoint(self):
        for group_size in (32, -1):
            with self.subTest(group_size=group_size):
                tensors, reference = autoawq_checkpoint(group_size)
                model = lib.load_checkpoint(AWQConfig(group_size), tensors)
                lib.assert_matches_reference(self, model, reference)

    def test_missing_tensor(self):
        tensors, _ = autoawq_checkpoint(32)
        del tensors["model.layers.0.self_attn.k_proj.scales"]
        with self.assertRaisesRegex(ValueError, "qkv_proj.*missing"):
            lib.load_checkpoint(AWQConfig(32), tensors)

    def test_shape_mismatch(self):
        tensors, _ = autoawq_checkpoint(32)
        key = "model.layers.0.mlp.down_proj.scales"
        tensors[key] = tensors[key][:, :32].contiguous()
        with self.assertRaisesRegex(ValueError, "does not match"):
            lib.load_checkpoint(AWQConfig(32), tensors)

    def test_group_size_mismatch(self):
        tensors, _ = autoawq_checkpoint(32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            lib.load_checkpoint(AWQConfig(16), tensors)

    def test_dense_weight_in_awq_checkpoint(self):
        tensors, _ = autoawq_checkpoint(32)
        tensors["model.layers.0.self_attn.o_proj.weight"] = torch.zeros(64, 64)
        with self.assertRaises(AttributeError):
            lib.load_checkpoint(AWQConfig(32), tensors)

    def test_unexpected_g_idx(self):
        tensors, _ = autoawq_checkpoint(32)
        tensors["model.layers.0.mlp.up_proj.g_idx"] = torch.arange(64, dtype=torch.int32) // 32
        with self.assertRaisesRegex(ValueError, "g_idx"):
            lib.load_checkpoint(AWQConfig(32), tensors)


if __name__ == "__main__":
    unittest.main()
