"""AWQ checkpoints, checked against the vendored AutoAWQ packer and dequantizer."""

import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import unittest

import torch
import torch.nn.functional as F

from nanovllm.quantization import resolve_quant_config
from nanovllm.quantization.awq import AWQConfig, awq_to_natural
from nanovllm.quantization.base import pack_int4, unpack_int4
from upstream import autoawq
from upstream.nano_awq_packing import pack_awq_qweight, pack_awq_qzeros

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


def nano_awq_checkpoint(group_size: int):
    tensors, reference = lib.unquantized_tensors(), {}
    generator = torch.Generator().manual_seed(42)
    for name, (k, n) in lib.PROJECTIONS.items():
        groups = torch.arange(k) // group_size
        q = torch.randint(0, 16, (n, k), generator=generator, dtype=torch.int32)
        zeros = torch.randint(1, 16, (-(-k // group_size), n), generator=generator, dtype=torch.int32)
        scales = (torch.rand(zeros.shape, generator=generator) * 0.02 + 0.001).half()
        prefix = f"model.layers.0.{name}"
        tensors[f"{prefix}.qweight"] = pack_awq_qweight(q, bits=4)
        tensors[f"{prefix}.qzeros"] = pack_awq_qzeros(zeros - 1, bits=4)
        tensors[f"{prefix}.scales"] = scales
        weight = ((q.t() - zeros[groups]).float() * scales[groups].float()).t()
        reference[name] = lambda x, w=weight: F.linear(x, w)
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
            ({**base, "checkpoint_format": "unknown"}, ValueError),
            ({**base, "checkpoint_format": "nano-vllm", "group_size": -1}, ValueError),
            ({**base, "group_size": 0}, ValueError),
        ):
            with self.subTest(raw=raw), self.assertRaises(error):
                self.resolve(raw)
        with self.assertRaises(ValueError):
            self.resolve(base, "gptq")

    def test_reference_format_is_explicit(self):
        raw = {"quant_method": "awq", "w_bit": 4, "q_group_size": 32, "modules_to_not_convert": ["lm_head"]}
        self.assertEqual(self.resolve({**raw, "checkpoint_format": "nano-vllm"}), AWQConfig(32, "nano-vllm"))
        hf_config = lib.tiny_qwen3_config()
        hf_config.quantization_config = raw
        self.assertEqual(resolve_quant_config(hf_config, "/nonexistent", "awq", "nano-vllm"), AWQConfig(32, "nano-vllm"))


class TestAWQPacking(unittest.TestCase):
    def test_reference_repo_zero_offset_and_natural_order(self):
        config = AWQConfig(32, "nano-vllm")
        q = torch.arange(8, dtype=torch.int32).repeat(32, 1)
        zeros = torch.arange(1, 9, dtype=torch.int32).unsqueeze(0)
        qweight = pack_awq_qweight(q.t(), bits=4)
        qzeros = pack_awq_qzeros(zeros - 1, bits=4)
        self.assertIs(config.convert_checkpoint_tensor("qweight", qweight), qweight)
        torch.testing.assert_close(config.unpack_qzeros(qzeros), zeros)
        torch.testing.assert_close(config.dequantize(qweight, qzeros, torch.ones(1, 8)), (q - zeros).float().t())

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
    def test_reference_repo_checkpoint(self):
        for group_size in (32, 24):
            with self.subTest(group_size=group_size):
                tensors, reference = nano_awq_checkpoint(group_size)
                model = lib.load_checkpoint(AWQConfig(group_size, "nano-vllm"), tensors)
                lib.assert_matches_reference(self, model, reference)

    def test_fused_reference_checkpoint(self):
        tensors, reference = nano_awq_checkpoint(32)
        for target, sources in lib.FUSED.items():
            if len(sources) == 1:
                continue
            for suffix in ("qweight", "qzeros", "scales"):
                keys = [f"model.layers.0.{source}.{suffix}" for source in sources]
                tensors[f"model.layers.0.{target}.{suffix}"] = torch.cat([tensors.pop(key) for key in keys], dim=1)
        model = lib.load_checkpoint(AWQConfig(32, "nano-vllm"), tensors)
        lib.assert_matches_reference(self, model, reference)

    def test_partial_checkpoint_selection_and_fused_validation(self):
        tensors, reference = nano_awq_checkpoint(32)
        prefix = "model.layers.0.self_attn.o_proj"
        dense = torch.randn(64, 64)
        for suffix in ("qweight", "qzeros", "scales"):
            del tensors[f"{prefix}.{suffix}"]
        tensors[f"{prefix}.weight"] = dense
        reference["self_attn.o_proj"] = lambda x: F.linear(x, dense)
        config = AWQConfig(32, "nano-vllm", frozenset(key.removesuffix(".qweight") for key in tensors if key.endswith(".qweight")))
        model = lib.load_checkpoint(config, tensors)
        self.assertIsNone(model.model.layers[0].self_attn.o_proj.quant_config)
        lib.assert_matches_reference(self, model, reference)
        config = AWQConfig(32, "nano-vllm", frozenset({"model.layers.0.self_attn.q_proj"}))
        with self.assertRaisesRegex(ValueError, "part of fused layer"):
            lib.load_checkpoint(config, tensors)

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
