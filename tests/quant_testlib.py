"""One-layer Qwen3 checkpoints for the AWQ and GPTQ loading tests."""

import gpu_stubs  # noqa: F401  (must precede nanovllm imports)

import os
import tempfile

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

HIDDEN, INTERMEDIATE, HEAD_DIM = 64, 128, 32
PROJECTIONS = {
    # checkpoint module name: (in_features, out_features)
    "self_attn.q_proj": (HIDDEN, 2 * HEAD_DIM),
    "self_attn.k_proj": (HIDDEN, HEAD_DIM),
    "self_attn.v_proj": (HIDDEN, HEAD_DIM),
    "self_attn.o_proj": (2 * HEAD_DIM, HIDDEN),
    "mlp.gate_proj": (HIDDEN, INTERMEDIATE),
    "mlp.up_proj": (HIDDEN, INTERMEDIATE),
    "mlp.down_proj": (INTERMEDIATE, HIDDEN),
}
# nano-vllm module -> checkpoint modules fused into it, in output order.
FUSED = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "self_attn.o_proj": ("self_attn.o_proj",),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp.down_proj": ("mlp.down_proj",),
}


def init_dist():
    if not dist.is_initialized():
        init_file = os.path.join(tempfile.mkdtemp(), "dist_init")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=1, rank=0)


def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )


def dense_linears(seed: int = 0) -> dict[str, torch.nn.Linear]:
    torch.manual_seed(seed)
    return {name: torch.nn.Linear(k, n, bias=False) for name, (k, n) in PROJECTIONS.items()}


def minmax_params(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit min/max scales and integer zeros, both ``[num_groups, out_features]``."""
    n, k = weight.shape
    grouped = weight.reshape(n, k // group_size, group_size)
    w_max = grouped.amax(-1).clamp(min=0)
    w_min = grouped.amin(-1).clamp(max=0)
    scales = ((w_max - w_min) / 15).clamp(min=1e-5)
    zeros = torch.round(-w_min / scales).clamp(0, 15)
    return scales.t().contiguous(), zeros.t().contiguous()


def unquantized_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    tensors = {
        "model.embed_tokens.weight": torch.randn(32, HIDDEN, generator=generator).half(),
        "lm_head.weight": torch.randn(32, HIDDEN, generator=generator).half(),
    }
    for norm in ("model.norm", "model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm"):
        tensors[f"{norm}.weight"] = torch.ones(HIDDEN, dtype=torch.float16)
    for norm in ("q_norm", "k_norm"):
        tensors[f"model.layers.0.self_attn.{norm}.weight"] = torch.ones(HEAD_DIM, dtype=torch.float16)
    return tensors


def load_checkpoint(quant_config, tensors: dict[str, torch.Tensor]) -> Qwen3ForCausalLM:
    init_dist()
    model = Qwen3ForCausalLM(tiny_qwen3_config(), quant_config)
    with tempfile.TemporaryDirectory() as path:
        save_file(tensors, os.path.join(path, "model.safetensors"))
        load_model(model, path)
    return model


class KernelChecks:
    """Triton kernel vs the CPU dequantizer; subclasses set ``configs`` and ``random_packed``."""

    configs: tuple = ()
    # (in_features, out_features) of Qwen3-8B qkv, o, gate_up and down projections.
    shapes = ((4096, 6144), (4096, 4096), (4096, 24576), (12288, 4096))

    def random_packed(self, config, k, n, generator):
        raise NotImplementedError

    def random_inputs(self, config, k, n, generator):
        groups = config.num_groups(k)
        q = torch.randint(0, 16, (k, n), generator=generator)
        zeros = torch.randint(0, 16, (groups, n), generator=generator)
        scales = (torch.rand(groups, n, generator=generator) * 0.02 + 0.001).half()
        return q, zeros, scales

    def check(self, config, k, n, m, bias=False, transposed_input=False):
        generator = torch.Generator().manual_seed(k + n + m)
        qweight, qzeros, scales = (t.cuda() for t in self.random_packed(config, k, n, generator))
        if transposed_input:
            x = torch.randn(k, m, generator=generator).half().cuda().t()
        else:
            x = torch.randn(m, k, generator=generator).half().cuda()
        b = torch.randn(n, generator=generator).half().cuda() if bias else None
        weight = config.dequantize(qweight, qzeros, scales)
        expected = torch.nn.functional.linear(x.float(), weight.float(), None if b is None else b.float())
        actual = config.matmul(x, qweight, qzeros, scales, b)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-2)

    def test_qwen3_8b_projections(self):
        for config in self.configs:
            for k, n in self.shapes:
                for m in (1, 6, 48, 333):
                    with self.subTest(config=config, k=k, n=n, m=m):
                        self.check(config, k, n, m)

    def test_bias_odd_sizes_and_strided_input(self):
        for config in self.configs:
            with self.subTest(config=config):
                self.check(config, 4096, 1024 + 8, 17, bias=True)
                self.check(config, 4096, 1024, 5, transposed_input=True)

    def test_cuda_graph_replay_matches_eager(self):
        config = self.configs[0]
        generator = torch.Generator().manual_seed(0)
        qweight, qzeros, scales = (t.cuda() for t in self.random_packed(config, 4096, 6144, generator))
        static_x = torch.randn(8, 4096, generator=generator).half().cuda()
        config.matmul(static_x, qweight, qzeros, scales, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = config.matmul(static_x, qweight, qzeros, scales, None)
        static_x.copy_(torch.randn(8, 4096, generator=generator).half().cuda())
        graph.replay()
        torch.cuda.synchronize()
        eager = config.matmul(static_x, qweight, qzeros, scales, None)
        torch.testing.assert_close(static_out, eager, rtol=0, atol=0)


def assert_matches_reference(testcase, model, reference, atol=1e-5):
    """``reference[name](x)`` is the upstream output of checkpoint module ``name``."""
    layer = model.model.layers[0]
    for module_name, sources in FUSED.items():
        in_features = PROJECTIONS[sources[0]][0]
        x = torch.randn(5, in_features, generator=torch.Generator().manual_seed(7))
        with testcase.subTest(module=module_name):
            expected = torch.cat([reference[source](x) for source in sources], dim=-1)
            actual = layer.get_submodule(module_name)(x)
            torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-5, atol=atol)
