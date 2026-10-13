"""DSpark acceptance, config checks, and aux-hidden shapes. No GPU."""

import quant_testlib as lib  # noqa: F401  (installs GPU stubs first)

import json
import math
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.config import Config
from nanovllm.dspark import accept_dspark_draft
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.sampling_params import SamplingParams
from test_quantization_config import model_dir


def dspark_dir(**overrides) -> str:
    path = model_dir()
    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    raw.update({
        "block_size": 7,
        "target_layer_ids": [0],
        "mask_token_id": 1,
        "markov_rank": 4,
    })
    raw.update(overrides)
    with open(config_path, "w") as f:
        json.dump(raw, f)
    return path


def dspark_config(path: str, **overrides):
    raw = {"method": "dspark", "model": path, "num_speculative_tokens": 7}
    raw.update(overrides)
    return raw


class TestDSparkAccept(unittest.TestCase):
    def logits(self, peaks):
        out = torch.full((1, len(peaks), 16), -100.0)
        for index, token in enumerate(peaks):
            out[0, index, token] = 100
        return out

    def test_greedy_stops_at_first_mismatch_and_takes_target_token(self):
        result = accept_dspark_draft(
            torch.tensor([[1, 2, 3]]),
            torch.zeros(1, 3, 16),
            self.logits([1, 9, 3]),
            torch.zeros(1, 16),
            torch.ones(1),
            True,
            Sampler(),
        )
        self.assertEqual(result.num_accepted.tolist(), [1])
        self.assertEqual(result.num_tokens.tolist(), [2])
        self.assertEqual(result.token_ids[0, :2].tolist(), [1, 9])

    def test_greedy_accepts_the_block_and_samples_bonus(self):
        bonus = torch.full((1, 16), -100.0)
        bonus[0, 5] = 100
        result = accept_dspark_draft(
            torch.tensor([[1, 2, 3]]),
            torch.zeros(1, 3, 16),
            self.logits([1, 2, 3]),
            bonus,
            torch.ones(1),
            True,
            Sampler(),
        )
        self.assertEqual(result.num_accepted.tolist(), [3])
        self.assertEqual(result.num_tokens.tolist(), [4])
        self.assertEqual(result.token_ids[0].tolist(), [1, 2, 3, 5])

    def test_sampling_rejects_when_target_mass_is_zero(self):
        class FixedSampler:
            def sample_with_probs(self, logits, temperatures):
                raise AssertionError("correction must sample from residual probabilities")

        result = accept_dspark_draft(
            torch.tensor([[0, 1]]),
            self.logits([0, 1]),
            self.logits([0, 4]),
            torch.zeros(1, 16),
            torch.ones(1),
            False,
            FixedSampler(),
        )
        self.assertEqual(result.num_accepted.tolist(), [1])
        self.assertEqual(result.token_ids[0, 1].item(), 4)

    def test_rejected_token_samples_residual_without_reapplying_temperature(self):
        draft = torch.tensor([[[1000.0, -1000.0, -1000.0]]])
        target = torch.tensor([[[-1000.0, 2 * math.log(2), 0.0]]])
        with patch("nanovllm.dspark.torch.multinomial", return_value=torch.tensor([1])) as sample:
            result = accept_dspark_draft(
                torch.tensor([[0]]), draft, target, torch.zeros(1, 3),
                torch.tensor([2.0]), False, Sampler(),
            )
        self.assertEqual(result.num_accepted.tolist(), [0])
        self.assertEqual(result.token_ids[0, 0].item(), 1)
        self.assertTrue(torch.allclose(sample.call_args.args[0], torch.tensor([0.0, 2 / 3, 1 / 3])))


class TestDSparkConfig(unittest.TestCase):
    def test_accepts_matching_draft(self):
        path = dspark_dir()
        config = Config(path, speculative_config=dspark_config(path))
        self.assertEqual(config.speculative_config.method, "dspark")
        self.assertEqual(config.num_speculative_tokens, 7)
        self.assertIsNone(config.speculative_model)

    def test_rejects_bad_drafts_and_the_autoregressive_draft_together(self):
        path = dspark_dir()
        with self.assertRaisesRegex(ValueError, "block_size"):
            Config(path, speculative_config=dspark_config(dspark_dir(block_size=4)))
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            Config(path, speculative_config=dspark_config(dspark_dir(target_layer_ids=[3])))
        with self.assertRaisesRegex(ValueError, "both"):
            Config(path, speculative_model=path, num_speculative_tokens=2, speculative_config=dspark_config(path))
        with self.assertRaisesRegex(ValueError, "dynamic K"):
            Config(
                path,
                speculative_config=dspark_config(path),
                num_speculative_tokens_per_batch_size=[(1, 8, 2)],
            )
        with self.assertRaisesRegex(ValueError, "method"):
            Config(path, speculative_config={"method": "eagle", "model": path, "num_speculative_tokens": 7})


class TestDSparkAuxHidden(unittest.TestCase):
    def test_selected_layers_return_hidden_plus_residual(self):
        lib.init_dist()
        model = Qwen3ForCausalLM(lib.tiny_qwen3_config())

        class StubLayer(nn.Module):
            def forward(self, positions, hidden_states, residual):
                if residual is None:
                    residual = torch.zeros_like(hidden_states)
                return hidden_states, residual + 1

        model.model.layers = nn.ModuleList([StubLayer(), StubLayer()])
        model.model.embed_tokens.weight.data.zero_()
        model.set_aux_hidden_state_layers((0, 1))
        hidden, aux = model(torch.zeros(3, dtype=torch.long), torch.arange(3))
        self.assertEqual(tuple(hidden.shape), (3, lib.HIDDEN))
        self.assertEqual([tuple(states.shape) for states in aux], [(3, lib.HIDDEN), (3, lib.HIDDEN)])
        self.assertTrue(torch.equal(aux[0], torch.ones(3, lib.HIDDEN)))
        self.assertTrue(torch.equal(aux[1], torch.full((3, lib.HIDDEN), 2)))

        model.set_aux_hidden_state_layers(())
        self.assertIsInstance(model(torch.zeros(2, dtype=torch.long), torch.arange(2)), torch.Tensor)


class TestDSparkSchedule(unittest.TestCase):
    def test_decode_reserves_the_draft_block_on_the_target_cache(self):
        seq = Sequence([1] * 250)
        manager = BlockManager(4, 256, num_speculative_tokens=7)
        manager.allocate(seq, manager.can_allocate(seq))
        self.assertEqual(len(seq.block_table), 1)
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)

        plain = Sequence([1] * 250)
        plain_manager = BlockManager(4, 256)
        plain_manager.allocate(plain, plain_manager.can_allocate(plain))
        plain_manager.may_append(plain)
        self.assertEqual(len(plain.block_table), 1)

    def test_batch_cannot_mix_greedy_and_sampling(self):
        path = dspark_dir()
        scheduler = Scheduler(Config(path, speculative_config=dspark_config(path)))
        scheduler.add(Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=4)))
        with self.assertRaisesRegex(ValueError, "mixing greedy"):
            scheduler.add(Sequence([1, 2, 3], SamplingParams(temperature=1, max_tokens=4)))


if __name__ == "__main__":
    unittest.main()
