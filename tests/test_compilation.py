"""The opt-out must avoid calling torch.compile, including at import time."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import torch


class TestCompilation(unittest.TestCase):
    def load_helper(self):
        path = Path(__file__).parents[1] / "nanovllm/utils/compilation.py"
        spec = importlib.util.spec_from_file_location("compilation_test", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        return helper

    def test_disabled_never_calls_compiler_and_warns(self):
        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}), patch.object(torch, "compile", side_effect=AssertionError("compiler imported")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                helper = self.load_helper()
            fn = lambda x: x + 1
            self.assertIs(helper.compile_if_enabled(fn), fn)
            self.assertIn("Triton kernels and CUDA Graph remain enabled", str(caught[0].message))

    def test_enabled_uses_compiler(self):
        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "0"}), patch.object(torch, "compile", return_value="compiled") as compiler:
            helper = self.load_helper()
            fn = lambda x: x + 1
            self.assertEqual(helper.compile_if_enabled(fn), "compiled")
            compiler.assert_called_once_with(fn)
