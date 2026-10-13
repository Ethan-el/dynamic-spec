"""Load the vendored AutoAWQ and AutoGPTQ code without their optional CUDA extensions."""

import importlib.util
import sys
import types
from pathlib import Path

import transformers.pytorch_utils  # noqa: F401  (AutoGPTQ's pack() reads it)

_DIR = Path(__file__).parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def autoawq():
    """Return AutoAWQ's ``WQLinear_GEMM`` packer and ``dequantize_gemm`` reference."""
    if "awq.modules.linear.gemm" not in sys.modules:
        module_utils = types.ModuleType("awq.utils.module")
        module_utils.try_import = lambda name: (None, f"{name} is not installed")
        utils = types.ModuleType("awq.utils.utils")
        utils.get_best_device = lambda: "cpu"
        for name, module in (
            ("awq", types.ModuleType("awq")),
            ("awq.utils", types.ModuleType("awq.utils")),
            ("awq.utils.module", module_utils),
            ("awq.utils.utils", utils),
        ):
            sys.modules.setdefault(name, module)
        _load("awq.utils.packing_utils", "autoawq_packing_utils.py")
        _load("awq.modules.linear.gemm", "autoawq_gemm.py")
    return (
        sys.modules["awq.modules.linear.gemm"].WQLinear_GEMM,
        sys.modules["awq.utils.packing_utils"].dequantize_gemm,
    )


def autogptq_quant_linear():
    """Return AutoGPTQ's ``QuantLinear`` (legacy v1 format, PyTorch fallback forward)."""
    name = "autogptq_qlinear_cuda_old"
    module = sys.modules.get(name) or _load(name, "autogptq_qlinear_cuda_old.py")
    return module.QuantLinear
