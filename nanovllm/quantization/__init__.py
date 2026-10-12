import json
import os
from types import SimpleNamespace

from nanovllm.fp8 import Fp8Config
from nanovllm.quantization.awq import AWQConfig
from nanovllm.quantization.base import WNA16Config
from nanovllm.quantization.gptq import GPTQConfig

QuantConfig = Fp8Config | WNA16Config
_WNA16_CONFIGS = {"awq": AWQConfig, "gptq": GPTQConfig}

# Older AutoGPTQ / AutoAWQ checkpoints keep their metadata next to config.json.
_METADATA_FILES = ("quantize_config.json", "quant_config.json")


def _read_raw_config(hf_config, model_path: str) -> dict | None:
    raw = getattr(hf_config, "quantization_config", None)
    if raw is not None:
        return dict(raw.to_dict() if hasattr(raw, "to_dict") else raw)
    for filename in _METADATA_FILES:
        path = os.path.join(model_path, filename)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            raw.setdefault("quant_method", "awq" if "w_bit" in raw else "gptq")
            return raw
    return None


def resolve_quant_config(hf_config, model_path: str, requested: str | None) -> QuantConfig | None:
    """Detect the checkpoint format; ``requested`` must agree with it when given."""
    raw = _read_raw_config(hf_config, model_path)
    method = str(raw.get("quant_method", "")).lower() if raw is not None else None
    requested = requested.lower() if requested is not None else None
    if requested is not None and requested not in ("fp8", "awq", "gptq"):
        raise ValueError(f"unsupported quantization: {requested!r}")
    if method is None:
        if requested is not None:
            raise ValueError(f"quantization={requested!r} requires a {requested} checkpoint")
        return None
    if requested is not None and requested != method:
        raise ValueError(f"checkpoint quant_method is {method!r}, not {requested!r}")
    if method == "fp8":
        return Fp8Config.from_hf_config(SimpleNamespace(quantization_config=raw), "fp8")
    if method in _WNA16_CONFIGS:
        return _WNA16_CONFIGS[method].from_config(raw)
    raise NotImplementedError(f"unsupported checkpoint quant_method {method!r}")
