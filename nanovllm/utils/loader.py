import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.layers.linear import LinearBase
from nanovllm.quantization.base import WNA16Config


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _wna16_module(model: nn.Module, param_name: str) -> LinearBase | None:
    try:
        module = model.get_submodule(param_name.rpartition(".")[0])
    except AttributeError:
        return None
    if isinstance(module, LinearBase) and isinstance(module.quant_config, WNA16Config):
        return module
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name, shard_id = weight_name, None
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        break
                module = _wna16_module(model, param_name)
                if module is not None and param_name.endswith(".g_idx"):
                    module.quant_config.validate_g_idx(f.get_tensor(weight_name), module.input_size)
                    continue
                try:
                    param = model.get_parameter(param_name)
                except AttributeError:
                    # GPTQ exporters save an all-zero bias for layers that have none.
                    if module is not None and param_name.endswith(".bias") and not f.get_tensor(weight_name).any():
                        continue
                    raise
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is None:
                    weight_loader(param, f.get_tensor(weight_name))
                else:
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
    for name, module in model.named_modules():
        if isinstance(module, LinearBase) and isinstance(module.quant_config, WNA16Config):
            module.check_wna16_loaded(name)
