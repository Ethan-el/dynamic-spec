# Tested Environment

The dynamic speculative decoding implementation was tested with:

| Component | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu128 |
| CUDA build | 12.8 |
| Triton | 3.5.1 |
| Transformers | 4.57.6 |
| FlashAttention | 2.8.3 |

The benchmark GPU was an NVIDIA RTX 3090.

## Install

Create an environment with the versions above, then install this project in
editable mode:

```bash
pip install -e .
```

The package metadata in `pyproject.toml` defines the supported version ranges.
The table above records the exact environment used for the current CUDA Graph
and speculative decoding experiments.

## Verify

```bash
python - <<'PY'
import flash_attn
import torch
import transformers
import triton

print("python dependency check")
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("transformers:", transformers.__version__)
print("flash_attn:", flash_attn.__version__)
PY
```
