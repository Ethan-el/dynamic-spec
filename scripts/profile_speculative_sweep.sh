#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. The implementation lives in one launcher.
exec bash /root/autodl-tmp/nano-vllm/scripts/run_speculative_sweep.sh --profile "$@"
