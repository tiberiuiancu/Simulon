#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/megatron-lm"

module load 2025 CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0 NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

if [[ ! -d "${MEGATRON_DIR}" ]]; then
  echo "Megatron-LM submodule is missing at ${MEGATRON_DIR}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}"

python - <<'PY'
from pathlib import Path
import os

megatron_dir = Path(os.environ["PYTHONPATH"].split(":", 1)[0])
moe_checks = [
    megatron_dir / "megatron" / "core" / "inference" / "moe",
    megatron_dir / "megatron" / "core" / "transformer" / "moe",
]

if not any(path.exists() for path in moe_checks):
    raise SystemExit("MoE support not found in Megatron-LM submodule")

print("Megatron-LM ready:", megatron_dir)
print("MoE support detected")
PY

echo "Megatron-LM environment installed for validation training."
