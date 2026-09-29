#!/usr/bin/env bash
set -euo pipefail

# Install two independent PyTorch CUDA wheels into this project, not the host
# Python or another checkout. uv's shared download cache avoids duplicate data.
# The CUDA 12/13 probes live in tests/integration/pytorch and run via run_no_gpu.sh.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
for variant in cu128 cu130; do
    venv="$project_dir/.venv-$variant"
    if [[ ! -x "$venv/bin/python" ]]; then
        uv --no-config venv --python /usr/bin/python3.12 "$venv"
    fi
    uv --no-config pip install --python "$venv/bin/python" \
        --torch-backend="$variant" "torch==2.11.0+$variant"
done
