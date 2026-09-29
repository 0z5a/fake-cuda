#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]] || [[ "$1" != vllm && "$1" != sglang ]]; then
    echo "Usage: $0 {vllm|sglang} /absolute/path/to/uv-venv [local-model-directory]" >&2
    exit 2
fi

framework="$1"
venv="$(realpath -e -- "$2")"
if [[ ! -x "$venv/bin/python" ]]; then
    echo "Missing Python environment: $venv" >&2
    exit 1
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
shim="$(realpath -e -- "$project_dir/build/libcuda.so.1")"
expat="$(realpath -e -- /usr/lib/aarch64-linux-gnu/libexpat.so.1)"
args=(
    --rm --pull=never --runtime=runc --network=none
    --mount "type=bind,src=$project_dir,dst=/workspace,readonly"
    --mount "type=bind,src=$venv,dst=$venv,readonly"
    --mount "type=bind,src=$shim,dst=/usr/lib/aarch64-linux-gnu/libcuda.so.1,readonly"
    --mount "type=bind,src=/usr/bin/python3.12,dst=/usr/bin/python3.12,readonly"
    --mount "type=bind,src=/usr/lib/python3.12,dst=/usr/lib/python3.12,readonly"
    --mount "type=bind,src=$expat,dst=/usr/lib/aarch64-linux-gnu/libexpat.so.1,readonly"
    --env "LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcuda.so.1"
    --env "FRAMEWORK_VENV=$venv"
)
# The CUDA toolkit wheel lives inside each isolated venv. DeepGEMM needs
# CUDA_HOME even for import, before any kernel compilation is attempted.
args+=(--env "CUDA_HOME=$venv/lib/python3.12/site-packages/nvidia/cu13")
if [[ -n "${FAKE_CUDA_TRACE:-}" ]]; then
    args+=(--env FAKE_CUDA_TRACE=1)
fi
if [[ -n "${FAKE_CUDA_TRACE_CALLS:-}" ]]; then
    args+=(--env FAKE_CUDA_TRACE_CALLS=1)
fi
if [[ "$framework" == vllm && "${FAKE_CUDA_VLLM_DIAGNOSE_WITHOUT_NVML:-}" == 1 ]]; then
    args+=(--env FAKE_CUDA_VLLM_DIAGNOSE_WITHOUT_NVML=1)
fi
if [[ $# -eq 3 ]]; then
    model="$(realpath -e -- "$3")"
    if [[ ! -d "$model" ]]; then
        echo "Local model must be a directory: $model" >&2
        exit 2
    fi
    args+=(--mount "type=bind,src=$model,dst=$model,readonly")
    model_args=(--model "$model")
else
    model_args=()
fi

# Keep offline model loading from attempting network access through hub libraries.
args+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
# Bound imports and model startup; remove a container left behind if Docker's CLI times out.
run_dir="$(mktemp -d)"
cidfile="$run_dir/container.id"
trap 'if [[ -s "$cidfile" ]]; then docker rm -f "$(cat "$cidfile")" >/dev/null 2>&1 || true; fi; rm -rf -- "$run_dir"' EXIT

if timeout --signal=TERM --kill-after=10s 300s docker run --cidfile "$cidfile" "${args[@]}" \
    ubuntu:24.04 "$venv/bin/python" \
    "/workspace/tests/integration/frameworks/${framework}_startup.py" "${model_args[@]}"; then
    :
else
    status=$?
    if [[ $status -eq 124 || $status -eq 137 ]]; then
        echo "BLOCKED: $framework startup probe exceeded the 300-second Docker timeout (no inference verified)" >&2
    fi
    exit "$status"
fi
