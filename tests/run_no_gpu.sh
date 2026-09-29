#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {cu128|cu130|all} [discovery|tensor|memory|streams|graph|multidevice]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

case "$1" in
    cu128) variants=(cu128) ;;
    cu130) variants=(cu130) ;;
    all)   variants=(cu128 cu130) ;;
    *)     usage; exit 2 ;;
esac

case "${2:-discovery}" in
    discovery) test_file=torch_discovery.py; device_count=1 ;;
    tensor) test_file=torch_tensor_smoke.py; device_count=1 ;;
    memory) test_file=torch_memory_copy.py; device_count=1 ;;
    streams) test_file=torch_stream_event.py; device_count=1 ;;
    graph) test_file=torch_graph_capture.py; device_count=1 ;;
    multidevice) test_file=torch_multidevice_smoke.py; device_count=2 ;;
    *) usage; exit 2 ;;
esac

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
shim="$(realpath -e -- "$project_dir/build/libcuda.so.1")"
expat="$(realpath -e -- /usr/lib/aarch64-linux-gnu/libexpat.so.1)"

for variant in "${variants[@]}"; do
    venv="$project_dir/.venv-$variant"
    if [[ ! -x "$venv/bin/python" ]]; then
        echo "Missing Python environment: $venv (create it with uv venv --python /usr/bin/python3.12)" >&2
        exit 1
    fi

    case "$variant" in
        cu128) runtime=12.8 ;;
        cu130) runtime=13.0 ;;
    esac

    echo "Testing PyTorch $variant in a no-GPU container"
    docker run --rm --pull=never --runtime=runc --network=none \
        --mount "type=bind,src=$project_dir,dst=/workspace,readonly" \
        --mount "type=bind,src=$shim,dst=/usr/lib/aarch64-linux-gnu/libcuda.so.1,readonly" \
        --mount "type=bind,src=/usr/bin/python3.12,dst=/usr/bin/python3.12,readonly" \
        --mount "type=bind,src=/usr/lib/python3.12,dst=/usr/lib/python3.12,readonly" \
        --mount "type=bind,src=$expat,dst=/usr/lib/aarch64-linux-gnu/libexpat.so.1,readonly" \
        --env "LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libcuda.so.1" \
        --env "FAKE_CUDA_EXPECTED_RUNTIME=$runtime" \
        --env "FAKE_CUDA_DEVICE_COUNT=$device_count" \
        ubuntu:24.04 "/workspace/.venv-$variant/bin/python" "/workspace/tests/integration/pytorch/$test_file"
done
