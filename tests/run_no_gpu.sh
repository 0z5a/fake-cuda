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
case "$(uname -m)" in
    aarch64) library_dir=/usr/lib/aarch64-linux-gnu ;;
    x86_64)  library_dir=/usr/lib/x86_64-linux-gnu ;;
    *) echo "Unsupported Linux architecture: $(uname -m)" >&2; exit 2 ;;
esac
container_shim="$library_dir/libcuda.so.1"

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

    # Preserve the interpreter paths used by the venv's symlinks. uv-managed
    # Python includes its own standard library and shared-library dependencies.
    python_base="$("$venv/bin/python" -c 'import sys; print(sys.base_prefix)')"
    if [[ "$python_base" == /usr || "$python_base" == /usr/local ]]; then
        python_executable="$(realpath -e -- "$venv/bin/python")"
        python_stdlib="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"
        expat="$(realpath -e -- "$library_dir/libexpat.so.1")"
        python_mounts=(
            --mount "type=bind,src=$python_executable,dst=$python_executable,readonly"
            --mount "type=bind,src=$python_stdlib,dst=$python_stdlib,readonly"
            --mount "type=bind,src=$expat,dst=$library_dir/libexpat.so.1,readonly"
        )
        python_library="$(ldd "$python_executable" | awk '/libpython/ { print $3; exit }')"
        if [[ -n "$python_library" ]]; then
            python_mounts+=(--mount "type=bind,src=$python_library,dst=$python_library,readonly")
        fi
    else
        python_mounts=(--mount "type=bind,src=$python_base,dst=$python_base,readonly")
    fi

    echo "Testing PyTorch $variant in a no-GPU container"
    docker run --rm --pull=never --runtime=runc --network=none \
        --mount "type=bind,src=$project_dir,dst=/workspace,readonly" \
        --mount "type=bind,src=$shim,dst=$container_shim,readonly" \
        "${python_mounts[@]}" \
        --env "LD_PRELOAD=$container_shim" \
        --env "FAKE_CUDA_EXPECTED_RUNTIME=$runtime" \
        --env "FAKE_CUDA_DEVICE_COUNT=$device_count" \
        ubuntu:24.04 "/workspace/.venv-$variant/bin/python" "/workspace/tests/integration/pytorch/$test_file"
done
