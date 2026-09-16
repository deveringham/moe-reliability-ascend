# Ascend NPU runtime image (Atlas A2 / 910B, Ubuntu 22.04, CANN 9.1.0, Python 3.12).
# For Atlas A3 use quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12 as base image.
#
# Build from the repository root:
#   docker build -t moe-reliability .
#
# Run (expose the NPUs and driver of the host; add one --device per NPU):
#   docker run --rm -it --shm-size=16g \
#     --device /dev/davinci0 --device /dev/davinci1 \
#     --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
#     -v /usr/local/dcmi:/usr/local/dcmi \
#     -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#     -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
#     -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
#     -v /etc/ascend_install.info:/etc/ascend_install.info \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     -v "$PWD/results:/workspace/moe-reliability/results" \
#     -v "$PWD/models:/workspace/moe-reliability/models" \
#     -e HF_TOKEN \
#     moe-reliability moe-reliability run configs/examples/smoke_test.toml
FROM quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-system

# Build tools for source-only dependencies (arctic-inference) and git for provenance.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        gcc g++ cmake ninja-build libnuma-dev git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/moe-reliability
COPY . .

# Install the locked environment (or resolve it if no lock file has been committed yet).
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && if [ -f uv.lock ]; then uv sync --locked; else uv sync; fi

ENV VIRTUAL_ENV=/workspace/moe-reliability/.venv \
    PATH="/workspace/moe-reliability/.venv/bin:$PATH"

# Activate CANN and NNAL for every command.
RUN printf '#!/bin/bash\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\n[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh\nexec "$@"\n' \
        > /usr/local/bin/ascend-entrypoint && chmod +x /usr/local/bin/ascend-entrypoint
ENTRYPOINT ["/usr/local/bin/ascend-entrypoint"]
CMD ["moe-reliability", "doctor"]
