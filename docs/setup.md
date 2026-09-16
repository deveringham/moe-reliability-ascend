# Setup

The experiments run exclusively on Huawei Ascend NPUs through `torch_npu` and the vLLM Ascend
plugin. Only the result dataset library (`packages/moe-reliability-results`) can run without NPUs.

## Validated software stack

The Python environment pins the vLLM Ascend 0.23.0 compatibility set. Its components are validated
together and must not be mixed with versions from other releases.

| layer | component | version |
|---|---|---|
| host | Ascend HDK (driver and firmware) | 26.0.RC1 |
| Ascend runtime | CANN Toolkit + Ops | 9.1.0 |
| Ascend runtime | NNAL (ATB) | 9.1.0 |
| framework | PyTorch | 2.10.0 |
| framework | torch_npu | 2.10.0.post4 |
| kernels | Triton Ascend | 3.2.2 |
| inference engine | vLLM | 0.23.0 |
| hardware plugin | vLLM Ascend | 0.23.0 |
| models | Transformers | 5.5.4 |
| Python | CPython | 3.12 (3.11 supported) |

The host components (HDK, CANN, NNAL) are installed outside this project. uv installs everything
from PyTorch upwards.

## Requirements

| component | requirement |
|---|---|
| NPUs | Atlas A2 series / 910B. The vLLM Ascend wheel installed by `uv sync` targets A2; see [Other Ascend hardware](#other-ascend-hardware) |
| OS | Linux on aarch64 or x86_64 with glibc >= 2.34 (e.g. Ubuntu 22.04, openEuler 24.03) |
| host software | Ascend driver and firmware, CANN Toolkit + Ops and NNAL 9.1.0; `npu-smi info` lists the NPUs |
| tools | [uv](https://docs.astral.sh/uv/getting-started/installation/), `git`, `tee`, and `gcc`, `g++`, `cmake`, `ninja` (for dependencies built from source) |
| network | Hugging Face Hub (models, `cais/mmlu`), PyPI, `download.pytorch.org` (x86_64 hosts) and `mirrors.huaweicloud.com` (Triton Ascend) |
| disk | model weights in the Hugging Face cache; one full copy of the model per non-zero imbalance level in `imbalance.model_dir`; NPU profiler traces for profiled runs |

Memory: Atlas 910B NPUs have 64 GB each. Mixtral-8x7B has about
94 GB of float16 weights, and DeepSeek-V2-Lite has about 31 GB. vLLM uses
`server.gpu_memory_utilization` of each NPU's memory for weights, activations and KV cache.

## 1. Prepare the host

Check the driver and firmware, then activate CANN and NNAL in every shell that runs experiments:

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Add both `source` lines to `~/.bashrc` (or your job scripts) on dedicated experiment hosts. The runtime
refuses to start when `ASCEND_TOOLKIT_HOME` is not set, and it warns when NNAL is not activated.

If the host does not have CANN 9.1.0, install it following the
[CANN installation resources](https://www.hiascend.com/cann/download), or use the
[Docker image](#docker) based on the official CANN container.

## 2. Install

```bash
git clone https://github.com/deveringham/moe-reliability-ascend.git
cd moe-reliability-ascend
uv venv          # Python 3.12 (see .python-version)
uv sync          # installs moe-reliability, moe-reliability-results and the Ascend stack
```

`uv sync` resolves dependencies from four sources:

Optional extras:

```bash
uv sync --extra notebooks    # Jupyter for notebooks/
```

The `dev` group with pytest is installed by default.

## 3. Verify

```bash
uv run moe-reliability doctor --n-npus 8
```

`doctor` checks the CANN activation, the CANN and driver versions, `npu-smi`, the torch_npu import,
NPU visibility, the vLLM Ascend plugin, every package of the validated stack and Triton Ascend.
Pass a configuration file to check the NPUs that configuration needs: 
`uv run moe-reliability doctor configs/examples/smoke_test.toml`.

Then run the test suite and a short real run on 2 NPUs:

```bash
uv run pytest
uv run moe-reliability run configs/examples/smoke_test.toml
```

## Selecting NPUs

By default all NPUs visible to the process are available, and a run uses the first
`hardware.n_npus` of them. Restrict a run to specific NPUs in its configuration:

```toml
[hardware]
n_npus = 4
visible_devices = "4,5,6,7"
```

Or use `--set hardware.visible_devices=4,5,6,7`. Runs that share a host
concurrently need disjoint `visible_devices` and different `server.port` values.

## Other Ascend hardware

The vLLM Ascend wheel on PyPI is built for Atlas A2. For Atlas A3, Atlas 300I DUO / 200I Pro or
950DT, install the matching vLLM Ascend 0.23.0 build into the project environment after `uv sync`
(a source build with the right `SOC_VERSION`, or the hardware-specific wheel variant), following the
[vLLM Ascend installation guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/). Alternatively,
start from the official `quay.io/ascend/vllm-ascend:v0.23.0-<hardware>` image. Atlas 300I DUO and
200I Pro do not support Triton Ascend; remove `triton-ascend` from `pyproject.toml` on those systems.
`moe-reliability doctor` reports the resulting versions.

## Docker

The `Dockerfile` builds on the official CANN 9.1.0 image for Atlas A2 / 910B (Ubuntu 22.04, Python 3.12).
The container entrypoint activates CANN and NNAL. The host's driver and NPU devices are mounted at run
time:

```bash
docker build -t moe-reliability .
docker run --rm -it --shm-size=16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/results:/workspace/moe-reliability/results" \
  -v "$PWD/models:/workspace/moe-reliability/models" \
  -e HF_TOKEN \
  moe-reliability moe-reliability run configs/examples/smoke_test.toml
```

Add one `--device /dev/davinciN` per NPU used by the run. Without arguments, the container runs
`moe-reliability doctor`. For Atlas A3, change the base image to
`quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12` and install the A3 vLLM Ascend build.

## Analysis-only installation

Machines that need only read results need neither NPUs nor the Ascend stack:

```bash
uv pip install ./packages/moe-reliability-results
# or: pip install ./packages/moe-reliability-results
moe-reliability-results --results-dir /path/to/results list
```

## Environment variables

| variable | used by | effect |
|---|---|---|
| `ASCEND_TOOLKIT_HOME`, `ATB_HOME_PATH`, ... | runs | set by the CANN and NNAL `set_env.sh` scripts; required |
| `HF_TOKEN` | model and dataset downloads | Hugging Face authentication |
| `ASCEND_RT_VISIBLE_DEVICES` | runs | NPUs visible to the process; set per run with `hardware.visible_devices` |
| `MOE_RESULTS_DIR` | `moe-reliability-results`, `ResultsStore()`, `resume`/`analyze`/`list`/`show` | default results directory (otherwise `./results`) |
| `[environment]` table | runs | set before the run starts and inherited by the vLLM server, e.g. `HCCL_CONNECT_TIMEOUT`, `HCCL_BUFFSIZE`, `TASK_QUEUE_ENABLE` |
