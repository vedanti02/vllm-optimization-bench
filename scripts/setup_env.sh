#!/bin/bash
# Phase 0 — build the CUDA Python env INSIDE a GPU allocation.
# Run this from an interactive GPU shell:
#   srun --partition=general --gres=gpu:L40S:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
#   bash scripts/setup_env.sh
#
# System Python is 3.9.21 (too old for recent vLLM). We provision Python >=3.10
# via `uv` (fetches a standalone interpreter, no admin) and build a fresh venv.

set -euo pipefail

# `module` is a shell function defined only in interactive/login shells; a script
# run via `bash setup_env.sh` is a non-interactive child that does NOT inherit it,
# so we source the modules init explicitly (otherwise `module load` aborts under -e).
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh "${MODULESHOME:-/usr/share/Modules}/init/bash"; do
        [ -f "$init" ] && source "$init" && break
    done
fi

module load cuda-12.4

# CRITICAL: home NFS is tiny (~15 GB free). The cu12 torch stack (~7 GB venv) + uv
# cache (~14 GB) do NOT fit on $HOME and will fill it to 100% mid-install (silent
# exit 2). Put the venv AND all uv state on /data/user_data/$USER (411 GB free).
USER_DATA="${VOB_USER_DATA:-/data/user_data/$USER}"
VENV="${VOB_VENV:-$USER_DATA/vob-venv}"
PYVER="${VOB_PYVER:-3.11}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$USER_DATA/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$USER_DATA/uv-python}"
mkdir -p "$USER_DATA"

# --- uv (standalone; installs to ~/.local/bin) ---
if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> provisioning Python ${PYVER} + venv at ${VENV}"
uv python install "${PYVER}"
rm -rf "${VENV}"                       # clean rebuild (purge any prior cu130 torch)
uv venv --python "${PYVER}" "${VENV}"
source "${VENV}/bin/activate"

echo ">> installing vLLM (cu129 wheel variant) + this package"
# Driver 575.51.03 supports CUDA 12.9 max. vLLM's PyPI wheel (all versions >=0.20)
# is CUDA 13 and its compiled _C links libcudart.so.13 (needs driver >=580) — NO torch
# flag or version pin fixes that. vLLM DOES publish per-release cu129 wheel *variants*
# (ABI3) whose deps are all nvidia-*-cu12==12.9.*. Install that wheel directly, with
# PyTorch's cu129 index for torch. Bump VLLM_VER to retarget; lift entirely on driver>=580.
VLLM_VER="${VLLM_VER:-0.24.0}"
ARCH="$(uname -m)"
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}%2Bcu129-cp38-abi3-manylinux_2_28_${ARCH}.whl"
uv pip install "vllm @ ${VLLM_WHL}" --torch-backend=cu129
uv pip install -e ".[dev]"

echo ">> storage off \$HOME"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/user_data/$USER/hf}"
mkdir -p "${HF_HUB_CACHE}"

echo ">> Phase 0 exit check"
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| device', torch.cuda.get_device_name())"

cat <<EOF

Next:
  1) Record the printed vllm/torch/cuda versions + driver (575.51.03) in scratchpad.md.
  2) Confirm DCGM field ids:   dcgmi dmon -l
  3) Manual bench smoke:
       vllm serve unsloth/Llama-3.1-8B-Instruct --port 8000 &
       vllm bench serve --model unsloth/Llama-3.1-8B-Instruct --host 127.0.0.1 --port 8000 \\
         --dataset-name random --random-input-len 256 --random-output-len 256 \\
         --num-prompts 50 --save-result --result-filename /tmp/bench.json
EOF
