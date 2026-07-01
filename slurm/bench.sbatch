#!/bin/bash
#SBATCH --job-name=vob-bench
#SBATCH --partition=general
#SBATCH --output=logs/%j_vob.log
#SBATCH --error=logs/%j_vob.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# Sweep unattended as an array:  sbatch --array=0-3 slurm/bench.sbatch
# (each task runs `vob run`; resumability makes overlapping tasks skip done cells)

set -euo pipefail
mkdir -p logs

# --- Environment (mirrors ~/SRS-map2map/scripts/infer_set0.slurm pattern) ---
# Ensure `module` is defined (batch scripts don't always inherit the shell function).
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh "${MODULESHOME:-/usr/share/Modules}/init/bash"; do
        [ -f "$init" ] && source "$init" && break
    done
fi
module load cuda-12.4

# GPU venv built in Phase 0 (uv-provisioned Python >=3.10), off-$HOME because the
# cu12 torch stack (~7 GB) does not fit on the tiny home NFS. Override with VOB_VENV.
VENV="${VOB_VENV:-/data/user_data/$USER/vob-venv}"
source "${VENV}/bin/activate"

# --- Storage off $HOME (home NFS has ~15 GB free; never cache 8B blobs there) ---
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/user_data/$USER/hf}"
# vLLM torch.compile + CUDA-graph artifacts: persist OFF $HOME so the ~10 min cold
# compile happens ONCE and is reused by every subsequent cell (huge budget saver).
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/user_data/$USER/vllm-cache}"
mkdir -p "${HF_HUB_CACHE}" "${VLLM_CACHE_ROOT}"

echo "Job start: $(date) on $(hostname)" >&2
nvidia-smi >&2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
dcgmi discovery -l >&2 || echo "dcgmi discovery failed (telemetry will fall back to pynvml)" >&2

# --- Run pending cells under the budget guard (resumable) ---
# VOB_ONLY_SOURCE (baseline|ofat|interaction) narrows the group; unset = all.
# VOB_RUN_ARGS passes arbitrary extra flags to `vob run` (e.g. "--limit 1" for a smoke).
EXTRA_ARGS=()
if [[ -n "${VOB_ONLY_SOURCE:-}" ]]; then
    EXTRA_ARGS+=(--only-source "${VOB_ONLY_SOURCE}")
fi

vob run "${EXTRA_ARGS[@]}" ${VOB_RUN_ARGS:-}

echo "Done: $(date)" >&2
