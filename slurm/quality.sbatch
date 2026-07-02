#!/bin/bash
#SBATCH --job-name=vob-quality
#SBATCH --partition=general
#SBATCH --output=logs/%j_quality.log
#SBATCH --error=logs/%j_quality.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
# Phase 4.5 FP8 quality gate: perplexity of each precision vs BF16.

set -euo pipefail
mkdir -p logs

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh "${MODULESHOME:-/usr/share/Modules}/init/bash"; do
        [ -f "$init" ] && source "$init" && break
    done
fi
module load cuda-12.4

VENV="${VOB_VENV:-/data/user_data/$USER/vob-venv}"
source "${VENV}/bin/activate"

export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/user_data/$USER/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/user_data/$USER/vllm-cache}"
mkdir -p "${HF_HUB_CACHE}" "${VLLM_CACHE_ROOT}"

echo "Quality gate start: $(date) on $(hostname)" >&2
vob quality

echo "Done: $(date)" >&2
