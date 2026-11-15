#!/bin/bash
#SBATCH --job-name=sf-dmd
#SBATCH --account=bjki-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=44:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
# Self Forcing training with DMD (14B teacher), from an ODE-initialized student.
# Submit from the repo root. Create the log folder first:  mkdir -p logs/slurm
#
#   INIT=baseline  generator from logs/ode_baseline/checkpoint_model_003000  -> logs/dmd_baseline
#   INIT=swift     generator from logs/ode_swift/checkpoint_model_003000     -> logs/dmd_swift
#
#   sbatch --export=ALL,INIT=baseline slurm/train_dmd.sbatch
#   sbatch --export=ALL,INIT=swift    slurm/train_dmd.sbatch
#
# Optional variables:
#   ODE_STEP   step of the ODE checkpoint (default 3000)
#   GEN_CKPT   a model.pt to start from (overrides INIT and ODE_STEP)
#   LOGDIR     checkpoint folder (default: logs/dmd_<init>)
#   OVERRIDES  space-separated key=value config overrides (default max_iters=600)
#
# The two DMD runs use the same seed, prompts and config. They differ only in GEN_CKPT.
# A requeued or resubmitted job with the same LOGDIR continues from its newest checkpoint.
#
# Needs wan_models/Wan2.1-T2V-14B (about 70 GB) and prompts/vidprom_filtered_extended.txt.
# Time: 600 steps took about 1.5 h on 64 H100 in the Self Forcing paper (about 96 GPU-hours).
# On 4 GH200 with 16 accumulation steps, expect about 24-32 h, so about 100-130 SU per run.
set -euo pipefail
: "${INIT:?set INIT to baseline or swift}"
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

for path in wan_models/Wan2.1-T2V-14B prompts/vidprom_filtered_extended.txt; do
  [ -e "$path" ] || { echo "missing $path"; exit 1; }
done

ODE_STEP=${ODE_STEP:-3000}
GEN_CKPT=${GEN_CKPT:-logs/ode_$INIT/checkpoint_model_$(printf %06d "$ODE_STEP")/model.pt}
# WAIT_HOURS > 0: a job that starts before the ODE run is done waits for its last checkpoint.
# optimizer.pt is written after model.pt, so its presence means that model.pt is complete.
if [ "${WAIT_HOURS:-0}" -gt 0 ]; then wait_for "$(dirname "$GEN_CKPT")/optimizer.pt" "$WAIT_HOURS"; fi
[ -f "$GEN_CKPT" ] || { echo "missing $GEN_CKPT"; exit 1; }
LOGDIR=${LOGDIR:-logs/dmd_$INIT}

run_train --config_path configs/self_forcing_dmd.yaml --logdir "$LOGDIR" ${ARGS:---disable-wandb --no_visualize} \
  generator_ckpt="$GEN_CKPT" ${OVERRIDES:-max_iters=600}
