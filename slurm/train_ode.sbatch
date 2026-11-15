#!/bin/bash
#SBATCH --job-name=sf-ode
#SBATCH --account=bjki-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=40:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
# ODE initialization of Self Forcing, baseline and Swift. Submit from the repo root.
# Create the log folder first:  mkdir -p logs/slurm
#
# The experiment is three jobs. The baseline and the Swift run both resume from the
# step-750 checkpoint of the shared run, so they differ only in the regression target.
#
#   STAGE=shared    steps    0 ->  750, plain target      -> logs/ode_shared
#   STAGE=baseline  steps  750 -> 3000, plain target      -> logs/ode_baseline
#   STAGE=swift     steps  750 -> 3000, Swift target      -> logs/ode_swift
#   STAGE=shared_swift  the shared run, then the Swift run, in one job (see slurm/submit_all.sh)
#                   the weak model is the frozen step-750 checkpoint of the shared run
#
#   sbatch --export=ALL,STAGE=shared --time=14:00:00 slurm/train_ode.sbatch
#   sbatch --export=ALL,STAGE=baseline slurm/train_ode.sbatch
#   sbatch --export=ALL,STAGE=swift    slurm/train_ode.sbatch
#   sbatch --export=ALL,STAGE=swift,OVERRIDES="swift_alpha=0.5",LOGDIR=logs/ode_swift_a0.5 slurm/train_ode.sbatch
#
# Optional variables:
#   LOGROOT    folder of all checkpoint folders (default: logs)
#   LOGDIR     checkpoint folder (default: $LOGROOT/ode_<stage>)
#   OVERRIDES  space-separated key=value config overrides
#   ARGS       train.py flags (default: "--disable-wandb --no_visualize")
#
# A job that is resubmitted with the same LOGDIR continues from the newest checkpoint
# in LOGDIR that has an optimizer state (train.py does this).
#
# Cost: 4 GPUs = 4 SU per hour. 400 GB of memory and 64 cores add no charge (the free
# share of a 4-GPU job is 443 GB and 288 cores).
set -euo pipefail
: "${STAGE:?set STAGE to shared, baseline, swift or shared_swift}"
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh

SHARED=${SHARED:-$LOGROOT/ode_shared}
branch=$SHARED/checkpoint_model_000750   # the checkpoint where the baseline and the Swift run start
config=configs/self_forcing_ode.yaml

# WAIT_HOURS > 0: a branch job that starts before the step-750 checkpoint exists waits for it.
WAIT_HOURS=${WAIT_HOURS:-0}

case "$STAGE" in
  shared)   stage_args=(max_iters=750) ;;
  baseline) stage_args=(resume_from=$branch) ;;
  swift)    stage_args=(resume_from=$branch weak_generator_ckpt=$branch/model.pt swift_alpha=0.25) ;;
  shared_swift)
    # The shared run and then the Swift run in one job: one queue wait less on the longest path.
    run_train --config_path "$config" --logdir "$SHARED" ${ARGS:---disable-wandb --no_visualize} max_iters=750
    stage_args=(resume_from=$branch weak_generator_ckpt=$branch/model.pt swift_alpha=0.25)
    LOGDIR=${LOGDIR:-$LOGROOT/ode_swift} ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
LOGDIR=${LOGDIR:-$LOGROOT/ode_$STAGE}
if [ "$STAGE" = baseline ] || [ "$STAGE" = swift ]; then
  [ "$WAIT_HOURS" -gt 0 ] && wait_for "$branch/optimizer.pt" "$WAIT_HOURS"
fi

# Later key=value arguments override earlier ones.
run_train --config_path "$config" --logdir "$LOGDIR" ${ARGS:---disable-wandb --no_visualize} \
  "${stage_args[@]}" ${OVERRIDES:-}
