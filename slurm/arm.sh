#!/bin/bash
# One arm of the pilot experiment, on one 4-GPU node: ODE initialization, then DMD,
# with VBench evaluations in between. slurm/pilot.sbatch starts one arm per node.
#
#   bash slurm/arm.sh swift      shared run 0 -> 250, Swift run 250 -> 1000, then DMD
#   bash slurm/arm.sh baseline   waits for the shared step 250, baseline 250 -> 1000, then DMD
#
# Steps of an arm, in order:
#   1. ODE to step 1000. Checkpoints at 250, 500, 1000 (later: every 500).
#   2. VBench subset of ODE step 1000.
#   3. DMD in pieces of 50 steps from ODE step 1000. Checkpoint every 50 steps.
#      VBench subset after steps 50, 100, 150; full VBench after step 200.
# Every step is resumable, so the same command after an interruption continues where it stopped:
# a finished training piece only reloads and exits, and VBench skips the videos and scores that exist.
#
# Variables: LOGROOT (checkpoint folders), ALPHA (Swift alpha, default 0.25),
#            DMD_STEPS (default "50 100 150 200"), DMD_FULL_EVAL_AT (default 200).
# Needs slurm/eval_vbench.sbatch and evaluation/ (the VBench evaluation).
set -euo pipefail
ARM=${1:?usage: arm.sh swift|baseline}
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh
ALPHA=${ALPHA:-0.25}
ARGS=(--disable-wandb --no_visualize)
ODE_SAVE=(log_iters=500 "save_steps=[250]" "keep_optimizer_steps=[250]")
ODE_CONFIG=configs/self_forcing_ode.yaml
branch=$LOGROOT/ode_shared/checkpoint_model_000250
ode_final=$LOGROOT/ode_$ARM/checkpoint_model_001000

evaluate() {  # evaluate <model.pt> full|subset
  local subset=()
  [ "$2" = subset ] && subset=(PROMPTS_PER_DIMENSION=8 SAMPLES_PER_PROMPT=1)
  echo "########## VBench ($2) of $1"
  # WEIGHTS=generator: the plain weights. Up to DMD step 200 the EMA equals them or does not exist.
  # A failed evaluation must not stop the training of the arm: report it and continue.
  env CKPT="$1" WEIGHTS=generator "${subset[@]}" bash slurm/eval_vbench.sbatch \
    || echo "########## VBench FAILED for $1 ($2): rerun slurm/eval_vbench.sbatch for it later"
}

# Step 1: ODE initialization
case "$ARM" in
  swift)
    # If the shared run fails, the flag file tells the baseline arm to stop waiting.
    rm -f "$LOGROOT/ode_shared/FAILED"
    run_train --config_path "$ODE_CONFIG" --logdir "$LOGROOT/ode_shared" "${ARGS[@]}" \
      max_iters=250 "${ODE_SAVE[@]}" || { touch "$LOGROOT/ode_shared/FAILED"; exit 1; }
    run_train --config_path "$ODE_CONFIG" --logdir "$LOGROOT/ode_swift" "${ARGS[@]}" \
      resume_from="$branch" weak_generator_ckpt="$branch/model.pt" swift_alpha="$ALPHA" \
      max_iters=1000 "${ODE_SAVE[@]}" ;;
  baseline)
    # While the other node runs the shared ODE part (about 2.5 h), test the DMD stage on
    # this node: 2 steps from the base model, and one checkpoint. It shows early whether
    # the 14B teacher fits in GPU and host memory. A failure here does not stop the arm.
    # A restarted job skips the test once the shared checkpoint exists.
    if [ ! -e "$branch/optimizer.pt" ]; then
      rm -rf "$LOGROOT/dmd_smoke"
      run_train --config_path configs/self_forcing_dmd.yaml --logdir "$LOGROOT/dmd_smoke" "${ARGS[@]}" \
        max_iters=2 seed=1 && echo "########## DMD smoke test passed" \
        || echo "########## DMD smoke test FAILED: the DMD steps of both arms will likely fail too"
      rm -rf "$LOGROOT/dmd_smoke"
    fi
    waited=0
    until [ -e "$branch/optimizer.pt" ]; do
      [ -e "$LOGROOT/ode_shared/FAILED" ] && { echo "the shared ODE run failed"; exit 1; }
      [ "$waited" -ge 28800 ] && { echo "gave up waiting for $branch"; exit 1; }
      sleep 120; waited=$((waited + 120))
    done
    run_train --config_path "$ODE_CONFIG" --logdir "$LOGROOT/ode_baseline" "${ARGS[@]}" \
      resume_from="$branch" max_iters=1000 "${ODE_SAVE[@]}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

# Step 2: VBench subset of the ODE result
evaluate "$ode_final/model.pt" subset

# Step 3: DMD, evaluated every 50 steps. seed=1: the config has seed 0, which draws a new
# seed at every launch; a fixed seed gives both arms the same prompts and noise, also across pieces.
for steps in ${DMD_STEPS:-50 100 150 200}; do
  run_train --config_path configs/self_forcing_dmd.yaml --logdir "$LOGROOT/dmd_$ARM" "${ARGS[@]}" \
    generator_ckpt="$ode_final/model.pt" max_iters="$steps" log_iters=50 seed=1
  ckpt=$LOGROOT/dmd_$ARM/checkpoint_model_$(printf %06d "$steps")/model.pt
  if [ "$steps" = "${DMD_FULL_EVAL_AT:-200}" ]; then evaluate "$ckpt" full; else evaluate "$ckpt" subset; fi
done
echo "########## arm $ARM done"
