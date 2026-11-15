#!/bin/bash
# Submit the whole experiment at once, as five chained DeltaAI jobs. Run from the repo root.
#
#   ode_shared (0 -> 750)
#     +-- ode_baseline (750 -> 3000) --> dmd_baseline (600 steps)
#     +-- ode_swift    (750 -> 3000) --> dmd_swift    (600 steps)
#
# All jobs enter the queue now. A job starts only after the job it needs ends with
# success (afterok). This removes the delay of a person submitting the next job.
# It does not shorten the queue wait: Slurm counts a job's age priority only from the
# time its dependency is met, so each stage still waits in the queue after the one before.
#
#   DRY_RUN=1 bash slurm/submit_all.sh    print the sbatch commands only
#   bash slurm/submit_all.sh
#
# Optional: ALPHA (Swift alpha, default 0.25), SKIP_SHARED=1 (logs/ode_shared/checkpoint_model_000750 exists).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/slurm
ALPHA=${ALPHA:-0.25}

submit() {  # submit <dependency job id or ""> <sbatch arguments...>
  local dep=$1; shift
  local args=(--parsable)
  [ -n "$dep" ] && args+=(--dependency=afterok:$dep --kill-on-invalid-dep=yes)
  if [ -n "${DRY_RUN:-}" ]; then
    echo "sbatch ${args[*]} $*" >&2; echo "DRY$RANDOM"
  else
    sbatch "${args[@]}" "$@"
  fi
}

shared=""
if [ -z "${SKIP_SHARED:-}" ]; then
  # 750 steps at about 36 s per step is about 7.5 h. Ask for 10 h: a shorter limit backfills sooner.
  shared=$(submit "" --job-name=sf-ode-shared --time=10:00:00 --export=ALL,STAGE=shared slurm/train_ode.sbatch)
fi
# 2250 steps: about 22.5 h for the baseline; the Swift run adds a forward pass of the weak model.
baseline=$(submit "$shared" --job-name=sf-ode-baseline --time=32:00:00 --export=ALL,STAGE=baseline slurm/train_ode.sbatch)
swift=$(submit "$shared" --job-name=sf-ode-swift --time=38:00:00 \
  --export=ALL,STAGE=swift,OVERRIDES="swift_alpha=$ALPHA" slurm/train_ode.sbatch)
dmd_baseline=$(submit "$baseline" --job-name=sf-dmd-baseline --export=ALL,INIT=baseline slurm/train_dmd.sbatch)
dmd_swift=$(submit "$swift" --job-name=sf-dmd-swift --export=ALL,INIT=swift slurm/train_dmd.sbatch)

echo "shared=$shared baseline=$baseline swift=$swift dmd_baseline=$dmd_baseline dmd_swift=$dmd_swift"
