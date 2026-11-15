#!/bin/bash
# Submit the whole experiment at once, as four DeltaAI jobs. Run from the repo root.
#
#   J1  ode_shared (0 -> 750), then ode_swift (750 -> 3000), in one job   about 36 h
#   J2  ode_baseline (750 -> 3000)                                         about 23 h
#   J3  dmd_swift    (600 steps, from J1's step 3000)                      about 24-32 h
#   J4  dmd_baseline (600 steps, from J2's step 3000)                      about 24-32 h
#
# The queue wait (12 h to 4 days per 4-GPU job at a low fairshare) costs more time than
# the training. So the jobs do not wait for each other to END before they enter the queue:
#   - J1 holds the longest path (shared + Swift), so that path has one queue wait less.
#   - J2, J3 and J4 use "after:<job>+<minutes>": each one becomes eligible (and starts to
#     gain queue priority) a fixed time after the job it needs STARTS, before its input
#     exists. If a job starts too early, it waits for its input checkpoint (WAIT_HOURS)
#     and costs 4 SU per hour of waiting. With queue waits of 12 h or more, that is rare.
#
#   DRY_RUN=1 bash slurm/submit_all.sh    print the sbatch commands only
#   bash slurm/submit_all.sh
#
# Optional: ALPHA (Swift alpha, default 0.25).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/slurm
ALPHA=${ALPHA:-0.25}

submit() {  # submit <slurm dependency or ""> <sbatch arguments...>
  local dep=$1; shift
  local args=(--parsable)
  [ -n "$dep" ] && args+=(--dependency=$dep)
  if [ -n "${DRY_RUN:-}" ]; then
    echo "sbatch ${args[*]} $*" >&2; echo "DRY$RANDOM"
  else
    sbatch "${args[@]}" "$@"
  fi
}

# J1: 7.5 h + 28 h of training. Ask for the 48 h maximum, because the step time on this code is not measured yet.
j1=$(submit "" --job-name=sf-ode-shared-swift --time=48:00:00 \
  --export=ALL,STAGE=shared_swift,OVERRIDES="swift_alpha=$ALPHA" slurm/train_ode.sbatch)
# J2: eligible when J1 starts. The step-750 checkpoint exists about 7.5 h after that.
j2=$(submit "after:$j1" --job-name=sf-ode-baseline --time=34:00:00 \
  --export=ALL,STAGE=baseline,WAIT_HOURS=10 slurm/train_ode.sbatch)
# J3: eligible 15 h after J1 starts. J1 ends about 36 h after it starts, so with a
# queue wait of at least 12 h, J3 waits at most about 9 h for its checkpoint.
j3=$(submit "after:$j1+900" --job-name=sf-dmd-swift --time=48:00:00 \
  --export=ALL,INIT=swift,WAIT_HOURS=14 slurm/train_dmd.sbatch)
# J4: eligible 10 h after J2 starts. J2 ends about 23-30 h after it starts.
j4=$(submit "after:$j2+600" --job-name=sf-dmd-baseline --time=48:00:00 \
  --export=ALL,INIT=baseline,WAIT_HOURS=16 slurm/train_dmd.sbatch)

echo "J1 shared+swift=$j1  J2 baseline=$j2  J3 dmd_swift=$j3  J4 dmd_baseline=$j4"
