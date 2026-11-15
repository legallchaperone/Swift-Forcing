#!/bin/bash
#SBATCH --job-name=sf-pilot
#SBATCH --account=bjki-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x-%j.out
# The pilot experiment in one 2-node job: the Swift arm on one node, the baseline arm on
# the other (see slurm/arm.sh). One job means one queue wait for everything.
# The two arms do not communicate, except that the baseline arm waits for the shared
# step-250 checkpoint, which the Swift arm writes.
#
#   mkdir -p logs/slurm
#   sbatch --export=ALL,LOGROOT=/work/nvme/bjki/$USER/swift-forcing/logs slurm/pilot.sbatch
#
# Each arm writes its own log: logs/slurm/sf-pilot-<job>-<arm>.out
# A resubmitted job with the same LOGROOT continues both arms where they stopped.
set -euo pipefail
cd "${REPO:-$SLURM_SUBMIT_DIR}"
source slurm/env.sh
mkdir -p "$LOGROOT"
export LOGROOT REPO=$PWD

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
pids=()
for i in 0 1; do
  arm=$([ "$i" = 0 ] && echo swift || echo baseline)
  srun --nodes=1 --ntasks=1 --exclusive -w "${nodes[$i]}" --gpus-per-node=4 --cpus-per-task=64 \
    --output="logs/slurm/sf-pilot-$SLURM_JOB_ID-$arm.out" bash slurm/arm.sh "$arm" &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
