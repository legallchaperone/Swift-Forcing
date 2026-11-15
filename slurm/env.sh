# Shared by the sbatch scripts. The code runs inside the NGC PyTorch container, with a
# Python virtual environment on PATH. DeltaAI is aarch64, so x86 wheels do not work.
SIF=${SIF:-/sw/user/NGC_containers/pytorch_26.01-py3.sif}
VENV=${VENV:-/work/nvme/bjki/$USER/venvs/causal-forcing}
NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-4}}

# NCCL_NET_PLUGIN=none: the container preloads the libfabric (cxi) NCCL network plugin,
# which aborts at start-up in a plain sbatch allocation. One node needs only the
# shared-memory transport of NCCL.
in_container() {  # in_container <command...>
  apptainer exec --nv --bind /projects,/work,/tmp "$SIF" \
    env PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" HF_HUB_OFFLINE=1 NCCL_NET_PLUGIN=none "$@"
}

run_train() {  # run_train <train.py arguments...>
  echo "########## train.py $*"
  in_container python -m torch.distributed.run --nproc_per_node="$NGPU" train.py "$@"
}

VBENCH_VENV=${VBENCH_VENV:-/work/nvme/bjki/$USER/venvs/vbench}
VBENCH_ROOT=${VBENCH_ROOT:-/work/nvme/bjki/$USER/vbench}   # VBench source and scoring models
in_vbench_container() {  # in_vbench_container <command...>
  apptainer exec --nv --bind /projects,/work,/tmp "$SIF" \
    env PATH="$VBENCH_VENV/bin:/usr/local/bin:/usr/bin:/bin" HF_HUB_OFFLINE=1 \
    VBENCH_ROOT="$VBENCH_ROOT" VBENCH_CACHE_DIR="$VBENCH_ROOT/models" \
    HF_HOME="$VBENCH_ROOT/models/hf" TORCH_HOME="$VBENCH_ROOT/models/torch" "$@"
}

# The checkpoint folders of all training runs. Set LOGROOT to put them outside the repo,
# for example on /work/nvme, which has more free space than /work/hdd.
LOGROOT=${LOGROOT:-logs}

wait_for() {  # wait_for <file> <max hours>: block until <file> exists, so a job may start before its input is ready
  local file=$1 max_s=$(( ${2:-12} * 3600 )) waited=0
  until [ -e "$file" ]; do
    if [ "$waited" -ge "$max_s" ]; then echo "gave up waiting for $file"; exit 1; fi
    [ "$waited" -eq 0 ] && echo "########## waiting for $file (at most ${2:-12} h)"
    sleep 300; waited=$(( waited + 300 ))
  done
  [ "$waited" -gt 0 ] && echo "########## $file is ready after $(( waited / 60 )) min"
  return 0
}
