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
