"""
Checkpoint layout of a training run (written by trainer/base.py):

    <logdir>/checkpoint_model_000750/model.pt       {"generator": state dict, ...}, after 750 steps
    <logdir>/checkpoint_model_000750/optimizer.pt   {"generator": AdamW state, ...}, needed to resume

A checkpoint is named by its folder. The step count is read from the folder name.
"""
import glob
import os
import re
from typing import Iterable, Optional

import torch

MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"


def checkpoint_dir(logdir: str, step: int) -> str:
    return os.path.join(logdir, f"checkpoint_model_{step:06d}")


def checkpoint_step(folder: str) -> int:
    """The number of optimizer updates of the checkpoint in `folder`."""
    match = re.fullmatch(r"checkpoint_model_(\d+)", os.path.basename(os.path.normpath(folder)))
    if match is None:
        raise ValueError(f"'{folder}' is not a checkpoint folder (expected .../checkpoint_model_<step>)")
    return int(match.group(1))


def resumable_checkpoints(logdir: str) -> list:
    """The checkpoint folders in `logdir` that have an optimizer state, oldest first."""
    folders = [
        os.path.dirname(path)
        for path in glob.glob(os.path.join(logdir, "checkpoint_model_*", OPTIMIZER_FILE))
    ]
    return sorted(folders, key=checkpoint_step)


def latest_resumable_checkpoint(logdir: str) -> Optional[str]:
    folders = resumable_checkpoints(logdir)
    return folders[-1] if folders else None


def prune_optimizer_states(logdir: str, keep_steps: Iterable[int]) -> None:
    """
    Remove optimizer.pt from every checkpoint in `logdir` whose step is not in `keep_steps`.
    The optimizer state is about twice the size of the model. The model weights stay.
    """
    keep_steps = set(keep_steps)
    for folder in resumable_checkpoints(logdir):
        if checkpoint_step(folder) not in keep_steps:
            os.remove(os.path.join(folder, OPTIMIZER_FILE))


def atomic_save(obj, path: str) -> None:
    """Write to a temporary file first, so that a killed job never leaves a partial file at `path`."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_generator_state_dict(ckpt_path: str, key: str = None) -> dict:
    """
    Load generator weights from a checkpoint written by this repo.
    The weights are under the key "generator" or "generator_ema", with or without FSDP
    prefixes. A checkpoint from trainer/distillation.py holds both keys; then `key`
    must say which one to use, because inference.py uses "generator_ema".
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    found = [k for k in ("generator", "generator_ema") if k in state_dict]
    if key is None:
        if len(found) > 1:
            raise ValueError(
                f"{ckpt_path} holds both 'generator' and 'generator_ema'. "
                f"Pass `key` to choose one.")
        key = found[0] if found else None
    if key is not None:
        if key not in state_dict:
            raise KeyError(f"{ckpt_path} has no key '{key}' (keys: {list(state_dict)})")
        state_dict = state_dict[key]
    print(f"Loaded generator weights from {ckpt_path} (key: {key or 'top level'})")
    return {
        k.replace("model._fsdp_wrapped_module.", "model.", 1): v
        for k, v in state_dict.items()
    }
