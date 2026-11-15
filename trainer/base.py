import gc
import json
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from utils.checkpoint import (
    MODEL_FILE, OPTIMIZER_FILE, atomic_save, checkpoint_dir, checkpoint_step,
    latest_resumable_checkpoint, prune_optimizer_states
)
from utils.distributed import (
    barrier, fsdp_wrap, fsdp_state_dict, fsdp_optim_state_dict,
    fsdp_load_optim_state_dict, launch_distributed_job
)
from utils.misc import set_seed


class BaseTrainer:
    """
    The run plumbing that every training stage shares: distributed setup, seed, config
    record, checkpoints, resume, gradient accumulation and metrics.

    A stage is a subclass with two methods:
        build()            create the loss module, wrap its networks with `self.fsdp`, and fill
                           `self.modules` and `self.optimizers`
        train_one_step()   do one update and call `self.log`

    `self.modules` maps a name to a trained network. `self.optimizers` maps the same name to
    the optimizer of that network. A checkpoint holds the state of both:

        <logdir>/checkpoint_model_<step>/model.pt       {name: network state, ...}
        <logdir>/checkpoint_model_<step>/optimizer.pt   {name: optimizer state, ...}

    One `step` is one pass through `train_one_step`. An optimizer update uses
    `total_batch_size` samples: `batch_size` per GPU, times the number of GPUs, times
    `self.grad_accum_steps`.

    Config keys:
        max_iters             stop after this many steps (unset: run until killed)
        log_iters             checkpoint interval
        save_steps            more steps that get a checkpoint (for example [250] with
                              log_iters 500 saves 250, 500, 1000, 1500, ...)
        total_batch_size      samples per optimizer update (default: batch_size * number of GPUs)
        resume_from           a checkpoint folder of another run. This run starts from its
                              networks, optimizer states and step count.
        keep_optimizer_steps  steps whose optimizer.pt is kept. The trainer removes the
                              others when it writes a later checkpoint.

    A run that is started again with the same logdir continues from its newest
    checkpoint that has an optimizer state.

    The random draws of a step depend only on the seed, the rank and the step count,
    not on where a run was started or resumed.
    """

    def __init__(self, config):
        self.config = config

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = self.rank == 0

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.disable_wandb = config.disable_wandb
        self.max_iters = config.get("max_iters")
        self.output_path = config.logdir
        self.keep_optimizer_steps = list(config.get("keep_optimizer_steps") or [])
        self.save_steps = set(config.get("save_steps") or [])

        self.resume_dir = self._find_resume_checkpoint()
        self.start_step = checkpoint_step(self.resume_dir) if self.resume_dir else 0
        self.step = self.start_step

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
            if self.start_step > 0 and self.is_main_process:
                print("WARNING: seed 0 draws a new seed at every launch, so a resumed run "
                      "does not see the same data as the run it continues")
        set_seed(config.seed + self.rank)

        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            self._save_config()
            if not self.disable_wandb:
                wandb.login(host=config.wandb_host, key=config.wandb_key)
                wandb.init(
                    config=OmegaConf.to_container(config, resolve=True),
                    name=config.config_name,
                    mode="online",
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    dir=config.wandb_save_dir
                )

        samples_per_pass = config.batch_size * self.world_size
        self.total_batch_size = config.get("total_batch_size") or samples_per_pass
        assert self.total_batch_size % samples_per_pass == 0, (
            f"total_batch_size {self.total_batch_size} is not a multiple of "
            f"batch_size * number of GPUs = {samples_per_pass}")
        self.grad_accum_steps = self.total_batch_size // samples_per_pass
        if self.is_main_process:
            print(f"{self.total_batch_size} samples per update = {config.batch_size} per GPU x "
                  f"{self.world_size} GPUs x {self.grad_accum_steps} accumulation steps", flush=True)

        self.modules = {}
        self.optimizers = {}
        self.build()
        assert self.modules and set(self.optimizers) <= set(self.modules), \
            "build() must fill self.modules, and name each optimizer after its network"
        if self.resume_dir:
            self._load_checkpoint(self.resume_dir)

    # ------------------------------------------------------------------ for subclasses

    def build(self):
        raise NotImplementedError

    def train_one_step(self):
        raise NotImplementedError

    def after_step(self):
        """Called after the step count is increased and before the checkpoint is written."""

    def extra_state(self) -> dict:
        """More entries for model.pt, gathered on every rank (for example EMA weights)."""
        return {}

    def load_extra_state(self, state: dict) -> None:
        """Restore the entries of `extra_state` from a loaded model.pt."""

    def fsdp(self, module, wrap_strategy, **kwargs):
        """Wrap a network for distributed training with the settings of the config."""
        return fsdp_wrap(
            module,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.config.mixed_precision,
            wrap_strategy=wrap_strategy,
            **kwargs
        )

    def adamw(self, module, lr, betas):
        return torch.optim.AdamW(
            [param for param in module.parameters() if param.requires_grad],
            lr=lr, betas=betas, weight_decay=self.config.weight_decay)

    def log(self, stats: dict, show=()):
        """
        Write the metrics of one step to metrics.jsonl and wandb.
        The console shows the keys in `show` every 10 steps.
        """
        if not self.is_main_process:
            return
        stats = {"step": self.step, **stats,
                 "elapsed_seconds": round(time.time() - self._train_start, 1)}
        with open(os.path.join(self.output_path, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(stats) + "\n")
        if self.step % 10 == 0 or self.step == self.start_step:
            shown = " ".join(f"{key}={stats[key]:.5f}" for key in show if key in stats)
            print(f"[step {self.step}] t={stats['elapsed_seconds']:.0f}s {shown}", flush=True)
        if not self.disable_wandb:
            wandb.log({k: v for k, v in stats.items() if k != "step"}, step=self.step)

    # ------------------------------------------------------------------ checkpoints

    def _find_resume_checkpoint(self):
        """
        Return the checkpoint folder to start from, or None for a new run.
        The newest checkpoint of this run wins over `resume_from`, so that an
        interrupted run continues and does not start again from its branch point.
        """
        resume_dir = self.config.get("resume_from")
        if resume_dir:
            for name in (MODEL_FILE, OPTIMIZER_FILE):
                path = os.path.join(resume_dir, name)
                assert os.path.isfile(path), f"resume_from: missing {path}"
        own = latest_resumable_checkpoint(self.output_path)
        if own and (not resume_dir or checkpoint_step(own) > checkpoint_step(resume_dir)):
            resume_dir = own
        if resume_dir and self.is_main_process:
            print(f"Resuming from {resume_dir} (step {checkpoint_step(resume_dir)})", flush=True)
        return resume_dir

    def _save_config(self):
        """Record the settings of this launch next to its checkpoints."""
        record = {k: v for k, v in OmegaConf.to_container(self.config, resolve=True).items()
                  if not k.startswith("wandb_")}  # no credentials in the log folder
        OmegaConf.save(OmegaConf.create(record),
                       os.path.join(self.output_path, f"config_from_step_{self.start_step:06d}.yaml"))

    def _load_checkpoint(self, folder: str):
        model_state = torch.load(os.path.join(folder, MODEL_FILE), map_location="cpu")
        for name, module in self.modules.items():
            module.load_state_dict(model_state[name], strict=True)
        self.load_extra_state(model_state)
        del model_state

        # Inherit the optimizer states (moments and step count) of the run we resume from.
        optimizer_state = torch.load(os.path.join(folder, OPTIMIZER_FILE), map_location="cpu")
        for name, optimizer in self.optimizers.items():
            # load_state_dict also restores the saved hyperparameters. Keep the
            # moments, but use the values of this run's config.
            keys = ("lr", "betas", "weight_decay")
            hyperparameters = [{k: group[k] for k in keys} for group in optimizer.param_groups]
            fsdp_load_optim_state_dict(self.modules[name], optimizer, optimizer_state[name])
            for group, values in zip(optimizer.param_groups, hyperparameters):
                group.update(values)

    def save(self):
        """Write checkpoint_model_<step>, which holds the state after exactly `step` steps."""
        print("Start gathering distributed model states...")
        model_state = {name: fsdp_state_dict(module) for name, module in self.modules.items()}
        model_state.update(self.extra_state())
        optimizer_state = {
            name: fsdp_optim_state_dict(self.modules[name], optimizer)
            for name, optimizer in self.optimizers.items()
        }

        if self.is_main_process:
            folder = checkpoint_dir(self.output_path, self.step)
            os.makedirs(folder, exist_ok=True)
            atomic_save(model_state, os.path.join(folder, MODEL_FILE))
            atomic_save(optimizer_state, os.path.join(folder, OPTIMIZER_FILE))
            print("Checkpoint saved to", folder, flush=True)
            # Keep only the newest optimizer state, and the ones asked for.
            prune_optimizer_states(self.output_path, keep_steps=[self.step, *self.keep_optimizer_steps])

    # ------------------------------------------------------------------ the loop

    def train(self):
        self._train_start = time.time()
        while self.max_iters is None or self.step < self.max_iters:
            set_seed(self.config.seed + self.rank + 1000 * self.step)
            self.train_one_step()
            barrier()
            if self.step % self.config.gc_interval == 0:
                gc.collect()
            self.step += 1
            self.after_step()

            if (not self.config.no_save) and (
                    self.step % self.config.log_iters == 0 or self.step in self.save_steps
                    or self.step == self.max_iters):
                self.save()
                torch.cuda.empty_cache()
