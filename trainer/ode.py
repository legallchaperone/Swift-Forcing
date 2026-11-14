import contextlib

import torch
import torch.distributed as dist
import wandb

from losses import ODERegression
from trainer.base import BaseTrainer
from utils.dataset import ODERegressionLMDBDataset, ResumableDistributedSampler, resumable_cycle

MAX_GRAD_NORM = 10.0


class Trainer(BaseTrainer):
    """
    ODE initialization (CausVid Sec 4.3), with the optional Swift target.
    One step is one update of the student. See BaseTrainer for checkpoints and resume.

    Config keys in addition to the ones of BaseTrainer (see configs/self_forcing_ode.yaml):
        grad_accum_no_sync    skip the gradient communication on all but the last
                              micro-batch of an update (default true; it keeps one
                              unsharded fp32 gradient copy per GPU, about 6 GB)
        swift_alpha, weak_generator_ckpt   see losses/ode_regression.py

    The data order depends only on the seed and the step count. So the baseline and
    the Swift run see the same data, and a resumed run sees the same data as an
    uninterrupted one.
    """

    def build(self):
        config = self.config

        # Step 1: The loss module and its networks
        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)
        self.model.generator = self.fsdp(self.model.generator, config.generator_fsdp_wrap_strategy)
        self.model.text_encoder = self.fsdp(
            self.model.text_encoder, config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=config.get("text_encoder_cpu_offload", False))
        if self.model.weak_generator is not None:
            # Frozen and used without gradients, so it needs no FSDP wrapper. Use the
            # precision of the FSDP student: parameters in self.dtype, buffers in fp32.
            self.model.weak_generator = self.model.weak_generator.to(device=self.device)
            for param in self.model.weak_generator.parameters():
                param.data = param.data.to(self.dtype)

        self.visualize = not config.no_visualize and not self.disable_wandb and self.is_main_process
        if self.visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        self.modules = {"generator": self.model.generator}
        self.optimizers = {
            "generator": self.adamw(self.model.generator, config.lr, (config.beta1, config.beta2))}

        # Step 2: The data. On resume, skip the micro-batches that the first start_step updates used.
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=config.get("max_pair", int(1e8)))
        sampler = ResumableDistributedSampler(
            dataset, shuffle=True, drop_last=True, seed=config.seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=config.get("num_workers", 4), drop_last=True)
        self.dataloader = resumable_cycle(
            dataloader, consumed_batches=self.start_step * self.grad_accum_steps)
        if self.is_main_process:
            print(f"{len(dataset)} ODE pairs", flush=True)

        self.grad_accum_no_sync = bool(config.get("grad_accum_no_sync", True))
        self.num_levels = len(self.model.denoising_step_list)

    def _level_sums(self, values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Sum `values` [B, F] into one bin per denoising step."""
        sums = torch.zeros(self.num_levels, device=values.device, dtype=torch.float32)
        return sums.index_add_(0, index.flatten().to(values.device), values.flatten().float())

    def train_one_step(self):
        VISUALIZE = self.visualize and self.step % 100 == 0
        self.model.eval()  # prevent any randomness (e.g. dropout)
        optimizer = self.optimizers["generator"]

        # Sums over the samples of this rank. The last three have one entry per denoising step.
        loss_sum = torch.zeros(1, device=self.device)
        loss_to_y = torch.zeros(self.num_levels, device=self.device)
        count = torch.zeros(self.num_levels, device=self.device)
        weak_gap = torch.zeros(self.num_levels, device=self.device)

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(self.grad_accum_steps):
            # Step 1: Get the next batch of text prompts
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            ode_latent = batch["ode_latent"].to(
                device=self.device, dtype=self.dtype)

            # Step 2: Extract the conditional infos
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

            # Step 3: Train the generator. Only the last micro-batch reduces the
            # gradients over the GPUs; the others accumulate them locally.
            sync = not self.grad_accum_no_sync or micro_step == self.grad_accum_steps - 1
            with contextlib.nullcontext() if sync else self.model.generator.no_sync():
                generator_loss, log_dict = self.model.generator_loss(
                    ode_latent=ode_latent,
                    conditional_dict=conditional_dict
                )
                (generator_loss / self.grad_accum_steps).backward()

            index = log_dict["timestep_index"]
            loss_sum += generator_loss.detach().float()
            loss_to_y += self._level_sums(log_dict["loss_to_y"], index)
            count += self._level_sums(torch.ones_like(log_dict["loss_to_y"]), index)
            if "weak_gap" in log_dict:
                weak_gap += self._level_sums(log_dict["weak_gap"], index)

        generator_grad_norm = self.model.generator.clip_grad_norm_(MAX_GRAD_NORM)
        optimizer.step()

        # Step 4: Combine the statistics of all GPUs
        stats_tensor = torch.cat([loss_sum, loss_to_y, count, weak_gap])
        if self.world_size > 1:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        if self.is_main_process:
            loss_sum = stats_tensor[0]
            loss_to_y, count, weak_gap = stats_tensor[1:].view(3, self.num_levels)
            self._log(loss_sum, loss_to_y, count, weak_gap, generator_grad_norm)

        # Step 5: Visualization
        if VISUALIZE:
            # Visualize the input, output, and ground truth of the last batch
            videos = {
                "input": log_dict["input"],
                "output": log_dict["output"],
                "ground_truth": ode_latent[:, -1],
            }
            wandb.log({
                name: wandb.Video(
                    255.0 * (self.model.vae.decode_to_pixel(latent).cpu().numpy() * 0.5 + 0.5),
                    caption=name, fps=16, format="mp4")
                for name, latent in videos.items()
            }, step=self.step)

    def _log(self, loss_sum, loss_to_y, count, weak_gap, generator_grad_norm):
        """Turn the sums over all GPUs into the metrics of this update."""
        total = count.sum().clamp(min=1)
        stats = {
            # the training loss (against the Swift target when Swift is on)
            "generator_loss": (loss_sum / (self.grad_accum_steps * self.world_size)).item(),
            # the distance to the plain ODE target y: compare runs with this value
            "loss_to_y": (loss_to_y.sum() / total).item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }
        per_level = {"loss_to_y": loss_to_y}
        if self.model.weak_generator is not None:
            # the distance of the weak model to y
            stats["weak_loss_to_y"] = (weak_gap.sum() / total).item()
            per_level["weak_loss_to_y"] = weak_gap
        timesteps = self.model.denoising_step_list.tolist()
        for name, sums in per_level.items():
            for level in range(self.num_levels):
                if count[level] > 0:
                    stats[f"{name}_at_time_{timesteps[level]:.0f}"] = (sums[level] / count[level]).item()
        self.log(stats, show=("generator_loss", "loss_to_y", "weak_loss_to_y"))
