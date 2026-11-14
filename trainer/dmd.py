import torch

from losses import DMD
from trainer.base import BaseTrainer
from utils.dataset import ResumableDistributedSampler, ShardingLMDBDataset, TextDataset, resumable_cycle
from utils.distributed import EMA_FSDP


class Trainer(BaseTrainer):
    """
    Self Forcing training with the DMD loss.

    One step is one update of the critic. The generator has an update on every
    `dfake_gen_update_ratio`-th step, before the critic update of that step.
    See BaseTrainer for checkpoints, resume and gradient accumulation.

    Config keys in addition to the ones of BaseTrainer (see configs/self_forcing_dmd.yaml):
        generator_ckpt   the ODE-initialized student: a model.pt of the ODE initialization.
                         A new run starts from it. A resumed run ignores it.
        ema_weight, ema_start_step   the EMA of the generator starts at step `ema_start_step`.
                                     model.pt holds it under the key "generator_ema".
    """

    def build(self):
        config = self.config

        # Step 1: The loss module and its networks
        assert config.distribution_loss == "dmd", "Only the DMD loss is supported"
        self.model = DMD(config, device=self.device)
        self.model.generator = self.fsdp(self.model.generator, config.generator_fsdp_wrap_strategy)
        self.model.real_score = self.fsdp(self.model.real_score, config.real_score_fsdp_wrap_strategy)
        self.model.fake_score = self.fsdp(self.model.fake_score, config.fake_score_fsdp_wrap_strategy)
        self.model.text_encoder = self.fsdp(
            self.model.text_encoder, config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=config.get("text_encoder_cpu_offload", False))

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        self.modules = {"generator": self.model.generator, "critic": self.model.fake_score}
        self.optimizers = {
            "generator": self.adamw(
                self.model.generator, config.lr, (config.beta1, config.beta2)),
            "critic": self.adamw(
                self.model.fake_score, config.get("lr_critic") or config.lr,
                (config.beta1_critic, config.beta2_critic)),
        }
        self.max_grad_norm = {
            "generator": config.get("max_grad_norm_generator", 10.0),
            "critic": config.get("max_grad_norm_critic", 10.0),
        }

        if config.get("generator_ckpt") and not self.resume_dir:
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)

        # The EMA of the generator is created in after_step, or restored from a checkpoint.
        self.ema_weight = config.get("ema_weight") or 0.0
        self.generator_ema = None

        # Step 2: The data. On resume, skip the batches that the first start_step steps used.
        if config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = ResumableDistributedSampler(
            dataset, shuffle=True, drop_last=True, seed=config.seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8, drop_last=True)
        self.gen_update_ratio = config.dfake_gen_update_ratio
        generator_updates = -(-self.start_step // self.gen_update_ratio)  # rounded up
        self.dataloader = resumable_cycle(
            dataloader,
            consumed_batches=(self.start_step + generator_updates) * self.grad_accum_steps)
        if self.is_main_process:
            print(f"{len(dataset)} prompts", flush=True)

        self.unconditional_dict = None

    # ------------------------------------------------------------------ EMA

    def after_step(self):
        if self.generator_ema is None and self.ema_weight > 0 and self.step >= self.config.ema_start_step:
            print(f"Setting up EMA with weight {self.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

    def extra_state(self) -> dict:
        if self.generator_ema is None:
            return {}
        return {"generator_ema": self.generator_ema.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        if "generator_ema" in state and self.ema_weight > 0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
            self.generator_ema.load_state_dict(state["generator_ema"])

    # ------------------------------------------------------------------ one step

    def _loss(self, name: str, batch: dict):
        """The loss of the network `name` ("generator" or "critic") on one batch."""
        config = self.config
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        initial_latent = None
        if config.i2v:
            initial_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if self.unconditional_dict is None:
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[config.negative_prompt] * batch_size)
                self.unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}

        loss_function = self.model.generator_loss if name == "generator" else self.model.critic_loss
        return loss_function(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=self.unconditional_dict,
            clean_latent=None,
            initial_latent=initial_latent
        )

    def _update(self, name: str) -> dict:
        """One optimizer update of the network `name`. Return its metrics (of this rank)."""
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        stats = {f"{name}_loss": 0.0}
        for _ in range(self.grad_accum_steps):
            loss, log_dict = self._loss(name, next(self.dataloader))
            (loss / self.grad_accum_steps).backward()
            stats[f"{name}_loss"] += loss.item() / self.grad_accum_steps
            if "dmdtrain_gradient_norm" in log_dict:
                stats["dmdtrain_gradient_norm"] = stats.get("dmdtrain_gradient_norm", 0.0) + (
                    log_dict["dmdtrain_gradient_norm"].mean().item() / self.grad_accum_steps)
        stats[f"{name}_grad_norm"] = self.modules[name].clip_grad_norm_(self.max_grad_norm[name]).item()
        optimizer.step()
        return stats

    def train_one_step(self):
        self.model.eval()  # prevent any randomness (e.g. dropout)
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        stats = {}
        if self.step % self.gen_update_ratio == 0:
            stats.update(self._update("generator"))
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
        stats.update(self._update("critic"))
        self.log(stats, show=("generator_loss", "critic_loss"))
