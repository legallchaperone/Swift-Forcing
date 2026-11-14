import torch.nn.functional as F
from typing import Tuple
import torch

from losses.base import BaseModel
from utils.checkpoint import load_generator_state_dict
from networks.wrappers import WanDiffusionWrapper


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the module inside any FSDP wrappers."""
    while hasattr(module, "_fsdp_wrapped_module"):
        module = module._fsdp_wrapped_module
    return module


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details

        Swift (https://arxiv.org/abs/2610.25844): with `swift_alpha` != 0 the regression
        target y becomes y + swift_alpha * (y - G_weak(x_t, t)), where G_weak is a frozen
        early checkpoint of this same student (`weak_generator_ckpt`).
        """
        super().__init__(args, device)

        # Step 1: Configure the generator created in _initialize_models.
        # The trainer loads the weights of a resumed run.
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self._configure_causal_model(self.generator)
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        # Step 3 (Swift): the frozen weak model
        self.swift_alpha = float(getattr(args, "swift_alpha", 0.0))
        self.weak_generator = None
        if self.swift_alpha != 0:
            weak_ckpt = getattr(args, "weak_generator_ckpt", None)
            assert weak_ckpt, "swift_alpha != 0 requires weak_generator_ckpt"
            print(f"[Swift] alpha={self.swift_alpha}, weak generator from {weak_ckpt}")
            self.weak_generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
            self.weak_generator.load_state_dict(load_generator_state_dict(weak_ckpt), strict=True)
            self.weak_generator.requires_grad_(False)
            self.weak_generator.eval()
            self._configure_causal_model(self.weak_generator)

    def _configure_causal_model(self, wrapper: WanDiffusionWrapper) -> None:
        if self.num_frame_per_block > 1:
            wrapper.model.num_frame_per_block = self.num_frame_per_block
        if self.independent_first_frame:
            wrapper.model.independent_first_frame = True

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size, num_frames].
            - index: the position of each timestep in the denoising step list [batch_size, num_frames].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape
        # The trajectory holds one state per denoising step, plus the clean latent at the end.
        assert num_denoising_steps == len(self.denoising_step_list) + 1, (
            f"The ODE data has {num_denoising_steps} states, but a student with "
            f"{len(self.denoising_step_list)} denoising steps needs "
            f"{len(self.denoising_step_list) + 1} (one per step, plus the clean latent)")

        # Step 1: Randomly choose a timestep for each block of frames
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index.cpu()].to(self.device)

        return noisy_input, timestep, index

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep, index = self._prepare_generator_input(
            ode_latent=ode_latent)

        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2 (Swift): move the target away from the weak model's prediction
        regression_target = target_latent
        weak_gap = None
        if self.weak_generator is not None:
            with torch.no_grad():
                # Both models use the same causal attention mask. Share the student's
                # mask so that the weak model does not build a second copy.
                unwrap(self.weak_generator.model).block_mask = unwrap(self.generator.model).block_mask
                _, weak_pred = self.weak_generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep
                )
                correction = target_latent.float() - weak_pred.float()
                regression_target = (
                    target_latent.float() + self.swift_alpha * correction
                ).to(target_latent.dtype)
                # per-frame squared distance of the weak model to y
                weak_gap = correction.pow(2).mean(dim=[2, 3, 4])

        # Step 3: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], regression_target[mask], reduction="mean")

        # `loss_to_y` is always the distance to the plain ODE target y, so it is
        # comparable between baseline and Swift runs. Shape [batch_size, num_frames].
        loss_to_y = F.mse_loss(
            pred_image_or_video.float(), target_latent.float(), reduction='none'
        ).mean(dim=[2, 3, 4]).detach()

        log_dict = {
            "loss_to_y": loss_to_y,
            "timestep": timestep.float().mean(dim=1).detach(),
            "timestep_index": index.detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }
        if weak_gap is not None:
            log_dict["weak_gap"] = weak_gap.detach()

        return loss, log_dict
