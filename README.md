# Swift on the ODE initialization of Self Forcing

This repository applies Swift (https://www.alphaxiv.org/pdf/2610.25844) to the ODE initialization stage of
[Self Forcing](https://arxiv.org/abs/2506.08009). It is a reduced fork of the
[Self Forcing code](https://github.com/guandeh17/Self-Forcing): only the text-to-video path for
ODE initialization, Self Forcing training with DMD, and inference remains.

The ODE initialization trains the few-step causal student `G` to regress the ODE solution `y` of the teacher:

    baseline:  || G(x_t, t) - y ||^2
    Swift:     || G(x_t, t) - [ y + alpha * (y - G_weak(x_t, t)) ] ||^2

`G_weak` is a frozen early checkpoint of the same student.

## Layout

The code has four layers. Each layer calls only the layers below it.

| Layer | Path | Content |
| --- | --- | --- |
| Entry points | `train.py` | Reads a config file and `key=value` overrides, then starts a trainer. |
| | `inference.py` | Generates videos from a checkpoint. |
| Trainers | `trainer/base.py` | The run plumbing that both stages share: setup, seed, config record, checkpoints, resume, gradient accumulation, metrics. |
| | `trainer/ode.py` | One update of the ODE initialization. |
| | `trainer/dmd.py` | One step of Self Forcing training with DMD. |
| Losses | `losses/ode_regression.py` | The ODE regression loss and the Swift target. **The research code is here.** |
| | `losses/dmd.py`, `losses/base.py` | The DMD loss. |
| Networks | `networks/wrappers.py`, `networks/scheduler.py` | The generator, text encoder and VAE, and the noise schedule. |
| | `networks/rollout.py`, `networks/sampler.py` | Autoregressive generation: for DMD training, and for inference. |
| | `wan/` | The Wan2.1 network code, not edited. |

Other folders:

| Path | Content |
| --- | --- |
| `configs/` | `self_forcing_ode.yaml` (ODE initialization, baseline and Swift) and `self_forcing_dmd.yaml`. |
| `utils/` | Helpers: checkpoint folders, data sets, FSDP, LMDB. |
| `scripts/` | Generation of ODE pairs from the teacher. |
| `slurm/` | Job scripts for DeltaAI. |
| `paper/` | The paper (ICML 2026 template). |

To add a training stage, write a loss module in `losses/` and a subclass of `BaseTrainer`
with `build()` and `train_one_step()`. Then add it to `TRAINERS` in `train.py`.

## Setup

    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B

The ODE pairs are in `data/ode_pairs_16k_shift5` (from `zhengqili/Self-Forcing-Data`).
`slurm/env.sh` sets the container and the Python environment.

## The experiment

The baseline and the Swift run both start from the step-750 checkpoint of one shared run.
Thus they differ only in the regression target.

| Stage | Steps | Target | Output |
| --- | --- | --- | --- |
| `shared` | 0 to 750 | plain | `logs/ode_shared` |
| `baseline` | 750 to 3000 | plain | `logs/ode_baseline` |
| `swift` | 750 to 3000 | Swift, weak model = step 750 of `shared` | `logs/ode_swift` |

    mkdir -p logs/slurm
    sbatch --export=ALL,STAGE=shared --time=14:00:00 slurm/train_ode.sbatch
    sbatch --export=ALL,STAGE=baseline slurm/train_ode.sbatch
    sbatch --export=ALL,STAGE=swift    slurm/train_ode.sbatch
    sbatch --export=ALL,STAGE=swift,OVERRIDES="swift_alpha=0.5",LOGDIR=logs/ode_swift_a0.5 slurm/train_ode.sbatch

Without Slurm, the Swift stage is this command:

    torchrun --nproc_per_node=4 train.py --config_path configs/self_forcing_ode.yaml \
        --logdir logs/ode_swift --disable-wandb --no_visualize \
        resume_from=logs/ode_shared/checkpoint_model_000750 \
        weak_generator_ckpt=logs/ode_shared/checkpoint_model_000750/model.pt \
        swift_alpha=0.25

### Output of a run

    logs/ode_swift/
      config_from_step_000750.yaml       the settings of the launch that started at step 750
      metrics.jsonl                      one line per optimizer update
      checkpoint_model_001000/model.pt       {"generator": weights after 1000 updates}
      checkpoint_model_001000/optimizer.pt   {"generator": AdamW state}

Compare runs with `loss_to_y` in `metrics.jsonl`. It is the distance to the plain target `y`
in every run. `generator_loss` is the distance to the Swift target when Swift is on.

### Resume

These rules are the same for both training stages.

- A job that is started again with the same `--logdir` continues from its newest checkpoint.
- `resume_from=<checkpoint folder>` starts a run from a checkpoint of another run: weights, AdamW state and step count.
- The data order and the timestep draws depend only on `seed` and the step count. A resumed run sees the same data as an uninterrupted run.
- The trainer keeps `optimizer.pt` only for the newest checkpoint and for the steps in `keep_optimizer_steps`. `model.pt` stays for every checkpoint.

## After the ODE initialization

Self Forcing training with DMD, from an ODE-initialized checkpoint:

    torchrun --nproc_per_node=4 train.py --config_path configs/self_forcing_dmd.yaml \
        --logdir logs/dmd_swift --disable-wandb \
        generator_ckpt=logs/ode_swift/checkpoint_model_003000/model.pt max_iters=600

- The teacher is Wan2.1-T2V-14B (`real_name` in the config). Download it to `wan_models/Wan2.1-T2V-14B`.
- The prompts are `prompts/vidprom_filtered_extended.txt` from `gdhe17/Self-Forcing`.
- An update uses `total_batch_size: 64` samples. On 4 GPUs the trainer accumulates the gradients of 16 batches for each update.
- `model.pt` holds the keys `generator`, `critic`, and `generator_ema` (after step `ema_start_step`).

Video generation:

    python inference.py --config_path configs/self_forcing_dmd.yaml \
        --checkpoint_path logs/dmd_swift/checkpoint_model_000600/model.pt \
        --data_path prompts/MovieGenVideoBench_extended.txt \
        --output_folder videos/dmd_swift --use_ema

## Removed from the original code

The interactive demo, the GAN, SiD and CausVid losses, the diffusion (teacher forcing) trainer,
the bidirectional and multi-step inference pipelines, and the Wan modules that only they used.
`git show 33593df:<path>` prints an original file.

## Acknowledgements

This code is built on [Self Forcing](https://github.com/guandeh17/Self-Forcing) (Huang et al., 2025),
which is built on [CausVid](https://github.com/tianweiy/CausVid) and [Wan2.1](https://github.com/Wan-Video/Wan2.1).
