import argparse
import os

import wandb
from omegaconf import OmegaConf

from trainer import DMDTrainer, ODETrainer

TRAINERS = {
    "ode": ODETrainer,  # ODE initialization, baseline and Swift
    "dmd": DMDTrainer,  # Self Forcing training with DMD
}


def main():
    parser = argparse.ArgumentParser(
        description="Any other argument is a config override of the form key=value, "
                    "for example: swift_alpha=0.5 max_iters=3000")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")

    args, overrides = parser.parse_known_args()
    # parse_known_args does not reject unknown flags, so check them here: a mistyped
    # flag such as `--no-save` must fail and not become a config key.
    for item in overrides:
        key = item.split("=", 1)[0]
        if item.startswith("-") or "=" not in item or not key:
            parser.error(f"unrecognized argument '{item}': flags are listed in --help, "
                         f"and config overrides have the form key=value")

    # Later sources win: default config < config file < key=value overrides < flags
    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
        OmegaConf.from_dotlist(overrides),
    )
    config.config_name = os.path.basename(args.config_path).split(".")[0]
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.trainer not in TRAINERS:
        parser.error(f"unknown trainer '{config.trainer}' (choose from {list(TRAINERS)})")
    TRAINERS[config.trainer](config).train()

    wandb.finish()


if __name__ == "__main__":
    main()
