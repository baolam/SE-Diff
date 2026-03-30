import argparse
import json
import logging
import os

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from dataset.ecg_latent_dataset import DictDataset
from unet.conditional_unet import ECGconditional
from utils.path_utils import ensure_exists, repo_path
from utils.simulator_trainer import train_model
from vae.vae_model import MiniDecoderMSE


def parse_args():
    parser = argparse.ArgumentParser(description="SE-Diff training")
    parser.add_argument(
        "--config",
        default=str(repo_path("configs", "train.json")),
        help="Path to training configuration JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = ensure_exists(args.config, "training config")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta = config["meta"]
    roots = config["dependencies"]
    h_ = config["hyper_para"]

    k_max = 0
    if not os.path.exists(roots["checkpoints_dir"]):
        os.makedirs(roots["checkpoints_dir"])
    for item in os.listdir(roots["checkpoints_dir"]):
        if meta["exp_type"] + "_" in item:
            k = int(item.split("_")[-1])
            k_max = k if k > k_max else k_max
    save_weights_path = os.path.join(roots["checkpoints_dir"], f"{meta['exp_type']}_{k_max + 1}")
    os.makedirs(save_weights_path, exist_ok=True)

    logger = logging.getLogger(f"{meta['exp_type']}_{k_max + 1}")
    logger.setLevel("INFO")
    fh = logging.FileHandler(os.path.join(save_weights_path, "train.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(meta)
    logger.info(h_)

    device = torch.device(meta["device"] if torch.cuda.is_available() else "cpu")

    dataset_path = ensure_exists(roots["dataset_path"], "training dataset")
    train_dataset = DictDataset(str(dataset_path))
    train_dataloader = DataLoader(train_dataset, batch_size=h_["batch_size"])

    unet = ECGconditional(
        h_["num_train_steps"],
        kernel_size=h_["unet_kernel_size"],
        num_levels=h_["unet_num_level"],
        n_channels=4,
        text_embed_dim=1536,
    )

    beta_schedule = h_.get("beta_schedule", "linear")
    if beta_schedule == "linear":
        diffused_model = DDPMScheduler(
            num_train_timesteps=h_["num_train_steps"],
            beta_start=h_["beta_start"],
            beta_end=h_["beta_end"],
            beta_schedule="linear",
        )
    elif beta_schedule in ["cosine", "squaredcos_cap_v2"]:
        diffused_model = DDPMScheduler(
            num_train_timesteps=h_["num_train_steps"],
            beta_schedule="squaredcos_cap_v2",
        )
    else:
        raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

    decoder = MiniDecoderMSE(target_len=61, z_channels=4, base=128).to(device)
    mini_decoder_path = ensure_exists(roots["mini_decoder_path"], "mini decoder checkpoint")
    ckpt = torch.load(mini_decoder_path, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.to(device)

    train_model(
        meta=meta,
        save_weights_path=save_weights_path,
        dataloader=train_dataloader,
        diffused_model=diffused_model,
        unet=unet,
        decoder=decoder,
        simulator_prior_path=roots["simulator_prior_path"],
        h_=h_,
        logger=logger,
    )


if __name__ == "__main__":
    main()
