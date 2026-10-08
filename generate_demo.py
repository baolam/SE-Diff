"""
generate_demo.py
================
Demo generation script for SE-Diff. Works offline even if OPENAI_API_KEY is not set
by using stored text embeddings or mock embeddings.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from unet.conditional_unet import ECGconditional
from utils.path_utils import ensure_exists, repo_path
from vae.vae_model import VAE_Decoder


def generation_from_net(diffused_model, net, batch_size, device, text_embed, condition, num_channels=4, dim=128, seed=None):
    net.eval()

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    xi = torch.randn(batch_size, num_channels, dim, device=device)
    for timestep in tqdm(diffused_model.timesteps):
        t = timestep * torch.ones(batch_size, dtype=torch.long, device=device)
        with torch.no_grad():
            noise_predict = net(xi, t, text_embed, condition)
            xi = diffused_model.step(model_output=noise_predict, timestep=timestep, sample=xi)["prev_sample"]
    return xi


def build_scheduler(beta_schedule: str) -> DDPMScheduler:
    if beta_schedule == "linear":
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="linear",
        )
    elif beta_schedule == "cosine":
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    else:
        raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
    scheduler.set_timesteps(1000)
    return scheduler


def prepare_text_embedding(text: str, device: torch.device, label_info: dict = None) -> torch.Tensor:
    if label_info is not None and "text_embed" in label_info:
        embedding = np.asarray(label_info["text_embed"], dtype=np.float32)
    else:
        try:
            from utils.text_embeddings import get_text_embedding
            embedding = np.asarray(get_text_embedding(text), dtype=np.float32)
        except Exception:
            vec = np.random.randn(1536).astype(np.float32)
            embedding = vec / (np.linalg.norm(vec) + 1e-8)

    if embedding.ndim == 1:
        embedding = embedding[np.newaxis, np.newaxis, :]
    elif embedding.ndim == 2:
        embedding = embedding[np.newaxis, :, :]
    return torch.tensor(embedding, device=device)


def prepare_condition_dict(label_info: dict, device: torch.device) -> dict:
    condition_dict = {
        "sex": torch.tensor([[1 if label_info["gender"] == "M" else 0]], dtype=torch.float32, device=device),
        "age": torch.tensor([[label_info["age"]]], dtype=torch.float32, device=device),
        "heart rate": torch.tensor([[label_info["hr"]]], dtype=torch.float32, device=device),
    }
    for key in condition_dict:
        condition_dict[key] = condition_dict[key].unsqueeze(-1)
    return condition_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ECGs with the SE-Diff demo pipeline.")
    parser.add_argument("--data_path", type=str, default=str(repo_path("prerequisites", "training_latent_dataset.pt")))
    parser.add_argument("--unet_path", type=str, default=str(repo_path("prerequisites", "diffusion_model.pth")))
    parser.add_argument("--vae_path", type=str, default=str(repo_path("prerequisites", "vae_decoder.pth")))
    parser.add_argument("--output_dir", type=str, default=str(repo_path("outputs", "generation")))
    parser.add_argument("--output_name", type=str, default="evaluation_output.json")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = ensure_exists(args.data_path, "generation dataset")
    unet_path = ensure_exists(args.unet_path, "U-Net checkpoint")
    vae_path = ensure_exists(args.vae_path, "VAE checkpoint")

    unet = ECGconditional(number_of_diffusions=1000, kernel_size=7, num_levels=7, n_channels=4)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device)

    decoder = VAE_Decoder()
    checkpoint = torch.load(vae_path, map_location=device)
    if "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
    elif "model_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        decoder.load_state_dict(checkpoint)
    decoder.to(device)
    decoder.eval()

    diffused_model = build_scheduler(args.beta_schedule)

    ground_truth_data = torch.load(data_path, map_location="cpu")
    keys = sorted(ground_truth_data.keys())[: args.num_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    for key in tqdm(keys, desc="Processing samples"):
        sample = ground_truth_data[key]
        label_info = sample["label"]
        text_embed = prepare_text_embedding(label_info["text"], device, label_info)
        condition_dict = prepare_condition_dict(label_info, device)

        generated_latents = generation_from_net(
            diffused_model=diffused_model,
            net=unet,
            batch_size=1,
            device=device,
            text_embed=text_embed,
            condition=condition_dict,
            seed=args.seed,
        )

        ground_truth_latent = sample["data"].unsqueeze(0).to(device)
        with torch.no_grad():
            ground_truth_ecg = decoder(ground_truth_latent).squeeze(0).cpu().numpy().transpose(1, 0)
            generated_ecg = decoder(generated_latents).squeeze(0).cpu().numpy().transpose(1, 0)

        all_results.append(
            {
                "sample_id": key,
                "conditions": {
                    "original_text": label_info["text"],
                    "age": label_info["age"],
                    "gender": label_info["gender"],
                    "heart_rate": label_info["hr"],
                    "subject_id": int(label_info["subject_id"]),
                },
                "ground_truth_ecg_data": ground_truth_ecg.tolist(),
                "generated_ecg_data": generated_ecg.tolist(),
            }
        )

    output_path = output_dir / args.output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved generation results to {output_path}")


if __name__ == "__main__":
    main()
