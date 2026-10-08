"""
Train Phase 1: VAE Encoder/Decoder + MiniDecoder (MiniDecoderMSE)
================================================================
This script trains the two components used in Phase 1 of SE-Diff:

  1. VAE (Encoder + Decoder)
     - Compresses raw 12-lead ECG of shape (B, L, 12) into a
       continuous latent space z of shape (B, 4, L/8).
     - Optimised with MSE reconstruction loss + KL divergence.

  2. MiniDecoderMSE (jointly fine-tuned from frozen encoder)
     - A lightweight decoder that maps z -> first-cycle ECG
       segment of shape (B, 61, 12).
     - Supervised with MSE against the first R-R cycle extracted
       by NeuroKit2.
     - The saved weights are consumed as a frozen module in Phase 2
       to calculate the Simulator Constraint Loss.

Usage
-----
  python train_phase1.py --config configs/train_phase1.json

Expected dataset layout
-----------------------
  prerequisites/raw_ecg_data/
      000001.pt
      000002.pt
      ...

  Each .pt file must be a dict with keys:
      'data'   : Tensor (L, 12)  – raw 12-lead ECG waveform
      'label'  : dict with keys 'text', 'age', 'gender', 'hr'
"""

import argparse
import json
import logging
import math
import os
import time

try:
    import neurokit2 as nk
    HAS_NK = True
except Exception:
    HAS_NK = False

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.path_utils import ensure_exists, repo_path
from vae.vae_model import (
    MiniDecoderMSE,
    VAE_Decoder,
    VAE_Encoder,
    loss_function,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RawECGDataset(Dataset):
    """
    Loads individual .pt files from a directory.

    Each file must contain:
        { 'data': Tensor (L, 12), 'label': dict }
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.file_list = sorted(
            [f for f in os.listdir(dataset_dir) if f.endswith(".pt")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_dir, self.file_list[idx])
        sample = torch.load(path, map_location="cpu")
        # data: (L, 12) – keep as-is; encoder expects (B, L, 12) -> transposes internally
        return sample["data"].float(), sample["label"]


# ---------------------------------------------------------------------------
# First-cycle extraction helper (ground-truth for MiniDecoder supervision)
# ---------------------------------------------------------------------------

def extract_first_cycle_batch(ecg_batch: torch.Tensor,
                              sampling_rate: float = 102.4,
                              cycle_len: int = 61) -> torch.Tensor:
    """
    For each sample in the batch, extract the first complete R-R cycle
    across all 12 leads.
    """
    B, L, n_lead = ecg_batch.shape
    ecg_np = ecg_batch.detach().cpu().numpy()  # (B, L, 12)

    window_b = int(0.2 * sampling_rate)
    window_a = int(0.4 * sampling_rate)

    cycles = []
    for i in range(B):
        leads_cycles = []
        for lead in range(n_lead):
            signal = ecg_np[i, :, lead]
            r_idx = []
            if HAS_NK:
                try:
                    _, r = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
                    r_idx = r["ECG_R_Peaks"]
                except Exception:
                    r_idx = []

            if len(r_idx) < 2:
                cycle = signal[:cycle_len]
                if len(cycle) < cycle_len:
                    cycle = np.pad(cycle, (0, cycle_len - len(cycle)))
            else:
                sec = r_idx[1]
                st = max(sec - window_b, 0)
                ed = min(sec + window_a, signal.shape[0])
                cycle = signal[st:ed]
                if len(cycle) < cycle_len:
                    cycle = np.pad(cycle, (0, cycle_len - len(cycle)))
                else:
                    cycle = cycle[:cycle_len]
            leads_cycles.append(torch.tensor(cycle, dtype=torch.float32))

        cycles.append(torch.stack(leads_cycles, dim=-1))  # (cycle_len, 12)

    return torch.stack(cycles, dim=0)  # (B, cycle_len, 12)


# ---------------------------------------------------------------------------
# Training loop – one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    dataloader: DataLoader,
    encoder: VAE_Encoder,
    decoder: VAE_Decoder,
    mini_decoder: MiniDecoderMSE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    kld_weight: float,
    mini_decoder_weight: float,
    logger: logging.Logger,
) -> dict:
    """Run one training epoch and return averaged losses."""

    encoder.train()
    decoder.train()
    mini_decoder.train()

    total_loss_list, recon_list, kld_list, mini_list = [], [], [], []

    for batch_idx, (ecg, _) in enumerate(dataloader):
        # ecg: (B, L, 12)
        ecg = ecg.to(device)

        # ----- VAE forward -----
        z, mu, log_var = encoder(ecg)                   # z: (B, 4, L/8)
        ecg_recon = decoder(z)                          # (B, L, 12)

        # Align lengths in case of off-by-one from strided convolutions
        min_len = min(ecg.shape[1], ecg_recon.shape[1])
        ecg_t = ecg[:, :min_len, :]
        ecg_r = ecg_recon[:, :min_len, :]

        vae_losses = loss_function(ecg_r, ecg_t, mu, log_var,
                                   kld_weight=kld_weight)
        vae_loss = vae_losses["loss"]

        # ----- MiniDecoder forward -----
        # Extract ground-truth first cycle from real ECG
        gt_cycle = extract_first_cycle_batch(ecg)       # (B, 61, 12)
        gt_cycle = gt_cycle.to(device)

        pred_cycle = mini_decoder(z.detach())            # (B, 61, 12)
        mini_loss = F.mse_loss(pred_cycle, gt_cycle)

        # ----- Combined loss -----
        loss = vae_loss + mini_decoder_weight * mini_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_list.append(loss.item())
        recon_list.append(vae_losses["mse"].item())
        kld_list.append(vae_losses["KLD"].item())
        mini_list.append(mini_loss.item())

    return {
        "total": sum(total_loss_list) / len(total_loss_list),
        "recon": sum(recon_list) / len(recon_list),
        "kld":   sum(kld_list) / len(kld_list),
        "mini":  sum(mini_list) / len(mini_list),
    }


# ---------------------------------------------------------------------------
# Latent dataset extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latent_dataset(
    dataloader: DataLoader,
    encoder: VAE_Encoder,
    device: torch.device,
    output_path: str,
    logger: logging.Logger,
):
    """
    Run the trained encoder over the full dataset, collect all latent
    vectors with their labels, and save to a single .pt file.

    Output format (dict[str, dict]):
        {
            "0": { "data": Tensor(4, L/8), "label": {...} },
            "1": { "data": Tensor(4, L/8), "label": {...} },
            ...
        }
    This format is directly compatible with the Phase 2 DictDataset loader.
    """
    encoder.eval()
    latent_dict = {}
    global_idx = 0

    for ecg, labels in dataloader:
        ecg = ecg.to(device)
        z, _, _ = encoder(ecg)          # (B, 4, L/8)

        B = z.shape[0]
        for b in range(B):
            label_b = {}
            for k, v in labels.items():
                if k == "text_embed":
                    if isinstance(v, torch.Tensor):
                        label_b[k] = [float(x) for x in v[b].cpu()]
                    elif isinstance(v, (list, tuple)):
                        label_b[k] = [float(item[b]) for item in v]
                elif isinstance(v, (list, tuple)):
                    label_b[k] = v[b]
                elif isinstance(v, torch.Tensor):
                    if v.ndim == 0:
                        label_b[k] = v.item()
                    elif v.ndim == 1:
                        label_b[k] = v[b].item()
                    else:
                        label_b[k] = v[b].cpu()
                else:
                    label_b[k] = v
            latent_dict[str(global_idx)] = {
                "data": z[b].cpu(),
                "label": label_b,
            }
            global_idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(latent_dict, output_path)
    logger.info(f"Saved {global_idx} latent samples to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SE-Diff Phase 1 Training")
    parser.add_argument(
        "--config",
        default=str(repo_path("configs", "train_phase1.json")),
        help="Path to Phase 1 training configuration JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = ensure_exists(args.config, "phase1 training config")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    meta   = config["meta"]
    roots  = config["dependencies"]
    h_     = config["hyper_para"]

    # ---- Checkpoint directory ----
    k_max = 0
    os.makedirs(roots["checkpoints_dir"], exist_ok=True)
    for item in os.listdir(roots["checkpoints_dir"]):
        if meta["exp_type"] + "_" in item:
            k = int(item.split("_")[-1])
            k_max = k if k > k_max else k_max
    save_dir = os.path.join(roots["checkpoints_dir"],
                            f"{meta['exp_type']}_{k_max + 1}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Logger ----
    logger = logging.getLogger(f"{meta['exp_type']}_{k_max + 1}")
    logger.setLevel("INFO")
    fh = logging.FileHandler(os.path.join(save_dir, "train_phase1.log"),
                             encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ---- Device ----
    device = torch.device(
        meta["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ---- Dataset ----
    dataset_dir = ensure_exists(roots["dataset_dir"], "raw ECG dataset directory")
    dataset = RawECGDataset(str(dataset_dir))
    dataloader = DataLoader(
        dataset,
        batch_size=h_["batch_size"],
        shuffle=True,
        num_workers=h_.get("num_workers", 4),
        pin_memory=True,
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    # ---- Models ----
    encoder     = VAE_Encoder().to(device)
    decoder     = VAE_Decoder().to(device)
    mini_decoder = MiniDecoderMSE(target_len=61, z_channels=4, base=128).to(device)

    # ---- Optimizer & Scheduler ----
    params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(mini_decoder.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=h_["lr"])
    total_steps = h_["epochs"] * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.1 * h_["lr"]
    )

    # ---- Training loop ----
    min_loss   = math.inf
    start_time = time.time()
    save_every = h_.get("save_every", 10)

    for epoch in range(1, h_["epochs"] + 1):
        t0 = time.time()
        losses = train_epoch(
            dataloader=dataloader,
            encoder=encoder,
            decoder=decoder,
            mini_decoder=mini_decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            kld_weight=h_["kld_weight"],
            mini_decoder_weight=h_["mini_decoder_weight"],
            logger=logger,
        )
        t1 = time.time()

        logger.info(
            f"Epoch {epoch:04d} | "
            f"total={losses['total']:.4f}  "
            f"recon={losses['recon']:.4f}  "
            f"kld={losses['kld']:.4f}  "
            f"mini={losses['mini']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={t1 - t0:.1f}s"
        )

        # Save best checkpoint (all three modules)
        if losses["total"] < min_loss:
            min_loss = losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "mini_decoder_state_dict": mini_decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": min_loss,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            logger.info(f"  -> best_model.pth saved (loss={min_loss:.4f})")

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "mini_decoder_state_dict": mini_decoder.state_dict(),
                },
                os.path.join(save_dir, f"model_epoch_{epoch}.pth"),
            )

    total_time = time.time() - start_time
    logger.info(f"Training finished. Total time: {total_time / 60:.1f} min")

    # ---- Extract and save latent dataset for Phase 2 ----
    logger.info("Extracting latent dataset for Phase 2...")
    # Reload best weights before extraction
    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pth"),
                           map_location="cpu")
    encoder.load_state_dict(best_ckpt["encoder_state_dict"])
    encoder.to(device)

    extract_dataloader = DataLoader(
        dataset,
        batch_size=h_["batch_size"],
        shuffle=False,
        num_workers=h_.get("num_workers", 4),
        pin_memory=True,
    )
    extract_latent_dataset(
        dataloader=extract_dataloader,
        encoder=encoder,
        device=device,
        output_path=roots["output_latent_path"],
        logger=logger,
    )

    # ---- Save individual prerequisite weights for Phase 2 ----
    prerequisites_dir = repo_path("prerequisites")
    os.makedirs(prerequisites_dir, exist_ok=True)

    torch.save(
        {"decoder": decoder.state_dict()},
        str(prerequisites_dir / "vae_decoder.pth"),
    )
    torch.save(
        {"model_state_dict": mini_decoder.state_dict()},
        str(prerequisites_dir / "mini_decoder.pth"),
    )
    logger.info(
        f"Saved vae_decoder.pth and mini_decoder.pth to {prerequisites_dir}"
    )
    logger.info("Phase 1 complete. Ready to run Phase 2 training.")


if __name__ == "__main__":
    main()
