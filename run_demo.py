"""
run_demo.py
===========
End-to-End Demo Script for SE-Diff (Phase 1 & Phase 2).

This script performs the following steps automatically:
  Step 1: Generates synthetic 12-lead ECG raw data & simulator prior pickle.
  Step 2: Executes Phase 1 training (VAE Encoder/Decoder + MiniDecoderMSE).
          - Produces best_model.pth for Phase 1.
          - Extracts training_latent_dataset.pt for Phase 2.
          - Saves vae_decoder.pth & mini_decoder.pth prerequisites.
  Step 3: Executes Phase 2 training (Conditional 1D UNet Diffusion Model).
          - Trains UNet with text embedding, demographic condition, and simulator loss.
  Step 4: Runs generation test (generate.py) using the trained weights to produce
          synthetic ECG JSON outputs.

Usage
-----
  python run_demo.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_step(title, command):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"Running: {' '.join(command)}\n")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] {title} completed.")

def find_latest_checkpoint(checkpoints_dir, prefix="phase2_demo"):
    if not os.path.exists(checkpoints_dir):
        return None
    k_max = -1
    latest_ckpt = None
    for item in os.listdir(checkpoints_dir):
        if prefix + "_" in item:
            try:
                k = int(item.split("_")[-1])
                if k > k_max:
                    folder = os.path.join(checkpoints_dir, item)
                    pths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pth")]
                    if pths:
                        k_max = k
                        if os.path.exists(os.path.join(folder, "best_model.pth")):
                            latest_ckpt = os.path.join(folder, "best_model.pth")
                        else:
                            pths.sort(key=lambda x: os.path.getmtime(x))
                            latest_ckpt = pths[-1]
            except ValueError:
                pass
    return latest_ckpt

def main():
    repo_root = Path(__file__).resolve().parent
    checkpoints_dir = repo_root / "checkpoints"

    # Step 1: Generate synthetic raw data & simulator prior
    run_step(
        "STEP 1: Generating Synthetic Raw ECG Data & Simulator Prior",
        [sys.executable, str(repo_root / "prepare_synthetic_demo.py")]
    )

    # Step 2: Run Phase 1 Training
    run_step(
        "STEP 2: Running Phase 1 Training (VAE & MiniDecoder)",
        [sys.executable, str(repo_root / "train_phase1.py"), "--config", str(repo_root / "configs" / "train_phase1_demo.json")]
    )

    # Step 3: Run Phase 2 Training
    run_step(
        "STEP 3: Running Phase 2 Training (Conditional UNet Diffusion Model)",
        [sys.executable, str(repo_root / "train.py"), "--config", str(repo_root / "configs" / "train_demo.json")]
    )

    # Step 4: Run Generation Test
    unet_ckpt = find_latest_checkpoint(str(checkpoints_dir), prefix="phase2_demo")
    if not unet_ckpt:
        unet_ckpt = str(checkpoints_dir / "phase2_demo_1" / "best_model.pth")

    run_step(
        "STEP 4: Running Generation Test (Inference)",
        [
            sys.executable, str(repo_root / "generate_demo.py"),
            "--data_path", str(repo_root / "prerequisites" / "training_latent_dataset.pt"),
            "--unet_path", unet_ckpt,
            "--vae_path", str(repo_root / "prerequisites" / "vae_decoder.pth"),
            "--num_samples", "2",
        ]
    )

    print("\n" + "=" * 70)
    print("  ALL SE-DIFF DEMO STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated Artifacts:")
    print("  - Raw Dataset      : prerequisites/raw_ecg_data/ (50 samples)")
    print("  - Simulator Prior  : prerequisites/simulator_prior.pkl")
    print("  - Latent Dataset   : prerequisites/training_latent_dataset.pt")
    print("  - VAE Decoder      : prerequisites/vae_decoder.pth")
    print("  - Mini Decoder     : prerequisites/mini_decoder.pth")
    print(f"  - Latest UNet Ckpt : {unet_ckpt}")
    print("  - Generated Output : outputs/generation/evaluation_output.json")

if __name__ == "__main__":
    main()
