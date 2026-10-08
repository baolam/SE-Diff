"""
prepare_synthetic_demo.py
=========================
Generates synthetic 12-lead ECG data and simulator prior files for SE-Diff demo testing.
This allows running the full Phase 1 and Phase 2 training pipelines without requiring
physionet credentials or downloading large datasets.
"""

import os
import pickle
import random
from pathlib import Path
import numpy as np
import torch

try:
    import neurokit2 as nk
    HAS_NK = True
except Exception:
    HAS_NK = False

def generate_synthetic_ecg_lead(duration=10.0, sampling_rate=102.4, heart_rate=60, noise_std=0.01):
    """Generate a single-lead synthetic ECG wave."""
    if HAS_NK:
        try:
            return nk.ecg_simulate(duration=duration, sampling_rate=int(sampling_rate), heart_rate=heart_rate)
        except Exception:
            pass

    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)
    freq = heart_rate / 60.0
    ecg = (np.sin(2 * np.pi * freq * t)
           + 0.5 * np.sin(4 * np.pi * freq * t)
           + 0.2 * np.sin(8 * np.pi * freq * t)
           + noise_std * np.random.randn(num_samples))
    return ecg

def generate_12_lead_ecg(duration=10.0, sampling_rate=102.4, heart_rate=60):
    """Generate synthetic 12-lead ECG signals."""
    num_samples = int(duration * sampling_rate)
    base_lead = generate_synthetic_ecg_lead(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate)
    
    if len(base_lead) < num_samples:
        base_lead = np.pad(base_lead, (0, num_samples - len(base_lead)))
    else:
        base_lead = base_lead[:num_samples]

    leads = np.zeros((num_samples, 12), dtype=np.float32)
    leads[:, 0] = base_lead                          # Lead I
    leads[:, 1] = base_lead * 1.2                    # Lead II
    leads[:, 2] = leads[:, 1] - leads[:, 0]          # Lead III = II - I
    leads[:, 3] = -(leads[:, 0] + leads[:, 1]) / 2.0 # aVR = -(I + II) / 2
    leads[:, 4] = (leads[:, 0] - leads[:, 2]) / 2.0  # aVL = (I - III) / 2
    leads[:, 5] = (leads[:, 1] + leads[:, 2]) / 2.0  # aVF = (II + III) / 2
    
    for v in range(6):
        scale = 0.8 + 0.1 * v
        leads[:, 6 + v] = base_lead * scale + 0.02 * np.random.randn(num_samples)
        
    return leads

def get_mock_text_embed(dim=1536):
    """Generate a dummy unit vector for text embedding (1536 float values)."""
    vec = np.random.randn(dim).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()

def create_synthetic_raw_dataset(out_dir, num_samples=50):
    """Create directory of raw .pt files for Phase 1 demo training."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    conditions = [
        "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
        "atrial fibrillation", "premature ventricular contractions"
    ]
    genders = ["M", "F"]

    print(f"Generating {num_samples} synthetic 12-lead ECG samples in '{out_dir}'...")
    for i in range(num_samples):
        hr = random.randint(50, 100)
        age = random.randint(20, 80)
        gender = random.choice(genders)
        cond = random.choice(conditions)
        
        signal = generate_12_lead_ecg(duration=10.0, sampling_rate=102.4, heart_rate=hr)
        
        sample = {
            "data": torch.tensor(signal, dtype=torch.float32), # (L, 12)
            "label": {
                "text": cond,
                "text_embed": get_mock_text_embed(1536),
                "age": age,
                "gender": gender,
                "hr": hr,
                "subject_id": 10000000 + i
            }
        }
        torch.save(sample, out_path / f"{i:06d}.pt")
    print(f"Successfully generated {num_samples} raw ECG samples.")

def create_simulator_prior_pkl(out_path):
    """Create a simulator prior pickle file with 61-len cycle templates for conditions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = [
        "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
        "atrial fibrillation", "atrial flutter", "supraventricular tachycardia",
        "ventricular tachycardia", "premature atrial contractions",
        "premature ventricular contractions", "left bundle branch block",
        "right bundle branch block", "left ventricular hypertrophy",
        "right ventricular hypertrophy", "wpw pattern", "prolonged qt",
        "st elevation", "st depression", "t wave inversion", "ischemia", "other"
    ]

    fz_val_dict = {}
    cycle_len = 61
    for cond in conditions:
        cycle_base = generate_synthetic_ecg_lead(duration=0.6, sampling_rate=102.4, heart_rate=75)[:cycle_len]
        if len(cycle_base) < cycle_len:
            cycle_base = np.pad(cycle_base, (0, cycle_len - len(cycle_base)))
            
        leads_61 = np.zeros((12, cycle_len), dtype=np.float32)
        for lead_idx in range(12):
            leads_61[lead_idx] = cycle_base * (0.8 + 0.05 * lead_idx)
        fz_val_dict[cond] = leads_61

    with open(out_path, "wb") as f:
        pickle.dump(fz_val_dict, f)
    print(f"Saved synthetic simulator prior to '{out_path}'.")

def main():
    repo_root = Path(__file__).resolve().parent
    raw_dir = repo_root / "prerequisites" / "raw_ecg_data"
    prior_path = repo_root / "prerequisites" / "simulator_prior.pkl"

    create_synthetic_raw_dataset(raw_dir, num_samples=50)
    create_simulator_prior_pkl(prior_path)

if __name__ == "__main__":
    main()
