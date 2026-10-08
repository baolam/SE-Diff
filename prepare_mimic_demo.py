"""
prepare_mimic_demo.py
=====================
Download a small subset of MIMIC-IV-ECG from PhysioNet and convert it
to the .pt format expected by SE-Diff Phase 1 training.

What this script does
---------------------
1. Reads the MIMIC-IV-ECG record list (records.csv or machine_measurements.csv).
2. Downloads N waveform records via the `wfdb` library (streaming, no full
   dataset download required).
3. Resamples each 12-lead signal from 500 Hz to 102.4 Hz.
4. Saves one .pt file per record to `prerequisites/raw_ecg_data/`.
5. Prints a summary table.

Requirements
------------
  pip install wfdb neurokit2 scipy numpy torch tqdm

PhysioNet credentials
---------------------
You MUST have a PhysioNet account and have been granted access to MIMIC-IV-ECG.
Set your credentials as environment variables before running:

  set PHYSIONET_USER=<your_username>
  set PHYSIONET_PASSWORD=<your_password>

Or pass them via --user / --password CLI flags.

Usage
-----
  python prepare_mimic_demo.py --n_records 50 --out_dir ./prerequisites/raw_ecg_data

  # Dry-run: list available records without downloading
  python prepare_mimic_demo.py --list_only
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHYSIONET_DB      = "mimic-iv-ecg/1.0"          # PhysioNet database path
TARGET_FS         = 102.4                        # Hz used by SE-Diff internally
SOURCE_FS         = 500                          # MIMIC-IV-ECG native sampling rate
RECORD_DURATION_S = 10                           # Each record is 10 s
N_LEADS           = 12

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Approximate integer ratio for resampling: 500 -> 512 is close to 102.4*5
# We use resample_poly(up=512, down=2500) which gives exactly 500*(512/2500)=102.4
RESAMPLE_UP   = 512
RESAMPLE_DOWN = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_mimic_records(user: str, password: str, max_records: int = 200) -> list[str]:
    """
    Retrieve the first `max_records` waveform record paths from MIMIC-IV-ECG.
    Returns a list of strings like 'files/p100/p10000032/s12345678'.
    """
    print("Fetching record list from PhysioNet...")
    try:
        # wfdb.get_record_list returns all record names in the database
        records = wfdb.get_record_list(
            db_dir=PHYSIONET_DB,
            pn_dir=PHYSIONET_DB,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch record list from PhysioNet.\n"
            f"Ensure your credentials are correct and you have access to {PHYSIONET_DB}.\n"
            f"Original error: {e}"
        )
    return records[:max_records]


def download_record(record_path: str, user: str, password: str):
    """
    Stream a single WFDB record from PhysioNet without saving raw files.
    Returns (signal_np, fields_dict) or (None, None) on failure.
    """
    try:
        record = wfdb.rdrecord(
            record_name=record_path,
            pn_dir=PHYSIONET_DB,
        )
        # p_signal: (n_samples, n_leads)
        return record.p_signal, record
    except Exception as e:
        warnings.warn(f"Failed to download {record_path}: {e}")
        return None, None


def resample_ecg(signal: np.ndarray, source_fs: float = SOURCE_FS) -> np.ndarray:
    """
    Resample ECG from source_fs to TARGET_FS (102.4 Hz).

    Parameters
    ----------
    signal : np.ndarray  (n_samples, n_leads)

    Returns
    -------
    np.ndarray  (n_resampled, n_leads)
    """
    resampled = resample_poly(signal, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=0)
    return resampled.astype(np.float32)


def reorder_leads(signal: np.ndarray, sig_name: list[str]) -> np.ndarray:
    """
    Reorder leads to canonical SE-Diff order: I II III aVR aVL aVF V1..V6.
    Missing leads are filled with zeros.

    Parameters
    ----------
    signal   : (n_samples, n_leads_available)
    sig_name : list of available lead names

    Returns
    -------
    (n_samples, 12)
    """
    n_samples = signal.shape[0]
    out = np.zeros((n_samples, N_LEADS), dtype=np.float32)
    for dst_idx, lead in enumerate(LEAD_ORDER):
        if lead in sig_name:
            src_idx = sig_name.index(lead)
            out[:, dst_idx] = signal[:, src_idx]
    return out


def extract_label(record, record_path: str) -> dict:
    """
    Extract the label dictionary from a WFDB record object.
    Fields: text, age, gender, hr, subject_id.
    """
    # Subject ID from path: files/p100/p10000032/s12345678 -> 10000032
    parts = record_path.replace("\\", "/").split("/")
    subject_id = 0
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            subject_id = int(part[1:])

    comments = record.comments if record.comments else []
    age, gender, hr = 0, "U", 0

    for line in comments:
        line_lower = line.lower()
        if "age:" in line_lower:
            try:
                age = int(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass
        if "sex:" in line_lower or "gender:" in line_lower:
            val = line.split(":")[-1].strip().upper()
            gender = "M" if val.startswith("M") else "F" if val.startswith("F") else "U"
        if "heart rate:" in line_lower or "ventricular rate:" in line_lower:
            try:
                hr = int(line.split(":")[-1].strip().split()[0])
            except Exception:
                pass

    # Build a minimal text description from available info
    rhythm_keywords = [
        "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
        "atrial fibrillation", "atrial flutter", "ventricular tachycardia",
        "normal ecg", "normal sinus rhythm",
    ]
    text_parts = []
    full_comment = " ".join(comments).lower()
    for kw in rhythm_keywords:
        if kw in full_comment:
            text_parts.append(kw)
    text = "|".join(text_parts) if text_parts else "normal sinus rhythm"

    return {
        "text":       text,
        "age":        age,
        "gender":     gender,
        "hr":         hr,
        "subject_id": subject_id,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a mini demo subset of MIMIC-IV-ECG for SE-Diff Phase 1."
    )
    parser.add_argument(
        "--n_records", type=int, default=50,
        help="Number of records to download (default: 50)."
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(Path(__file__).parent / "prerequisites" / "raw_ecg_data"),
        help="Output directory for .pt files."
    )
    parser.add_argument(
        "--user", type=str,
        default=os.environ.get("PHYSIONET_USER", ""),
        help="PhysioNet username. Can also be set via PHYSIONET_USER env var."
    )
    parser.add_argument(
        "--password", type=str,
        default=os.environ.get("PHYSIONET_PASSWORD", ""),
        help="PhysioNet password. Can also be set via PHYSIONET_PASSWORD env var."
    )
    parser.add_argument(
        "--list_only", action="store_true",
        help="Only list available records without downloading."
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N records (useful to avoid already-downloaded ones)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.user or not args.password:
        print(
            "[ERROR] PhysioNet credentials are required.\n"
            "  Set PHYSIONET_USER and PHYSIONET_PASSWORD environment variables, or\n"
            "  pass --user and --password flags.\n"
            "\n"
            "  Example:\n"
            "    set PHYSIONET_USER=your_username\n"
            "    set PHYSIONET_PASSWORD=your_password\n"
            "    python prepare_mimic_demo.py --n_records 50\n"
        )
        return

    # Configure wfdb credentials
    wfdb.io.config.DB_INDEX_URL = "https://physionet.org/files/"
    os.environ["WFDB_PHYSIONET_USER"]     = args.user
    os.environ["WFDB_PHYSIONET_PASSWORD"] = args.password

    # Fetch record list
    try:
        records = list_mimic_records(
            user=args.user,
            password=args.password,
            max_records=args.offset + args.n_records,
        )
    except RuntimeError as e:
        print(e)
        return

    records = records[args.offset: args.offset + args.n_records]
    print(f"  Found {len(records)} target records.")

    if args.list_only:
        for r in records:
            print(f"  {r}")
        return

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download & process
    success, failed = 0, 0
    for i, record_path in enumerate(tqdm(records, desc="Downloading records")):
        out_file = out_dir / f"{i + args.offset:06d}.pt"
        if out_file.exists():
            success += 1
            continue

        signal, record = download_record(record_path, args.user, args.password)
        if signal is None:
            failed += 1
            continue

        # Handle NaN / Inf from WFDB
        if np.any(~np.isfinite(signal)):
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Reorder to canonical 12-lead order
        signal = reorder_leads(signal, record.sig_name)

        # Resample: 500 Hz -> 102.4 Hz
        signal = resample_ecg(signal, source_fs=record.fs)

        # Convert to Tensor (L, 12)
        data_tensor = torch.tensor(signal, dtype=torch.float32)

        # Build label dict
        label = extract_label(record, record_path)

        # Save
        torch.save({"data": data_tensor, "label": label}, out_file)
        success += 1

    print(f"\n{'=' * 50}")
    print(f"Done! Saved to: {out_dir}")
    print(f"  Successful : {success}")
    print(f"  Failed     : {failed}")
    print(f"  Total .pt  : {len(list(out_dir.glob('*.pt')))}")
    print(f"\nNext step — train Phase 1:")
    print(f"  python train_phase1.py --config configs/train_phase1.json")


if __name__ == "__main__":
    main()
