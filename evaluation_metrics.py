import json
import os

import numpy as np


EPS = 1e-12


def ensure_leads_time(arr, expect_leads=12):
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"ECG array must be 2D, got {x.shape}")
    leads, time = x.shape
    if leads == expect_leads:
        return x
    if time == expect_leads:
        return x.T
    return x


def safe_corr(a, b):
    x, y = a.ravel(), b.ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + EPS
    return float(np.dot(x, y) / denom)


def calculate_mae(gt, gen):
    return float(np.mean(np.abs(gt - gen)))


def calculate_mse(gt, gen):
    return float(np.mean((gt - gen) ** 2))


def calculate_rmse(gt, gen):
    return float(np.sqrt(calculate_mse(gt, gen)))


def analyze_results_for_config(all_results, ecg_encoder_variant, text_encoder_variant, args, config_id):
    config_tag = f"cfg{config_id}_ecg_{ecg_encoder_variant}_txt_{text_encoder_variant}"
    detailed_results = []
    maes = []
    mses = []
    rmses = []
    corrs = []

    for sample in all_results:
        try:
            gt = ensure_leads_time(np.asarray(sample["ground_truth_ecg_data"]))
            gen = ensure_leads_time(np.asarray(sample["generated_ecg_data"]))

            mae = calculate_mae(gt, gen)
            mse = calculate_mse(gt, gen)
            rmse = calculate_rmse(gt, gen)
            corr = safe_corr(gt, gen)

            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            corrs.append(corr)

            detailed_results.append(
                {
                    "config": {
                        "id": config_id,
                        "ecg_encoder": ecg_encoder_variant,
                        "text_encoder": text_encoder_variant,
                    },
                    "sample_id": sample.get("sample_id"),
                    "conditions": sample.get("conditions", {}),
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "correlation": corr,
                }
            )
        except Exception as exc:
            print(f"[{config_tag}] Skipping sample {sample.get('sample_id', 'N/A')}: {exc}")

    summary = {
        "config": config_tag,
        "num_samples": len(detailed_results),
        "mae": float(np.mean(maes)) if maes else None,
        "mse": float(np.mean(mses)) if mses else None,
        "rmse": float(np.mean(rmses)) if rmses else None,
        "correlation": float(np.mean(corrs)) if corrs else None,
    }

    print(f"[{config_tag}] samples={summary['num_samples']} mae={summary['mae']} rmse={summary['rmse']} corr={summary['correlation']}")

    out_path = os.path.join(args.output_dir, f"evaluation_{config_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": detailed_results}, f, indent=2)
