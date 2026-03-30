import argparse
import json
import os

from evaluation_metrics import analyze_results_for_config
from utils.path_utils import repo_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ECG generation results with the SE-Diff main metrics pipeline."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=str(repo_path("outputs", "generation", "evaluation_output.json")),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_path("outputs", "evaluation")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    analyze_results_for_config(
        all_results,
        ecg_encoder_variant="v0",
        text_encoder_variant="v0",
        args=args,
        config_id="A",
    )
    analyze_results_for_config(
        all_results,
        ecg_encoder_variant="v1",
        text_encoder_variant="v0",
        args=args,
        config_id="B",
    )
    analyze_results_for_config(
        all_results,
        ecg_encoder_variant="v0",
        text_encoder_variant="v1",
        args=args,
        config_id="C",
    )


if __name__ == "__main__":
    main()
