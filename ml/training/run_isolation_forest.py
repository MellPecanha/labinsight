"""Run the first reproducible BIOPRO_L anomaly-detection experiment.

Usage from the repository root:
    .venv/bin/python -m ml.training.run_isolation_forest \
        --input data/raw/BIOPRO_L.xpt --output data/processed/isolation_forest.joblib
"""

import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from ml.preprocessing.biopro import load_biopro, select_model_features
from ml.training.isolation_forest import fit_isolation_forest, summarize_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BIOPRO_L Isolation Forest experiment.")
    parser.add_argument("--input", type=Path, required=True, help="Path to BIOPRO_L.xpt")
    parser.add_argument("--output", type=Path, required=True, help="Destination .joblib file")
    parser.add_argument("--anomaly-fraction", type=float, default=0.02)
    args = parser.parse_args()

    panel = select_model_features(load_biopro(args.input))
    train_data, holdout_data = train_test_split(panel, test_size=0.20, random_state=42)
    experiment = fit_isolation_forest(train_data, anomaly_fraction=args.anomaly_fraction)

    print("Holdout stability summary (not supervised accuracy):")
    for key, value in summarize_holdout(experiment, holdout_data).items():
        print(f"{key}: {value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(experiment, args.output)
    print(f"Saved experiment to {args.output}")


if __name__ == "__main__":
    main()
