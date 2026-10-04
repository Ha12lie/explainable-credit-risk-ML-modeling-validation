"""README Step 13: run explicit macroeconomic sensitivity scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_DATA = RESULTS / "03_modeling_dataset.csv"
DEFAULT_MODEL = RESULTS / "08_model_artifact.joblib"
DEFAULT_SPLIT = RESULTS / "06_split.json"
DEFAULT_OUTPUT = RESULTS / "13_stress_test.csv"


def calibrated_probability(calibrator, raw_probability: np.ndarray) -> np.ndarray:
    eps = 1e-6
    logits = np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def run_stress(data_path: Path, model_path: Path, split_path: Path, output_path: Path) -> None:
    artifact = joblib.load(model_path)
    metadata = artifact["metadata"]
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in metadata["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metadata["categorical"]:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test = frame["date"].dt.to_period("M") > pd.Period(split["calibration_through"])
    base_features = frame.loc[test, metadata["features"]].copy()
    model = artifact["model"]
    calibrator = artifact["calibrator"]
    scenarios = {
        "baseline": (0.0, 0.0, 0.0),
        "moderate": (1.0, 10.0, -0.15),
        "severe": (2.0, 20.0, -0.30),
    }
    rows = []
    macro_columns = metadata["blocks"]["macro"]
    for scenario, shocks in scenarios.items():
        stressed = base_features.copy()
        for column, shock in zip(macro_columns, shocks):
            stressed[column] = pd.to_numeric(stressed[column], errors="coerce") + shock
        raw = model.predict_proba(stressed)[:, 1]
        probability = calibrated_probability(calibrator, raw)
        rows.append({"scenario": scenario, "mean_pd": float(probability.mean()), "median_pd": float(np.median(probability))})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Step 13 complete: saved macro stress scenarios to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.data, args.model, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    run_stress(args.data, args.model, args.split, args.output)


if __name__ == "__main__":
    main()
