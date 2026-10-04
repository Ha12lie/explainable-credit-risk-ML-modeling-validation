"""README Step 8: fit Platt scaling and create calibrated test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_DATA = RESULTS / "03_modeling_dataset.csv"
DEFAULT_CANDIDATES = RESULTS / "07_model_candidates.joblib"
DEFAULT_SPLIT = RESULTS / "06_split.json"
DEFAULT_MODEL = RESULTS / "08_model_artifact.joblib"
DEFAULT_PREDICTIONS = RESULTS / "08_test_predictions.csv"


def fit_platt_scaler(raw_probability: np.ndarray, target: pd.Series) -> LogisticRegression:
    eps = 1e-6
    logits = np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps)).reshape(-1, 1)
    return LogisticRegression(max_iter=1000).fit(logits, target)


def apply_platt_scaler(calibrator: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    eps = 1e-6
    logits = np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def calibrate_model(data_path: Path, candidates_path: Path, split_path: Path, model_path: Path, prediction_path: Path) -> None:
    candidate_artifact = joblib.load(candidates_path)
    metadata = candidate_artifact["metadata"]
    selected = candidate_artifact["selected_model"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in metadata["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metadata["categorical"]:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    periods = frame["date"].dt.to_period("M")
    calibration = (periods > pd.Period(split["train_through"])) & (periods <= pd.Period(split["calibration_through"]))
    test = periods > pd.Period(split["calibration_through"])
    target = frame[metadata["target"]].astype(int)
    model = candidate_artifact["candidates"][selected]
    calibration_raw = model.predict_proba(frame.loc[calibration, metadata["features"]])[:, 1]
    calibrator = fit_platt_scaler(calibration_raw, target.loc[calibration])
    raw = model.predict_proba(frame.loc[test, metadata["features"]])[:, 1]
    calibrated = apply_platt_scaler(calibrator, raw)
    predictions = pd.DataFrame(
        {
            "Loan Identifier": frame.loc[test, "Loan Identifier"].to_numpy(),
            "observation_month": frame.loc[test, "date"].to_numpy(),
            "target_3m": target.loc[test].to_numpy(),
            "pd_raw": raw,
            "pd_calibrated": calibrated,
        }
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    joblib.dump({"model": model, "calibrator": calibrator, "selected": selected, "metadata": metadata}, model_path)
    model_path.with_name("08_model_metadata.json").write_text(json.dumps({"selected_model": selected}, indent=2), encoding="utf-8")
    print(f"Step 8 complete: calibrated {selected} and saved predictions to {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()
    for path in (args.data, args.candidates, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    calibrate_model(args.data, args.candidates, args.split, args.model_output, args.prediction_output)


if __name__ == "__main__":
    main()
