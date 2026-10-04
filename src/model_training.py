"""README Step 7: train challenger models and select the best model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RESULTS / "03_modeling_dataset.csv"
DEFAULT_METADATA = RESULTS / "04_feature_metadata.json"
DEFAULT_SPLIT = RESULTS / "06_split.json"
DEFAULT_OUTPUT = RESULTS / "07_model_candidates.joblib"
SEED = 2026


def make_model(columns: list[str], numeric: list[str], kind: str) -> Pipeline:
    """Build a complete preprocessing-plus-estimator pipeline."""
    numeric_columns = [column for column in columns if column in numeric]
    categorical_columns = [column for column in columns if column not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=True)
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_columns),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_columns),
        ]
    )
    if kind == "logistic":
        estimator = LogisticRegression(max_iter=1500, class_weight="balanced")
    elif kind == "forest":
        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("prep", prep), ("model", estimator)])


def train_candidates(data_path: Path, metadata_path: Path, split_path: Path, output_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in metadata["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metadata["categorical"]:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    periods = frame["date"].dt.to_period("M")
    train = periods <= pd.Period(split["train_through"])
    calibration = (periods > pd.Period(split["train_through"])) & (periods <= pd.Period(split["calibration_through"]))
    target = frame[metadata["target"]].astype(int)
    candidates = {}
    scores = {}
    for kind in ("logistic", "forest"):
        model = make_model(metadata["features"], metadata["numeric"], kind)
        model.fit(frame.loc[train, metadata["features"]], target.loc[train])
        calibration_score = model.predict_proba(frame.loc[calibration, metadata["features"]])[:, 1]
        candidates[kind] = model
        scores[kind] = float(roc_auc_score(target.loc[calibration], calibration_score))
    selected = max(scores, key=scores.get)
    artifact = {"candidates": candidates, "selected_model": selected, "calibration_auc": scores, "metadata": metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    output_path.with_name("07_model_selection.json").write_text(json.dumps({"selected_model": selected, "calibration_auc": scores}, indent=2), encoding="utf-8")
    print(f"Step 7 complete: selected {selected}; calibration AUCs={scores}")
    print(f"Saved trained challenger pipelines to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.input, args.metadata, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    train_candidates(args.input, args.metadata, args.split, args.output)


if __name__ == "__main__":
    main()
