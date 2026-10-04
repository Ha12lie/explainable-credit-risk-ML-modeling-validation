"""README Step 11: retrain the selected model with each feature block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_DATA = RESULTS / "03_modeling_dataset.csv"
DEFAULT_METADATA = RESULTS / "04_feature_metadata.json"
DEFAULT_MODEL_METADATA = RESULTS / "08_model_metadata.json"
DEFAULT_SPLIT = RESULTS / "06_split.json"
DEFAULT_OUTPUT = RESULTS / "11_ablation.csv"
SEED = 2026


def make_model(columns: list[str], numeric: list[str], kind: str) -> Pipeline:
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
    estimator = LogisticRegression(max_iter=1500, class_weight="balanced") if kind == "logistic" else RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def run_ablation(data_path: Path, metadata_path: Path, model_metadata_path: Path, split_path: Path, output_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_model = json.loads(model_metadata_path.read_text(encoding="utf-8"))["selected_model"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in metadata["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metadata["categorical"]:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    periods = frame["date"].dt.to_period("M")
    train = periods <= pd.Period(split["train_through"])
    test = periods > pd.Period(split["calibration_through"])
    target = frame[metadata["target"]].astype(int)
    rows = []
    for name, columns in {"all": metadata["features"], **metadata["blocks"]}.items():
        fitted = make_model(columns, metadata["numeric"], selected_model)
        fitted.fit(frame.loc[train, columns], target.loc[train])
        score = fitted.predict_proba(frame.loc[test, columns])[:, 1]
        rows.append(
            {
                "feature_set": name,
                "roc_auc": roc_auc_score(target.loc[test], score),
                "pr_auc": average_precision_score(target.loc[test], score),
                "brier": brier_score_loss(target.loc[test], score),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Step 11 complete: saved feature-block ablation results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.data, args.metadata, args.model_metadata, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    run_ablation(args.data, args.metadata, args.model_metadata, args.split, args.output)


if __name__ == "__main__":
    main()
