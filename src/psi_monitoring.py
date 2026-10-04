"""README Step 12: compare train/test feature distributions with PSI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_DATA = RESULTS / "03_modeling_dataset.csv"
DEFAULT_METADATA = RESULTS / "04_feature_metadata.json"
DEFAULT_SPLIT = RESULTS / "06_split.json"
DEFAULT_OUTPUT = RESULTS / "12_psi_report.csv"


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate PSI with explicit missing and out-of-range bins."""
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    if pd.api.types.is_numeric_dtype(expected):
        expected_numeric = pd.to_numeric(expected, errors="coerce")
        actual_numeric = pd.to_numeric(actual, errors="coerce")
        finite = expected_numeric.dropna()
        if finite.empty:
            return 0.0
        edges = np.unique(np.nanquantile(finite, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        expected_bin = pd.cut(expected_numeric, bins=edges, include_lowest=True).astype("string")
        actual_bin = pd.cut(actual_numeric, bins=edges, include_lowest=True).astype("string")
        expected_bin = expected_bin.fillna("Missing_or_outside")
        actual_bin = actual_bin.fillna("Missing_or_outside")
    else:
        expected_bin = expected.astype("string").fillna("Missing")
        actual_bin = actual.astype("string").fillna("Missing")
        common = expected_bin.value_counts().head(20).index
        expected_bin = expected_bin.where(expected_bin.isin(common), "Other")
        actual_bin = actual_bin.where(actual_bin.isin(common), "Other")
    expected_dist = expected_bin.value_counts(normalize=True)
    actual_dist = actual_bin.value_counts(normalize=True)
    categories = expected_dist.index.union(actual_dist.index)
    expected_dist = expected_dist.reindex(categories, fill_value=1e-4).clip(lower=1e-4)
    actual_dist = actual_dist.reindex(categories, fill_value=1e-4).clip(lower=1e-4)
    return float(((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum())


def run_psi(data_path: Path, metadata_path: Path, split_path: Path, output_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    periods = frame["date"].dt.to_period("M")
    train = periods <= pd.Period(split["train_through"])
    test = periods > pd.Period(split["calibration_through"])
    rows = []
    for feature in metadata["features"]:
        value = calculate_psi(frame.loc[train, feature], frame.loc[test, feature])
        interpretation = "stable" if value < 0.10 else "moderate_shift" if value < 0.25 else "significant_shift"
        rows.append({"feature": feature, "psi": value, "interpretation": interpretation})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Step 12 complete: saved train/test PSI report to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.data, args.metadata, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    run_psi(args.data, args.metadata, args.split, args.output)


if __name__ == "__main__":
    main()
