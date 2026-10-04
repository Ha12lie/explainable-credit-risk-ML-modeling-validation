"""Combined README Steps 1-6: build the modeling data and split artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [SCRIPT_ROOT / "loans_2025_panel_model_fields.csv", SCRIPT_ROOT / "loans_2026_panel_model_fields.csv"]
ID = "Loan Identifier"
MONTH = "Monthly Reporting Period"
HORIZON = 3
BLOCKS = {
    "bureau": ["Borrower Credit Score at Origination", "Debt-To-Income (DTI)", "Original Loan to Value Ratio (LTV)", "Original UPB", "Original Interest Rate"],
    "dealer_proxy": ["Seller Name", "Servicer Name", "Channel"],
    "alternative": ["MSA", "Occupancy Status", "Property Type", "Number of Units"],
    "macro": ["tnx", "vix", "spy_return_12m"],
}
NUMERIC = BLOCKS["bureau"] + ["Number of Units"] + BLOCKS["macro"]
FEATURES = [column for block in BLOCKS.values() for column in block]
REQUIRED_COLUMNS = {
    ID, MONTH, "Current Loan Delinquency Status", "Borrower Credit Score at Origination", "Debt-To-Income (DTI)",
    "Original Loan to Value Ratio (LTV)", "Original UPB", "Original Interest Rate", "Seller Name", "Servicer Name",
    "Channel", "MSA", "Occupancy Status", "Property Type", "Number of Units", "Loan Age", "Zero Balance Code", "Modification Flag",
}


def prepare_panel(input_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input panel does not exist: {path}")
        frame = pd.read_csv(path, low_memory=False)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel[ID] = panel[ID].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(12)
    panel[MONTH] = panel[MONTH].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    panel["date"] = pd.to_datetime(panel[MONTH], format="%m%Y", errors="coerce")
    panel["dq_num"] = pd.to_numeric(panel["Current Loan Delinquency Status"], errors="coerce")
    panel = panel.dropna(subset=[ID, "date", "dq_num"]).sort_values([ID, "date"])
    return panel.drop_duplicates([ID, "date"], keep="last").reset_index(drop=True)


def construct_target(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["bad"] = frame["dq_num"].ge(1)
    for step in range(1, HORIZON + 1):
        frame[f"date_{step}"] = frame.groupby(ID)["date"].shift(-step)
        frame[f"bad_{step}"] = frame.groupby(ID)["bad"].shift(-step)
    continuous = pd.Series(True, index=frame.index)
    current_period = frame["date"].dt.to_period("M")
    for step in range(1, HORIZON + 1):
        gap = frame[f"date_{step}"].dt.to_period("M") - current_period
        continuous &= gap.map(lambda value: value.n if pd.notna(value) else -1).eq(step)
    observations = frame.loc[continuous & ~frame["bad"]].copy()
    observations["target"] = False
    for step in range(1, HORIZON + 1):
        observations["target"] |= observations[f"bad_{step}"].fillna(False).astype(bool)
    observations["target"] = observations["target"].astype(int)
    return observations.drop_duplicates(ID, keep="first").reset_index(drop=True)


def add_macro_features(target_panel: pd.DataFrame) -> pd.DataFrame:
    frame = target_panel.copy()
    start = (frame["date"].min() - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    end = (frame["date"].max() + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    prices = yf.download(["^TNX", "^VIX", "SPY"], start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if prices.empty or "Close" not in prices:
        raise RuntimeError("Yahoo Finance returned no usable Close prices.")
    close = prices["Close"]
    monthly = close.resample("ME").last()
    macro = pd.DataFrame({"tnx": monthly["^TNX"], "vix": monthly["^VIX"], "spy_return_12m": monthly["SPY"].pct_change(12, fill_method=None)})
    macro.index = (macro.index.to_period("M") + 1).to_timestamp()
    return frame.merge(macro, left_on="date", right_index=True, how="left").dropna(subset=NUMERIC[-3:]).reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=True)
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), [c for c in FEATURES if c not in NUMERIC]),
    ])


def build_split(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    periods = frame["date"].dt.to_period("M")
    months = sorted(periods.dropna().unique())
    if len(months) < 3:
        raise ValueError("At least three unique observation months are required.")
    train_cut = months[max(0, int(len(months) * 0.60) - 1)]
    calibration_cut = months[max(1, int(len(months) * 0.80) - 1)]
    train = periods <= train_cut
    calibration = (periods > train_cut) & (periods <= calibration_cut)
    test = periods > calibration_cut
    plan = {"months": [str(month) for month in months], "train_through": str(train_cut), "calibration_through": str(calibration_cut), "counts": {"train": int(train.sum()), "calibration": int(calibration.sum()), "test": int(test.sum())}}
    return train, calibration, test, plan


def run_pipeline(input_paths: list[Path], output_dir: Path) -> None:
    panel = prepare_panel(input_paths)
    panel.to_csv(output_dir / "01_merged_panel.csv", index=False)
    target_panel = construct_target(panel)
    target_panel.to_csv(output_dir / "02_target_panel.csv", index=False)
    modeling_data = add_macro_features(target_panel)
    modeling_data.to_csv(output_dir / "03_modeling_dataset.csv", index=False)
    metadata = {"blocks": BLOCKS, "features": FEATURES, "numeric": NUMERIC, "categorical": [c for c in FEATURES if c not in NUMERIC], "target": "target", "date": "date"}
    (output_dir / "04_feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    modeling_data["date"] = pd.to_datetime(modeling_data["date"])
    for column in NUMERIC:
        modeling_data[column] = pd.to_numeric(modeling_data[column], errors="coerce")
    for column in metadata["categorical"]:
        modeling_data[column] = modeling_data[column].astype("string").fillna("Unknown")
    train, _, _, split = build_split(modeling_data)
    preprocessor = make_preprocessor()
    preprocessor.fit(modeling_data.loc[train, FEATURES])
    joblib.dump({"preprocessor": preprocessor, **metadata}, output_dir / "05_preprocessor.joblib")
    (output_dir / "05_preprocessing_summary.json").write_text(json.dumps({"training_rows": int(train.sum()), "transformed_features": len(preprocessor.get_feature_names_out())}, indent=2), encoding="utf-8")
    (output_dir / "06_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Combined Steps 1-6 complete: {len(modeling_data):,} modeling rows")
    print(f"Artifacts written to {output_dir}; split counts={split['counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
