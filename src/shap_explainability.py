"""README Step 10: calculate global SHAP feature importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_DATA = RESULTS / "03_modeling_dataset.csv"
DEFAULT_MODEL = RESULTS / "08_model_artifact.joblib"
DEFAULT_SPLIT = RESULTS / "06_split.json"


def calculate_shap(data_path: Path, model_path: Path, split_path: Path, output_dir: Path, sample_size: int = 100) -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Step 10 requires the optional 'shap' package. Install it with: pip install shap") from exc
    artifact = joblib.load(model_path)
    metadata = artifact["metadata"]
    frame = pd.read_csv(data_path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in metadata["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metadata["categorical"]:
        frame[column] = frame[column].astype("string").fillna("Unknown")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    periods = frame["date"].dt.to_period("M")
    test = periods > pd.Period(split["calibration_through"])
    sample = frame.loc[test, metadata["features"]].sample(n=min(sample_size, int(test.sum())), random_state=2026)
    model = artifact["model"]
    matrix = model.named_steps["prep"].transform(sample)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    feature_names = model.named_steps["prep"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if artifact["selected"] == "forest":
        background = matrix[: min(30, len(matrix))]
        explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation="interventional")
    else:
        explainer = shap.LinearExplainer(estimator, matrix)
    shap_values = explainer.shap_values(matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "10_shap_importance.csv", index=False)
    top = importance.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(top["feature"], top["mean_abs_shap"])
    axis.set(xlabel="Mean absolute SHAP value", ylabel="Feature", title=f"Global SHAP importance (n={len(sample)})")
    figure.tight_layout()
    figure.savefig(output_dir / "10_shap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Step 10 complete: saved SHAP outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    for path in (args.data, args.model, args.split):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calculate_shap(args.data, args.model, args.split, args.output_dir)


if __name__ == "__main__":
    main()
