"""README Step 9: evaluate discrimination, calibration, KS, and top-k capture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve


RESULTS = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RESULTS / "08_test_predictions.csv"


def ks_statistic(target, score) -> float:
    data = pd.DataFrame({"target": np.asarray(target), "score": np.asarray(score)}).sort_values("score", ascending=False)
    events = data["target"].sum()
    non_events = len(data) - events
    if events == 0 or non_events == 0:
        return 0.0
    return float(((data["target"].cumsum() / events) - ((1 - data["target"]).cumsum() / non_events)).abs().max())


def top_k_capture(target, score, fraction: float) -> float:
    data = pd.DataFrame({"target": np.asarray(target), "score": np.asarray(score)}).sort_values("score", ascending=False)
    total_events = data["target"].sum()
    return float(data.head(max(1, int(len(data) * fraction)))["target"].sum() / total_events) if total_events else np.nan


def metric_row(target, score, name: str) -> dict:
    return {
        "model": name,
        "n": len(target),
        "event_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, score)),
        "pr_auc": float(average_precision_score(target, score)),
        "brier": float(brier_score_loss(target, score)),
        "precision_0_5": float(precision_score(target, score >= 0.5, zero_division=0)),
        "recall_0_5": float(recall_score(target, score >= 0.5, zero_division=0)),
        "f1_0_5": float(f1_score(target, score >= 0.5, zero_division=0)),
        "ks": ks_statistic(target, score),
        "top_1pct_capture": top_k_capture(target, score, 0.01),
        "top_5pct_capture": top_k_capture(target, score, 0.05),
        "top_10pct_capture": top_k_capture(target, score, 0.10),
    }


def evaluate(prediction_path: Path, output_dir: Path) -> None:
    predictions = pd.read_csv(prediction_path)
    target = predictions["target_3m"].astype(int).to_numpy()
    raw = predictions["pd_raw"].to_numpy()
    calibrated = predictions["pd_calibrated"].to_numpy()
    metrics = pd.DataFrame([metric_row(target, raw, "raw"), metric_row(target, calibrated, "calibrated")])
    metrics.to_csv(output_dir / "09_evaluation_metrics.csv", index=False)

    fpr_raw, tpr_raw, _ = roc_curve(target, raw)
    fpr_cal, tpr_cal, _ = roc_curve(target, calibrated)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr_raw, tpr_raw, label=f"Raw AUC={roc_auc_score(target, raw):.3f}")
    axis.plot(fpr_cal, tpr_cal, label=f"Calibrated AUC={roc_auc_score(target, calibrated):.3f}")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-time ROC curve")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "09_roc.png", dpi=180); plt.close(figure)

    x_raw, y_raw = calibration_curve(target, raw, n_bins=10, strategy="quantile")
    x_cal, y_cal = calibration_curve(target, calibrated, n_bins=10, strategy="quantile")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(x_raw, y_raw, "o--", label=f"Raw Brier={brier_score_loss(target, raw):.5f}")
    axis.plot(x_cal, y_cal, "s-", label=f"Calibrated Brier={brier_score_loss(target, calibrated):.5f}")
    axis.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed event rate", title="Probability calibration")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "09_calibration.png", dpi=180); plt.close(figure)

    precision, recall, _ = precision_recall_curve(target, calibrated)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(recall, precision, label=f"PR-AUC={average_precision_score(target, calibrated):.4f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "09_pr_curve.png", dpi=180); plt.close(figure)
    print(f"Step 9 complete: saved evaluation metrics and charts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Run Step 8 first or provide --input: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluate(args.input, args.output_dir)


if __name__ == "__main__":
    main()
