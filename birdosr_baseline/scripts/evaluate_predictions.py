#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from birdosr.config import load_config
from birdosr.metrics import (
    apply_threshold,
    calibrate_tau,
    known_class_report,
    known_confusion_matrix,
    main_metrics,
    reconstruction_report,
    top_confusion_pairs,
    unknown_class_report,
    unknown_to_known_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.predictions)
    label_names = list(cfg["known_labels"])

    tau = args.tau
    if tau is None:
        tau = calibrate_tau(df["y_true"].to_numpy(), df["y_pred_known"].to_numpy(), df["score"].to_numpy()).tau
    df = df.copy()
    df["y_pred"] = apply_threshold(df["y_pred_known"].to_numpy(), df["score"].to_numpy(), tau)

    out = Path(args.output_dir or cfg["output_dir"])
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        main_metrics(df["y_true"].to_numpy(), df["y_pred_known"].to_numpy(), df["score"].to_numpy(), tau)
    ])
    metrics_df.to_csv(out / "metrics_from_predictions.csv", index=False)
    known_class_report(df, label_names).to_csv(out / "known_class_report.csv", index=False)
    unknown_class_report(df, label_names).to_csv(out / "unknown_class_report.csv", index=False)
    known_confusion_matrix(df, label_names).to_csv(out / "known_confusion_matrix.csv")
    unknown_to_known_matrix(df, label_names).to_csv(out / "unknown_to_known_matrix.csv")
    top_confusion_pairs(df, label_names).to_csv(out / "top_confusion_pairs.csv", index=False)
    reconstruction_report(df).to_csv(out / "reconstruction_report.csv", index=False)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

