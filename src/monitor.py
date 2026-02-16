"""Drift monitoring using Evidently AI.

Compares a reference (historical) dataset against a current (new logs)
dataset and reports per-column data drift scores.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


FEATURE_COLUMNS = [
    "user_transaction_count",
    "user_transaction_amount_avg",
    "user_transaction_amount_max",
]

REPORT_PATH = Path("data/drift_report.json")


def load_datasets(
    reference_path: str = "data/reference.csv",
    current_path: str = "data/current.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current datasets from CSV files."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    return reference[FEATURE_COLUMNS], current[FEATURE_COLUMNS]


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """Run an Evidently DataDrift report and return the result dict."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()

    # Persist the full JSON report for downstream consumers
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


def extract_drift_score(report_dict: dict) -> float:
    """Extract the dataset-level drift share from the Evidently report.

    The drift share is the fraction of columns that are detected as drifted
    (value between 0.0 and 1.0).
    """
    for metric in report_dict["metrics"]:
        metric_id = metric.get("metric", "")
        if metric_id == "DatasetDriftMetric":
            return float(metric["result"]["drift_share"])
    return 0.0


def main() -> float:
    """Run monitoring pipeline and return the drift score."""
    print("Loading datasets...")
    reference, current = load_datasets()

    print(f"Reference shape: {reference.shape}")
    print(f"Current shape:   {current.shape}")

    print("Running Evidently DataDrift report...")
    report_dict = run_drift_report(reference, current)

    drift_score = extract_drift_score(report_dict)
    print(f"Drift score (share of drifted columns): {drift_score:.4f}")
    print(f"Full report saved to {REPORT_PATH}")

    return drift_score


if __name__ == "__main__":
    score = main()
    sys.exit(0)
