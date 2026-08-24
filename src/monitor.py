"""Drift monitoring using Evidently AI.

Compares a reference (historical) dataset against a current (new logs)
dataset and reports per-column data drift scores.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


logger = logging.getLogger(__name__)

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
    try:
        reference = pd.read_csv(reference_path)
    except FileNotFoundError as exc:
        logger.error(f"Reference dataset not found at {reference_path}")
        raise
    except pd.errors.ParserError as exc:
        logger.error(f"Failed to parse reference dataset at {reference_path}: {exc}")
        raise

    try:
        current = pd.read_csv(current_path)
    except FileNotFoundError as exc:
        logger.error(f"Current dataset not found at {current_path}")
        raise
    except pd.errors.ParserError as exc:
        logger.error(f"Failed to parse current dataset at {current_path}: {exc}")
        raise

    for label, df in (("reference", reference), ("current", current)):
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{label} dataset is missing expected columns: {missing}")
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
    _write_report_atomically(result, REPORT_PATH)

    return result


def _write_report_atomically(result: dict, path: Path) -> None:
    """Serialize to a sibling temp file, then os.replace() it into place.

    Writing in place would truncate the previous good report before
    json.dump streams the new one, so a mid-write failure would leave
    invalid JSON on disk - which load_drift_report would then misreport as
    "malformed JSON" rather than a write failure, and which
    .github/workflows/mlops.yml uploads as an artifact via `if: always()`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, path)
    except (TypeError, ValueError) as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error("Failed to serialize drift report to JSON: %s", exc)
        raise RuntimeError(f"Drift report serialization failed: {exc}") from exc
    except (IOError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error("Failed to write drift report to %s: %s", path, exc)
        raise RuntimeError(f"Unable to persist drift report: {exc}") from exc
    logger.info("Drift report persisted to %s", path)


def load_drift_report(path: Path | None = None) -> dict:
    """Load persisted drift report from JSON file.

    REPORT_PATH is resolved at call time, not bound as a default, so this
    stays in step with run_drift_report when the module global is
    reassigned (which tests/test_monitor.py does).

    Raises RuntimeError if the file is missing or malformed.
    """
    path = path if path is not None else REPORT_PATH

    if not path.exists():
        logger.error("Drift report file not found: %s", path)
        raise RuntimeError(f"Drift report missing at {path}")

    try:
        with open(path, "r") as f:
            report_dict = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Drift report is malformed JSON at %s: %s", path, exc)
        raise RuntimeError(f"Drift report is malformed JSON: {exc}") from exc
    except (IOError, OSError) as exc:
        logger.error("Failed to read drift report at %s: %s", path, exc)
        raise RuntimeError(f"Unable to read drift report: {exc}") from exc

    logger.debug("Loaded drift report from %s", path)
    return report_dict


def extract_drift_score(report_dict: dict) -> float:
    """Extract the dataset-level drift share from the Evidently report.

    The drift share is the fraction of columns that are detected as drifted
    (value between 0.0 and 1.0).
    """
    for metric in report_dict["metrics"]:
        metric_id = metric.get("metric", "")
        if metric_id == "DatasetDriftMetric":
            try:
                return float(metric["result"]["drift_share"])
            except KeyError as exc:
                raise RuntimeError(
                    f"Unexpected Evidently report schema — missing key: {exc}"
                ) from exc
    # Metric not present — treat as no drift detected (fail-safe: don't retrain
    # on ambiguous report data).
    return 0.0


def main() -> float:
    """Run monitoring pipeline and return the drift score."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
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
