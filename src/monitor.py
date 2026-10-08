"""Drift monitoring using Evidently AI.

Compares a reference (historical) dataset against a current (new logs)
dataset and reports per-column data drift scores.
"""

import json
import logging
import os
import sys
import tempfile
import time
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

_LOAD_DATASETS_MAX_RETRIES = 3
_LOAD_DATASETS_RETRY_DELAY_SECS = 1.0
_LOAD_DATASETS_READ_TIMEOUT_SECS = 30.0


def _validate_feature_columns(label: str, df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing any of FEATURE_COLUMNS."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} dataset is missing expected columns: {missing}")


def _read_csv_with_timeout(
    path: str,
    timeout_secs: float = _LOAD_DATASETS_READ_TIMEOUT_SECS,
) -> pd.DataFrame:
    """Read CSV file with timeout enforcement.
    
    Raises TimeoutError if read exceeds timeout_secs.
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"CSV read exceeded {timeout_secs}s timeout")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_secs) + 1)  # Add 1s margin
    try:
        df = pd.read_csv(path)
        signal.alarm(0)
        return df
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def load_datasets(
    reference_path: str = "data/reference.csv",
    current_path: str = "data/current.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current datasets from CSV files with retry logic.
    
    Retries transient read failures up to _LOAD_DATASETS_MAX_RETRIES times
    with exponential backoff.
    """
    def _load_single_dataset(path: str, label: str) -> pd.DataFrame:
        """Load a single CSV with retries on transient failures."""
        last_exc = None
        
        for attempt in range(_LOAD_DATASETS_MAX_RETRIES):
            try:
                return _read_csv_with_timeout(path)
            except FileNotFoundError as exc:
                logger.error(f"{label} dataset not found at {path}")
                raise
            except (pd.errors.ParserError, TimeoutError) as exc:
                last_exc = exc
                if attempt < _LOAD_DATASETS_MAX_RETRIES - 1:
                    wait_time = _LOAD_DATASETS_RETRY_DELAY_SECS * (2 ** attempt)
                    logger.warning(
                        f"Failed to load {label} dataset from {path} (attempt {attempt + 1}/{_LOAD_DATASETS_MAX_RETRIES}): {exc}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to load {label} dataset from {path} after {_LOAD_DATASETS_MAX_RETRIES} attempts: {exc}"
                    )
        
        raise last_exc
    
    reference = _load_single_dataset(reference_path, "reference")
    current = _load_single_dataset(current_path, "current")

    for label, df in (("reference", reference), ("current", current)):
        _validate_feature_columns(label, df)
    return reference[FEATURE_COLUMNS], current[FEATURE_COLUMNS]


def _validate_drift_report_structure(report_dict: dict) -> list:
    """Validate drift report structure and return its metrics list.

    Returning the validated list lets callers use it with a concrete type
    instead of re-reading it from the dict as ``Any | None``.

    Raises RuntimeError if report structure is invalid.
    """
    if not isinstance(report_dict, dict):
        logger.error("Drift report is not a dict: %s", type(report_dict))
        raise RuntimeError("Drift report must be a dict")

    metrics = report_dict.get("metrics")
    if not isinstance(metrics, list):
        logger.error(
            "Drift report 'metrics' field is malformed: expected list, got %s",
            type(metrics),
        )
        raise RuntimeError("Drift report 'metrics' field must be a list")

    if not metrics:
        logger.warning("Drift report has empty metrics list")

    return metrics


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """Run an Evidently DataDrift report and return the result dict."""
    # Validate that feature columns exist, in case a caller bypasses load_datasets.
    for label, df in (("reference", reference), ("current", current)):
        _validate_feature_columns(label, df)

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
    metrics = _validate_drift_report_structure(report_dict)
    logger.debug("Extracting drift score from report with %d metrics", len(metrics))
    for metric in metrics:
        if not isinstance(metric, dict):
            logger.warning("Skipping non-dict metric entry: %s", type(metric))
            continue

        metric_id = metric.get("metric", "")
        if metric_id == "DatasetDriftMetric":
            logger.debug("Found DatasetDriftMetric, extracting drift_share")
            result = metric.get("result")
            if not isinstance(result, dict):
                logger.error(
                    "DatasetDriftMetric 'result' field is malformed: expected dict, got %s",
                    type(result),
                )
                raise RuntimeError(
                    "Unexpected Evidently report schema — 'result' must be a dict"
                )

            if "drift_share" not in result:
                logger.error(
                    "DatasetDriftMetric 'result' is missing 'drift_share' key"
                )
                raise RuntimeError(
                    "Unexpected Evidently report schema — missing 'drift_share' key"
                )

            try:
                drift_score = float(result["drift_share"])
                logger.info("Drift score extracted: %.4f", drift_score)
                return drift_score
            except (ValueError, TypeError) as exc:
                logger.error("Failed to convert drift_share to float: %s", exc)
                raise RuntimeError(
                    f"Unexpected Evidently report schema — drift_share is malformed: {exc}"
                ) from exc
    logger.warning("DatasetDriftMetric not present in report; defaulting drift score to 0.0")
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
