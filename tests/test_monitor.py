"""Tests for src/monitor.py — drift detection logic."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.generate_data import generate_current_data, generate_reference_data
from src.monitor import (
    FEATURE_COLUMNS,
    extract_drift_score,
    load_datasets,
    run_drift_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def reference_df():
    return generate_reference_data(n_samples=200, seed=42)


@pytest.fixture()
def current_no_drift_df():
    return generate_current_data(n_samples=200, seed=43, drift=False)


@pytest.fixture()
def current_drifted_df():
    return generate_current_data(n_samples=200, seed=99, drift=True)


@pytest.fixture()
def csv_datasets(tmp_path, reference_df, current_no_drift_df):
    """Write reference and current CSVs to a temp directory."""
    ref_path = tmp_path / "reference.csv"
    cur_path = tmp_path / "current.csv"
    reference_df.to_csv(ref_path, index=False)
    current_no_drift_df.to_csv(cur_path, index=False)
    return str(ref_path), str(cur_path)


# ── load_datasets ─────────────────────────────────────────────────────────


class TestLoadDatasets:
    def test_returns_two_dataframes(self, csv_datasets):
        ref, cur = load_datasets(*csv_datasets)
        assert isinstance(ref, pd.DataFrame)
        assert isinstance(cur, pd.DataFrame)

    def test_selects_only_feature_columns(self, csv_datasets):
        ref, cur = load_datasets(*csv_datasets)
        assert list(ref.columns) == FEATURE_COLUMNS
        assert list(cur.columns) == FEATURE_COLUMNS

    def test_preserves_row_counts(self, csv_datasets):
        ref, cur = load_datasets(*csv_datasets)
        assert len(ref) == 200
        assert len(cur) == 200

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_datasets(str(tmp_path / "nope.csv"), str(tmp_path / "nope2.csv"))


# ── run_drift_report ──────────────────────────────────────────────────────


class TestRunDriftReport:
    def test_returns_dict(self, reference_df, current_no_drift_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_no_drift_df[FEATURE_COLUMNS]
        result = run_drift_report(ref, cur)
        assert isinstance(result, dict)
        assert "metrics" in result

    def test_saves_json_report(self, reference_df, current_no_drift_df, monkeypatch):
        import src.monitor as monitor_mod

        ref = reference_df[FEATURE_COLUMNS]
        cur = current_no_drift_df[FEATURE_COLUMNS]

        # Redirect report to a temp location
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "report.json"
        monkeypatch.setattr(monitor_mod, "REPORT_PATH", tmp)

        run_drift_report(ref, cur)
        assert tmp.exists()

        with open(tmp) as f:
            data = json.load(f)
        assert "metrics" in data

    def test_report_contains_dataset_drift_metric(self, reference_df, current_no_drift_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_no_drift_df[FEATURE_COLUMNS]
        result = run_drift_report(ref, cur)

        metric_names = [m.get("metric", "") for m in result["metrics"]]
        assert "DatasetDriftMetric" in metric_names


# ── extract_drift_score ───────────────────────────────────────────────────


class TestExtractDriftScore:
    def test_extracts_score_from_valid_report(self):
        report = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {"drift_share": 0.67},
                }
            ]
        }
        assert extract_drift_score(report) == pytest.approx(0.67)

    def test_returns_zero_when_metric_missing(self):
        report = {"metrics": [{"metric": "SomethingElse", "result": {}}]}
        assert extract_drift_score(report) == 0.0

    def test_returns_zero_for_empty_metrics(self):
        assert extract_drift_score({"metrics": []}) == 0.0

    def test_score_is_float(self):
        report = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {"drift_share": 1},
                }
            ]
        }
        result = extract_drift_score(report)
        assert isinstance(result, float)


# ── Drift detection behavior ─────────────────────────────────────────────


class TestDriftDetection:
    def test_no_drift_on_similar_data(self, reference_df, current_no_drift_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_no_drift_df[FEATURE_COLUMNS]
        result = run_drift_report(ref, cur)
        score = extract_drift_score(result)
        assert score <= 0.5, f"Expected low drift on similar data, got {score}"

    def test_high_drift_on_shifted_data(self, reference_df, current_drifted_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_drifted_df[FEATURE_COLUMNS]
        result = run_drift_report(ref, cur)
        score = extract_drift_score(result)
        assert score > 0.3, f"Expected high drift on shifted data, got {score}"

    def test_drift_score_in_valid_range(self, reference_df, current_drifted_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_drifted_df[FEATURE_COLUMNS]
        result = run_drift_report(ref, cur)
        score = extract_drift_score(result)
        assert 0.0 <= score <= 1.0
