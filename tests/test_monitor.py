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
    load_drift_report,
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

    def test_raises_on_missing_column_in_current(self, reference_df, current_no_drift_df):
        ref = reference_df[FEATURE_COLUMNS]
        cur = current_no_drift_df[FEATURE_COLUMNS].drop(columns=[FEATURE_COLUMNS[0]])
        with pytest.raises(ValueError, match="current dataset is missing expected columns"):
            run_drift_report(ref, cur)

    def test_raises_on_missing_column_in_reference(self, reference_df, current_no_drift_df):
        ref = reference_df[FEATURE_COLUMNS].drop(columns=[FEATURE_COLUMNS[0]])
        cur = current_no_drift_df[FEATURE_COLUMNS]
        with pytest.raises(ValueError, match="reference dataset is missing expected columns"):
            run_drift_report(ref, cur)


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


class TestLoadDriftReport:
    """Cover the persistence round-trip and both failure branches."""

    def test_round_trip_follows_reassigned_report_path(
        self, reference_df, current_no_drift_df, tmp_path, monkeypatch
    ):
        """load_drift_report must resolve REPORT_PATH at call time, like run_drift_report."""
        import src.monitor as monitor_mod

        tmp = tmp_path / "report.json"
        monkeypatch.setattr(monitor_mod, "REPORT_PATH", tmp)

        written = run_drift_report(
            reference_df[FEATURE_COLUMNS], current_no_drift_df[FEATURE_COLUMNS]
        )
        loaded = load_drift_report()

        assert tmp.exists()
        assert "metrics" in loaded
        assert loaded == json.loads(json.dumps(written))

    def test_missing_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Drift report missing"):
            load_drift_report(tmp_path / "absent.json")

    def test_malformed_json_raises_runtime_error(self, tmp_path):
        bad = tmp_path / "report.json"
        bad.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(RuntimeError, match="malformed JSON"):
            load_drift_report(bad)

    def test_explicit_path_argument_wins_over_global(self, tmp_path, monkeypatch):
        import src.monitor as monitor_mod

        monkeypatch.setattr(monitor_mod, "REPORT_PATH", tmp_path / "global.json")
        explicit = tmp_path / "explicit.json"
        explicit.write_text('{"metrics": []}', encoding="utf-8")

        assert load_drift_report(explicit) == {"metrics": []}


class TestReportWriteIsAtomic:
    """A failed write must not destroy the previous good report."""

    def test_serialization_failure_leaves_existing_report_intact(
        self, tmp_path, monkeypatch
    ):
        import src.monitor as monitor_mod

        target = tmp_path / "report.json"
        target.write_text('{"metrics": ["previous good report"]}', encoding="utf-8")
        monkeypatch.setattr(monitor_mod, "REPORT_PATH", target)

        # A set is not JSON-serializable; json.dump raises partway through.
        with pytest.raises(RuntimeError, match="serialization failed"):
            monitor_mod._write_report_atomically({"bad": {1, 2, 3}}, target)

        assert json.loads(target.read_text(encoding="utf-8")) == {
            "metrics": ["previous good report"]
        }

    def test_failed_write_leaves_no_temp_files_behind(self, tmp_path):
        import src.monitor as monitor_mod

        target = tmp_path / "report.json"
        with pytest.raises(RuntimeError):
            monitor_mod._write_report_atomically({"bad": {1, 2, 3}}, target)

        assert list(tmp_path.iterdir()) == []
