"""Tests for src/retrain_trigger.py — retraining and self-healing logic."""

from pathlib import Path
from unittest.mock import patch

import joblib

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.generate_data import generate_reference_data
from src.retrain_trigger import (
    DRIFT_THRESHOLD,
    heal,
    save_model,
    train_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def training_csv(tmp_path):
    """Write a reference CSV for training."""
    df = generate_reference_data(n_samples=200, seed=42)
    path = tmp_path / "reference.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def trained_model(training_csv):
    return train_model(data_path=training_csv)


# ── train_model ───────────────────────────────────────────────────────────


class TestTrainModel:
    def test_returns_random_forest(self, training_csv):
        model = train_model(data_path=training_csv)
        assert isinstance(model, RandomForestClassifier)

    def test_model_has_100_estimators(self, training_csv):
        model = train_model(data_path=training_csv)
        assert model.n_estimators == 100

    def test_model_is_fitted(self, training_csv):
        model = train_model(data_path=training_csv)
        # A fitted RandomForest has the classes_ attribute
        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2  # binary: 0, 1

    def test_model_can_predict(self, training_csv):
        model = train_model(data_path=training_csv)
        sample = pd.DataFrame({
            "user_transaction_count": [5],
            "user_transaction_amount_avg": [50.0],
            "user_transaction_amount_max": [200.0],
        })
        preds = model.predict(sample)
        assert preds[0] in (0, 1)

    def test_training_accuracy_above_chance(self, training_csv):
        model = train_model(data_path=training_csv)
        df = pd.read_csv(training_csv)
        feature_cols = [
            "user_transaction_count",
            "user_transaction_amount_avg",
            "user_transaction_amount_max",
        ]
        accuracy = model.score(df[feature_cols], df["is_fraud"])
        assert accuracy > 0.5, "Model should beat random chance"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            train_model(data_path="/nonexistent/path.csv")


# ── save_model ────────────────────────────────────────────────────────────


class TestSaveModel:
    def test_creates_pkl_file(self, trained_model, monkeypatch):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        path = save_model(trained_model)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_filename_contains_timestamp(self, trained_model, monkeypatch):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        path = save_model(trained_model)
        assert path.name.startswith("fraud_model_")
        assert path.name.endswith("Z.pkl")

    def test_saved_model_is_loadable(self, trained_model, monkeypatch):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        path = save_model(trained_model)
        # Safe: file was written by save_model() two lines above — trusted source.
        loaded = joblib.load(path)
        assert isinstance(loaded, RandomForestClassifier)
        assert hasattr(loaded, "classes_")

    def test_creates_models_directory(self, trained_model, monkeypatch):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "new_models_dir"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        assert not tmp.exists()
        save_model(trained_model)
        assert tmp.exists()


# ── heal (integration) ────────────────────────────────────────────────────


class TestHeal:
    def test_triggers_retraining_on_high_drift(self, training_csv, monkeypatch, capsys):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        with patch("src.retrain_trigger.run_monitor", return_value=0.8):
            with patch("src.retrain_trigger.train_model") as mock_train:
                mock_train.return_value = RandomForestClassifier().fit(
                    [[1, 2, 3]], [0]
                )
                heal()
                mock_train.assert_called_once()

        output = capsys.readouterr().out
        assert "Drift Detected! Triggering retraining..." in output

    def test_skips_retraining_on_low_drift(self, capsys):
        with patch("src.retrain_trigger.run_monitor", return_value=0.1):
            heal()

        output = capsys.readouterr().out
        assert "No significant drift detected. Model is healthy." in output

    def test_triggers_retraining_at_exact_threshold(self, monkeypatch, capsys):
        """Drift score == threshold is treated as drift (operator is >=)."""
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        with patch("src.retrain_trigger.run_monitor", return_value=DRIFT_THRESHOLD):
            with patch("src.retrain_trigger.train_model") as mock_train:
                mock_train.return_value = RandomForestClassifier().fit(
                    [[1, 2, 3]], [0]
                )
                heal()
                mock_train.assert_called_once()

        output = capsys.readouterr().out
        assert "Drift Detected!" in output

    def test_retrains_just_above_threshold(self, monkeypatch, capsys):
        import src.retrain_trigger as trigger_mod
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / "models"
        monkeypatch.setattr(trigger_mod, "MODELS_DIR", tmp)

        with patch("src.retrain_trigger.run_monitor", return_value=DRIFT_THRESHOLD + 0.01):
            with patch("src.retrain_trigger.train_model") as mock_train:
                mock_train.return_value = RandomForestClassifier().fit(
                    [[1, 2, 3]], [0]
                )
                heal()
                mock_train.assert_called_once()

        output = capsys.readouterr().out
        assert "Drift Detected!" in output


# ── Configuration ─────────────────────────────────────────────────────────


class TestConfig:
    def test_drift_threshold_is_0_3(self):
        assert DRIFT_THRESHOLD == 0.3

    def test_drift_threshold_is_float(self):
        assert isinstance(DRIFT_THRESHOLD, float)
