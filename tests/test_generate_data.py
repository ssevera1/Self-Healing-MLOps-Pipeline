"""Tests for src/generate_data.py — synthetic dataset validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.generate_data import validate_dataset, generate_reference_data


class TestValidateDataset:
    def test_accepts_a_well_formed_reference_dataset(self):
        validate_dataset(generate_reference_data(n_samples=10), "reference")

    def test_rejects_missing_columns(self):
        df = generate_reference_data(n_samples=10).drop(columns=["is_fraud"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_dataset(df, "reference")

    def test_rejects_null_values(self):
        df = generate_reference_data(n_samples=10)
        df.loc[0, "user_transaction_amount_avg"] = None
        with pytest.raises(ValueError, match="null values"):
            validate_dataset(df, "reference")

    def test_rejects_empty_dataset(self):
        """An all-column-correct but rowless frame must not reach disk."""
        df = generate_reference_data(n_samples=0)
        with pytest.raises(ValueError, match="no rows"):
            validate_dataset(df, "reference")

    def test_rejects_unknown_dataset_type(self):
        with pytest.raises(ValueError, match="Unknown dataset type"):
            validate_dataset(generate_reference_data(n_samples=5), "nope")
