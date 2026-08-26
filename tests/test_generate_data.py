"""Tests for src/generate_data.py — synthetic dataset validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.generate_data import (
    generate_feast_data,
    generate_reference_data,
    validate_dataset,
    validate_fraud_distribution,
)


def _labelled_frame(labels: list[int]) -> pd.DataFrame:
    """Build a reference-shaped frame with an exact is_fraud column.

    Fabricated rather than generated: at p=0.05 a small generated frame can
    hold zero fraud rows, so the class fractions under test must be explicit.
    """
    n = len(labels)
    return pd.DataFrame({
        "user_transaction_count": [5] * n,
        "user_transaction_amount_avg": [50.0] * n,
        "user_transaction_amount_max": [200.0] * n,
        "is_fraud": labels,
    })


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


class TestValidateFraudDistribution:
    def test_accepts_a_well_formed_reference_dataset(self):
        validate_fraud_distribution(_labelled_frame([0] * 95 + [1] * 5), "reference")

    def test_accepts_a_well_formed_current_dataset(self):
        validate_fraud_distribution(_labelled_frame([0] * 80 + [1] * 20), "current")

    def test_rejects_all_legitimate_dataset(self):
        """No fraud rows means the retrained model can never learn the class."""
        with pytest.raises(ValueError, match="fraud class fraction"):
            validate_fraud_distribution(_labelled_frame([0] * 100), "reference")

    def test_rejects_all_fraud_dataset(self):
        with pytest.raises(ValueError, match="legitimate class fraction"):
            validate_fraud_distribution(_labelled_frame([1] * 100), "current")

    def test_rejects_fraud_fraction_just_below_threshold(self):
        """1 in 200 is 0.5%, under the 1% floor; 2 in 200 is exactly at it."""
        with pytest.raises(ValueError, match="fraud class fraction"):
            validate_fraud_distribution(_labelled_frame([0] * 199 + [1]), "reference")
        validate_fraud_distribution(_labelled_frame([0] * 198 + [1, 1]), "reference")

    def test_rejects_empty_dataset(self):
        with pytest.raises(ValueError, match="fraud class fraction"):
            validate_fraud_distribution(_labelled_frame([]), "reference")

    def test_skips_dataset_types_without_a_label_column(self):
        validate_fraud_distribution(generate_feast_data(n_users=10), "feast")

    def test_rejects_unknown_dataset_type(self):
        """A typo must not silently skip the gate, as validate_dataset also refuses."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            validate_fraud_distribution(_labelled_frame([0] * 95 + [1] * 5), "referance")
