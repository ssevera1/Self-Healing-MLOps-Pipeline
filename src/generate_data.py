"""Generate synthetic reference and current datasets for drift detection."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "reference": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max", "is_fraud"},
    "current": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max", "is_fraud"},
    "feast": {"user_id", "event_timestamp", "user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max"},
}

LABEL_COLUMN = "is_fraud"

MIN_FRAUD_FRACTION = 0.01
MIN_LEGITIMATE_FRACTION = 0.01


def validate_dataset(df: pd.DataFrame, dataset_type: str) -> None:
    """Validate that dataset contains expected columns and has no null values.
    
    Args:
        df: DataFrame to validate.
        dataset_type: One of 'reference', 'current', 'feast'.
        
    Raises:
        ValueError: If validation fails.
    """
    expected = EXPECTED_COLUMNS.get(dataset_type)
    if expected is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(f"{dataset_type} dataset missing columns: {missing}")
    
    extra = actual - expected
    if extra:
        raise ValueError(f"{dataset_type} dataset has unexpected columns: {extra}")
    
    null_counts = df[list(expected)].isnull().sum()
    if null_counts.any():
        null_info = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"{dataset_type} dataset contains null values: {null_info}")

    if len(df) == 0:
        raise ValueError(f"{dataset_type} dataset has no rows")


def validate_fraud_distribution(df: pd.DataFrame, dataset_type: str) -> None:
    """Validate that fraud class distribution meets minimum thresholds.

    Dataset types with no label column (e.g. 'feast') are skipped; an
    unrecognised type is an error, matching validate_dataset.

    Args:
        df: DataFrame to validate.
        dataset_type: One of 'reference', 'current', 'feast'.

    Raises:
        ValueError: If the dataset type is unknown, or if the fraud or
            legitimate class fraction falls below threshold.
    """
    expected = EXPECTED_COLUMNS.get(dataset_type)
    if expected is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if LABEL_COLUMN not in expected:
        return

    fraud_count = (df[LABEL_COLUMN] == 1).sum()
    total_count = len(df)

    fraud_fraction = fraud_count / total_count if total_count > 0 else 0
    legitimate_fraction = 1 - fraud_fraction

    if fraud_fraction < MIN_FRAUD_FRACTION:
        raise ValueError(
            f"{dataset_type} dataset: fraud class fraction {fraud_fraction:.4f} "
            f"below minimum threshold {MIN_FRAUD_FRACTION}"
        )

    if legitimate_fraction < MIN_LEGITIMATE_FRACTION:
        raise ValueError(
            f"{dataset_type} dataset: legitimate class fraction {legitimate_fraction:.4f} "
            f"below minimum threshold {MIN_LEGITIMATE_FRACTION}"
        )


def generate_reference_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a stable reference (historical) dataset."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_transaction_count": rng.poisson(lam=5, size=n_samples),
        "user_transaction_amount_avg": rng.normal(loc=50.0, scale=10.0, size=n_samples),
        "user_transaction_amount_max": rng.normal(loc=200.0, scale=30.0, size=n_samples),
        "is_fraud": rng.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
    })


def generate_current_data(
    n_samples: int = 500,
    seed: int = 99,
    drift: bool = False,
) -> pd.DataFrame:
    """Generate a current (new logs) dataset, optionally with drift."""
    rng = np.random.default_rng(seed)

    if drift:
        # Shifted distributions to simulate data drift
        tx_count = rng.poisson(lam=15, size=n_samples)
        tx_avg = rng.normal(loc=120.0, scale=25.0, size=n_samples)
        tx_max = rng.normal(loc=500.0, scale=80.0, size=n_samples)
        fraud_prob = [0.80, 0.20]
    else:
        tx_count = rng.poisson(lam=5, size=n_samples)
        tx_avg = rng.normal(loc=50.0, scale=10.0, size=n_samples)
        tx_max = rng.normal(loc=200.0, scale=30.0, size=n_samples)
        fraud_prob = [0.95, 0.05]

    return pd.DataFrame({
        "user_transaction_count": tx_count,
        "user_transaction_amount_avg": tx_avg,
        "user_transaction_amount_max": tx_max,
        "is_fraud": rng.choice([0, 1], size=n_samples, p=fraud_prob),
    })


def generate_feast_data(n_users: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate the Feast feature store parquet (one row per user)."""
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.utcnow()
    return pd.DataFrame({
        "user_id": range(n_users),
        "event_timestamp": [now] * n_users,
        "user_transaction_count": rng.poisson(lam=5, size=n_users).astype("int64"),
        "user_transaction_amount_avg": rng.normal(loc=50.0, scale=10.0, size=n_users).astype("float32"),
        "user_transaction_amount_max": rng.normal(loc=200.0, scale=30.0, size=n_users).astype("float32"),
    })


if __name__ == "__main__":
    Path("data").mkdir(parents=True, exist_ok=True)

    ref = generate_reference_data()
    validate_dataset(ref, "reference")
    validate_fraud_distribution(ref, "reference")
    ref.to_csv("data/reference.csv", index=False)

    # SIMULATE_DRIFT=false → generate on-distribution current data (healthy run).
    # Defaults to true so the pipeline demonstrates self-healing out of the box.
    simulate_drift = os.getenv("SIMULATE_DRIFT", "true").lower() != "false"
    cur = generate_current_data(drift=simulate_drift)
    validate_dataset(cur, "current")
    validate_fraud_distribution(cur, "current")
    cur.to_csv("data/current.csv", index=False)

    feast = generate_feast_data()
    validate_dataset(feast, "feast")
    feast.to_parquet("data/user_transactions.parquet", index=False)

    print(f"Reference data: {ref.shape}")
    print(f"Current data (with drift): {cur.shape}")
    print(f"Feast feature data: {feast.shape}")
