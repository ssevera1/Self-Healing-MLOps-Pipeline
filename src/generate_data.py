"""Generate synthetic reference and current datasets for drift detection."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "reference": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max", "is_fraud"},
    "current": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max", "is_fraud"},
    "feast": {"user_id", "event_timestamp", "user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max"},
}

NON_NEGATIVE_COLUMNS = {
    "reference": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max"},
    "current": {"user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max"},
    "feast": {"user_id", "user_transaction_count", "user_transaction_amount_avg", "user_transaction_amount_max"},
}

# Datasets that carry a binary fraud label, which should always contain both classes.
FRAUD_LABEL_DATASETS = {"reference", "current"}


def validate_dataset(df: pd.DataFrame, dataset_type: str) -> None:
    """Validate that dataset is well-formed before it's persisted.

    Checks: expected columns present (no missing/extra), no null values,
    at least one row, no negative values in count/amount columns, and — for
    labelled datasets — both fraud classes (0 and 1) are represented.

    Args:
        df: DataFrame to validate.
        dataset_type: One of 'reference', 'current', 'feast'.

    Raises:
        ValueError: If validation fails.
    """
    expected = EXPECTED_COLUMNS.get(dataset_type)
    if expected is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if df.shape[0] == 0:
        raise ValueError(f"{dataset_type} dataset has zero rows")

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

    for col in NON_NEGATIVE_COLUMNS.get(dataset_type, set()):
        if (df[col] < 0).any():
            raise ValueError(f"{dataset_type} dataset: column '{col}' contains negative values")

    if dataset_type in FRAUD_LABEL_DATASETS:
        fraud_classes = set(df["is_fraud"].unique())
        if fraud_classes != {0, 1}:
            raise ValueError(
                f"{dataset_type} dataset does not contain both fraud classes (0 and 1): got {fraud_classes}"
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


def _write_file(df: pd.DataFrame, path: str, format: str = "csv") -> None:
    """Write a DataFrame to disk, failing loudly instead of silently on error."""
    try:
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except OSError as e:
        print(f"Error: Failed to write {path} - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        Path("data").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Failed to create data/ directory - {e}", file=sys.stderr)
        sys.exit(1)

    ref = generate_reference_data()
    validate_dataset(ref, "reference")
    _write_file(ref, "data/reference.csv", format="csv")

    # SIMULATE_DRIFT=false → generate on-distribution current data (healthy run).
    # Defaults to true so the pipeline demonstrates self-healing out of the box.
    simulate_drift = os.getenv("SIMULATE_DRIFT", "true").lower() != "false"
    cur = generate_current_data(drift=simulate_drift)
    validate_dataset(cur, "current")
    _write_file(cur, "data/current.csv", format="csv")

    feast = generate_feast_data()
    validate_dataset(feast, "feast")
    _write_file(feast, "data/user_transactions.parquet", format="parquet")

    print(f"Reference data: {ref.shape}")
    print(f"Current data (with drift): {cur.shape}")
    print(f"Feast feature data: {feast.shape}")
