"""Generate synthetic reference and current datasets for drift detection."""

from pathlib import Path

import numpy as np
import pandas as pd


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
    ref.to_csv("data/reference.csv", index=False)

    cur = generate_current_data(drift=True)
    cur.to_csv("data/current.csv", index=False)

    feast = generate_feast_data()
    feast.to_parquet("data/user_transactions.parquet", index=False)

    print(f"Reference data: {ref.shape}")
    print(f"Current data (with drift): {cur.shape}")
    print(f"Feast feature data: {feast.shape}")
