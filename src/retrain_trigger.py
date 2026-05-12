"""Self-healing retraining trigger.

Reads the drift score produced by the monitor and, when the score exceeds
the configured threshold, retrains the fraud-detection model and saves a
new versioned artifact.
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.monitor import main as run_monitor

DRIFT_THRESHOLD = 0.3
MODELS_DIR = Path("models")


def train_model(data_path: str = "data/reference.csv") -> RandomForestClassifier:
    """Train a dummy fraud-detection model on the reference data."""
    df = pd.read_csv(data_path)

    feature_cols = [
        "user_transaction_count",
        "user_transaction_amount_avg",
        "user_transaction_amount_max",
    ]
    X = df[feature_cols]
    y = df["is_fraud"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"Model trained — in-sample accuracy: {accuracy:.4f}")
    return model


def save_model(model: RandomForestClassifier) -> Path:
    """Persist the model with a UTC-timestamped filename."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"fraud_model_{timestamp}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")
    return path


def heal() -> None:
    """Run the full self-healing loop: monitor → decide → retrain."""
    drift_score = run_monitor()

    print(f"\nDrift threshold: {DRIFT_THRESHOLD}")
    print(f"Current drift:   {drift_score:.4f}")

    if drift_score >= DRIFT_THRESHOLD:
        print("Drift Detected! Triggering retraining...")
        model = train_model()
        save_model(model)
        print("Self-healing retraining complete.")
    else:
        print("No significant drift detected. Model is healthy.")


if __name__ == "__main__":
    heal()
