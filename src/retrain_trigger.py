"""Self-healing retraining trigger.

Reads the drift score produced by the monitor and, when the score exceeds
the configured threshold, retrains the fraud-detection model and saves a
new versioned artifact.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.monitor import main as run_monitor

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
MODELS_DIR = Path("models")


def train_model(data_path: str = "data/reference.csv") -> RandomForestClassifier:
    """Train a fraud-detection model and report held-out accuracy.

    When called from ``heal()`` after drift is detected, ``data_path`` should
    point to the *current* data so the new model learns the shifted distribution.
    The default (reference data) is kept for standalone / test use.
    """
    df = pd.read_csv(data_path)

    feature_cols = [
        "user_transaction_count",
        "user_transaction_amount_avg",
        "user_transaction_amount_max",
    ]
    X = df[feature_cols]
    y = df["is_fraud"]

    if len(X) == 0:
        raise ValueError(f"No rows in training data: {data_path}")

    if y.isna().any() or X.isna().any().any():
        raise ValueError(f"NaN values detected in training data: {data_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained — held-out accuracy: {accuracy:.4f}")
    return model


def save_model(model: RandomForestClassifier) -> Path:
    """Persist the model with a UTC-timestamped filename."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MODELS_DIR / f"fraud_model_{timestamp}.pkl"
    # joblib (pickle-based) is used here because scikit-learn objects are not
    # expressible in a safer format (e.g. ONNX would be preferable for
    # cross-platform serving, but is out of scope for this pipeline).
    # Models are written by this process and read back only within the same
    # trusted CI/CD environment — never loaded from an external or user-supplied
    # source.  If the threat model changes (e.g. model registry from untrusted
    # parties), migrate to skops or ONNX with an explicit type allowlist.
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def heal() -> None:
    """Run the full self-healing loop: monitor → decide → retrain."""
    drift_score = run_monitor()

    print(f"\nDrift threshold: {DRIFT_THRESHOLD}")
    print(f"Current drift:   {drift_score:.4f}")

    if drift_score >= DRIFT_THRESHOLD:
        print("Drift Detected! Triggering retraining...")
        # Retrain on the *current* (drifted) data so the new model fits the
        # production distribution, not the stale reference distribution.
        try:
            model = train_model(data_path="data/current.csv")
            save_model(model)
            print("Self-healing retraining complete.")
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
            print(f"Error during retraining: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No significant drift detected. Model is healthy.")


if __name__ == "__main__":
    heal()
