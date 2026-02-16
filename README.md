# Self-Healing MLOps Retraining Pipeline

A self-healing pipeline for a fraud detection model. When data drift is detected in incoming transaction features, the system automatically retrains the model and saves a new versioned artifact — no human intervention required.

```
Transaction Data ──→ Drift Monitor (Evidently) ──→ drift_score > 0.3? ──→ Retrain & Save
                                                          │
                                                          └── No drift ──→ Skip
```

## Tech Stack

| Layer | Tool |
|---|---|
| Feature Store | [Feast](https://feast.dev/) (local provider, SQLite + Parquet) |
| Drift Detection | [Evidently AI](https://www.evidentlyai.com/) (DataDriftPreset) |
| ML Framework | [scikit-learn](https://scikit-learn.org/) (RandomForestClassifier) |
| Orchestration | GitHub Actions (cron every 6 hours) |

## Project Structure

```
├── .github/workflows/
│   └── mlops.yml                  # Scheduled CI/CD pipeline
├── design/
│   ├── diagrams/                  # C4 architecture diagrams (Mermaid.js)
│   │   ├── c4-1-context.md        #   L1: System boundary & actors
│   │   ├── c4-2-container.md      #   L2: Deployable units & data flows
│   │   ├── c4-3-component.md      #   L3: Internal components
│   │   └── c4-4-code.md           #   L4: Module classes & sequence diagram
│   ├── adrs/                      # Architecture Decision Records
│   │   ├── ADR-001-feast-feature-store.md
│   │   ├── ADR-002-evidently-drift-detection.md
│   │   ├── ADR-003-threshold-retraining-strategy.md
│   │   ├── ADR-004-scikit-learn-random-forest.md
│   │   ├── ADR-005-local-filesystem-storage.md
│   │   └── ADR-006-github-actions-orchestration.md
│   └── INDEX.md                   # Design docs index
├── feature_repo/
│   ├── feature_store.yaml         # Feast project config
│   └── features.py                # Feature definitions (user_transaction_count, etc.)
├── src/
│   ├── generate_data.py           # Synthetic data generator (with/without drift)
│   ├── monitor.py                 # Evidently drift detection
│   └── retrain_trigger.py         # Self-healing orchestrator
├── tests/
│   ├── test_monitor.py            # 14 tests for drift monitoring
│   └── test_retrain_trigger.py    # 16 tests for retraining logic
├── data/                          # Runtime artifacts (gitignored)
├── models/                        # Versioned .pkl models (gitignored)
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic datasets (reference + drifted current)
python -m src.generate_data

# Run drift detection only
python -m src.monitor

# Run the full self-healing loop (monitor → decide → retrain)
python -m src.retrain_trigger

# Run tests
python -m pytest tests/ -v
```

## How It Works

### 1. Feature Store (Feast)

Defines transaction features as code in `feature_repo/features.py`:

- **Entity**: `user_id`
- **Features**: `user_transaction_count` (Int64), `user_transaction_amount_avg` (Float32), `user_transaction_amount_max` (Float32)
- **TTL**: 1 day
- **Source**: Parquet file with `event_timestamp` field

### 2. Drift Monitor (`src/monitor.py`)

Compares a **reference** (historical) dataset against a **current** (new logs) dataset using Evidently's `DataDriftPreset`:

1. Loads both CSVs and selects the 3 feature columns
2. Runs KS / Wasserstein statistical tests per column
3. Computes `drift_share` — the fraction of columns flagged as drifted (0.0–1.0)
4. Saves the full JSON report to `data/drift_report.json`

### 3. Self-Healing Trigger (`src/retrain_trigger.py`)

Orchestrates the retraining decision:

- Calls the drift monitor to get the current `drift_score`
- If `drift_score > 0.3` (configurable via `DRIFT_THRESHOLD`):
  - Prints `"Drift Detected! Triggering retraining..."`
  - Trains a `RandomForestClassifier` (100 estimators) on the reference data
  - Saves the model as `models/fraud_model_<UTC_timestamp>.pkl`
- Otherwise: prints `"No significant drift detected. Model is healthy."`

### 4. GitHub Actions (`.github/workflows/mlops.yml`)

Runs the full pipeline on a schedule:

- **Trigger**: Cron (`0 */6 * * *` — every 6 hours) + manual `workflow_dispatch`
- **Steps**: Install deps → generate data → run retrain trigger → upload artifacts
- **Artifacts**: `drift_report.json` and any new `*.pkl` model files

## Tests

30 tests covering all pipeline components:

```
tests/test_monitor.py         — 14 tests
  TestLoadDatasets             (4)  CSV loading, column filtering, error handling
  TestRunDriftReport           (3)  Report structure, JSON persistence
  TestExtractDriftScore        (4)  Score extraction, edge cases
  TestDriftDetection           (3)  Behavioral: low/high drift detection

tests/test_retrain_trigger.py — 16 tests
  TestTrainModel               (6)  Model type, fitting, prediction, accuracy
  TestSaveModel                (4)  Serialization, naming, directory creation
  TestHeal                     (4)  Threshold boundary behavior
  TestConfig                   (2)  Configuration constants
```

```bash
python -m pytest tests/ -v
# ======================= 30 passed in ~6s ========================
```

## Design Documentation

The `design/` directory contains full architectural documentation:

- **C4 Diagrams** (Mermaid.js — render directly on GitHub): Context, Container, Component, and Code level views of the system
- **ADRs**: 6 Architecture Decision Records documenting the reasoning behind each major technology choice and its trade-offs

See [`design/INDEX.md`](design/INDEX.md) for the full index.

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `DRIFT_THRESHOLD` | `src/retrain_trigger.py:19` | `0.3` | Fraction of drifted columns that triggers retraining |
| `n_estimators` | `src/retrain_trigger.py:35` | `100` | Number of trees in the RandomForest |
| `FEATURE_COLUMNS` | `src/monitor.py:16` | 3 columns | Feature columns to monitor for drift |
| Cron schedule | `.github/workflows/mlops.yml:5` | `0 */6 * * *` | Pipeline execution frequency |

## License

This project is provided as-is for educational and demonstration purposes.
