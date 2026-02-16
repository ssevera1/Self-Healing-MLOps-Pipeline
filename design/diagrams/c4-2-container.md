# C4 Level 2 — Container Diagram

Zooms into the Self-Healing MLOps Pipeline to show the major deployable units
(containers), their responsibilities, and the data flows between them.

```mermaid
C4Container
    title Container Diagram — Self-Healing MLOps Pipeline

    Person(mleng, "ML Engineer", "Tunes thresholds, reviews drift reports")

    System_Boundary(pipeline, "Self-Healing MLOps Pipeline") {
        Container(datagen, "Data Generator", "Python / NumPy", "Produces synthetic reference and current datasets simulating transaction feature distributions")
        Container(feast, "Feast Feature Store", "Feast / SQLite / Parquet", "Defines and serves user_transaction_count and related features with entity tracking and TTL management")
        Container(monitor, "Drift Monitor", "Python / Evidently AI", "Compares reference vs current feature distributions using DataDriftPreset. Outputs drift_share score and JSON report")
        Container(healer, "Retrain Trigger", "Python / scikit-learn", "Orchestrates self-healing: evaluates drift score against threshold (0.3), retrains RandomForestClassifier, saves timestamped model")
        ContainerDb(datastore, "Data Store", "Local filesystem", "CSV datasets, Parquet feature files, JSON drift reports")
        ContainerDb(modelstore, "Model Store", "Local filesystem", "Versioned .pkl model artifacts with UTC timestamps")
    }

    System_Ext(github, "GitHub Actions", "Cron-scheduled CI/CD runner")

    Rel(github, healer, "Triggers on cron schedule", "workflow_dispatch / cron")
    Rel(healer, monitor, "Calls to get drift score", "Python import")
    Rel(monitor, datastore, "Reads reference.csv & current.csv", "pandas read_csv")
    Rel(monitor, datastore, "Writes drift_report.json", "JSON")
    Rel(healer, datastore, "Reads training data", "pandas read_csv")
    Rel(healer, modelstore, "Saves retrained model .pkl", "pickle")
    Rel(datagen, datastore, "Generates CSV datasets", "pandas to_csv")
    Rel(feast, datastore, "Reads/writes feature data", "Parquet / SQLite")
    Rel(mleng, healer, "Configures DRIFT_THRESHOLD", "Code edit")
    Rel(mleng, monitor, "Reviews drift_report.json", "File read")
```

## Data Flow Summary

```
┌─────────────┐    CSV     ┌──────────────┐  drift_score  ┌─────────────────┐
│  Data Gen   │ ────────→  │  Drift       │ ────────────→ │  Retrain        │
│  (NumPy)    │            │  Monitor     │               │  Trigger        │
└─────────────┘            │  (Evidently) │               │  (scikit-learn) │
                           └──────┬───────┘               └────────┬────────┘
                                  │                                │
                                  ▼                                ▼
                        drift_report.json              fraud_model_<ts>.pkl
```

## Latency & Throughput

| Container | Typical Latency | Bottleneck |
|---|---|---|
| Data Generator | < 1s (1500 rows) | NumPy RNG |
| Drift Monitor | 2–5s | Evidently statistical tests across 3 columns |
| Retrain Trigger | 3–10s | RandomForest fitting (100 trees × 1000 rows) |
| **Total pipeline** | **~10–15s per cycle** | Dominated by model training |
