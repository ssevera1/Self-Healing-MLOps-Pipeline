# C4 Level 1 — System Context Diagram

Shows the Self-Healing MLOps Pipeline as a black box, its users, and the
external systems it interacts with.

```mermaid
C4Context
    title System Context — Self-Healing Fraud Detection Pipeline

    Person(mleng, "ML Engineer", "Monitors pipeline health, reviews retrained models, tunes thresholds")
    Person(analyst, "Fraud Analyst", "Consumes fraud predictions, investigates flagged transactions")

    System(pipeline, "Self-Healing MLOps Pipeline", "Detects data drift in transaction features and automatically retrains the fraud detection model when drift exceeds threshold")

    System_Ext(txn_source, "Transaction System", "Upstream source of raw user transaction events (payments, transfers, purchases)")
    System_Ext(github, "GitHub Actions", "CI/CD runner that executes the monitoring and retraining workflow on a 6-hour cron schedule")
    System_Ext(artifact_store, "Artifact Storage", "GitHub Actions Artifacts — stores drift reports and retrained model .pkl files")

    Rel(txn_source, pipeline, "Feeds raw transaction logs", "CSV / Parquet")
    Rel(pipeline, artifact_store, "Uploads drift reports & model versions", "GitHub Artifacts API")
    Rel(github, pipeline, "Triggers scheduled pipeline run", "Cron / workflow_dispatch")
    Rel(mleng, pipeline, "Configures thresholds, reviews reports", "CLI / Git")
    Rel(pipeline, analyst, "Serves fraud predictions via updated model", "Model .pkl")
```

## Key Observations

| Boundary | Description |
|---|---|
| **Inside the system** | Data generation, drift detection, retraining logic, model serialization |
| **Outside the system** | Transaction data source, CI/CD orchestration, artifact persistence |
| **Latency** | Pipeline is batch-oriented (6-hour cycle). Not real-time inference |
| **Trust boundary** | Transaction data is untrusted input; drift detection validates distributional integrity before retraining |
