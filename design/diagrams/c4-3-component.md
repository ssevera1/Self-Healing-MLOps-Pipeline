# C4 Level 3 — Component Diagram

Zooms into the **Drift Monitor** and **Retrain Trigger** containers to show
their internal components and interactions.

## Drift Monitor — Internal Components

```mermaid
C4Component
    title Component Diagram — Drift Monitor (src/monitor.py)

    Container_Boundary(monitor, "Drift Monitor") {
        Component(loader, "Dataset Loader", "load_datasets()", "Reads reference.csv and current.csv, selects FEATURE_COLUMNS, returns two DataFrames")
        Component(reporter, "Drift Reporter", "run_drift_report()", "Configures Evidently Report with DataDriftPreset, executes statistical tests (Wasserstein, KS, etc.), persists JSON report")
        Component(extractor, "Score Extractor", "extract_drift_score()", "Parses Evidently result dict, locates DatasetDriftMetric, extracts drift_share float")
        Component(orchestrator, "Monitor Orchestrator", "main()", "Coordinates load → report → extract. Returns drift_score to caller")
    }

    ContainerDb(csvstore, "CSV Data Store", "reference.csv, current.csv")
    ContainerDb(jsonstore, "Report Store", "drift_report.json")

    Rel(orchestrator, loader, "1. Load data")
    Rel(orchestrator, reporter, "2. Run drift analysis")
    Rel(orchestrator, extractor, "3. Extract score")
    Rel(loader, csvstore, "Reads CSVs", "pandas")
    Rel(reporter, jsonstore, "Writes JSON report", "json.dump")
```

## Retrain Trigger — Internal Components

```mermaid
C4Component
    title Component Diagram — Retrain Trigger (src/retrain_trigger.py)

    Container_Boundary(healer, "Retrain Trigger") {
        Component(heal_loop, "Heal Orchestrator", "heal()", "Runs monitor, compares drift_score against DRIFT_THRESHOLD (0.3), decides whether to retrain")
        Component(trainer, "Model Trainer", "train_model()", "Loads reference data, splits into X/y, fits RandomForestClassifier(n_estimators=100), reports training accuracy")
        Component(saver, "Model Saver", "save_model()", "Serializes model with pickle, generates UTC-timestamped filename, writes to models/ directory")
        Component(threshold, "Drift Threshold", "DRIFT_THRESHOLD = 0.3", "Configurable constant governing the retrain/no-retrain decision boundary")
    }

    Container_Ext(monitor, "Drift Monitor", "src/monitor.py")
    ContainerDb(modelstore, "Model Store", "models/*.pkl")
    ContainerDb(datastore, "Data Store", "reference.csv")

    Rel(heal_loop, monitor, "Calls main() → drift_score", "Python import")
    Rel(heal_loop, threshold, "Reads threshold value")
    Rel(heal_loop, trainer, "Invokes if score > threshold")
    Rel(trainer, datastore, "Reads training data", "pandas")
    Rel(heal_loop, saver, "Passes trained model")
    Rel(saver, modelstore, "Writes .pkl file", "pickle")
```

## Feast Feature Store — Internal Components

```mermaid
C4Component
    title Component Diagram — Feast Feature Store (feature_repo/)

    Container_Boundary(feast, "Feast Feature Store") {
        Component(entity, "User Entity", "user_id", "Join key that ties feature rows to individual users")
        Component(source, "File Source", "FileSource", "Points to data/user_transactions.parquet with event_timestamp field")
        Component(fv, "Feature View", "user_transaction_features", "Exposes user_transaction_count (Int64), user_transaction_amount_avg (Float32), user_transaction_amount_max (Float32) with 1-day TTL")
        Component(config, "Store Config", "feature_store.yaml", "Project: fraud_detection, provider: local, registry: SQLite, online store: SQLite, offline store: file")
    }

    ContainerDb(parquet, "Parquet Store", "data/user_transactions.parquet")
    ContainerDb(registry, "Registry DB", "data/registry.db")
    ContainerDb(online, "Online Store", "data/online_store.db")

    Rel(source, parquet, "Reads feature data")
    Rel(fv, source, "Backed by")
    Rel(fv, entity, "Keyed by user_id")
    Rel(config, registry, "Tracks metadata")
    Rel(config, online, "Serves online features")
```

## Component Interaction Map

```
                    ┌────────────────────────────────────────────┐
                    │            Retrain Trigger                  │
                    │  ┌──────────┐  ┌─────────┐  ┌──────────┐  │
  GitHub ──────────→│  │  heal()  │→ │ train() │→ │  save()  │  │
  Actions           │  └────┬─────┘  └─────────┘  └──────────┘  │
                    │       │ drift_score                         │
                    │       ▼                                     │
                    │  ┌────────────────────────────────────┐    │
                    │  │         Drift Monitor               │    │
                    │  │  load() → report() → extract()     │    │
                    │  └────────────────────────────────────┘    │
                    └────────────────────────────────────────────┘
```
