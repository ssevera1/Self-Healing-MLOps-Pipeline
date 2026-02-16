# C4 Level 4 — Code Diagram

Module and function-level view of the pipeline codebase. Shows the actual
Python modules, their public interfaces, and call relationships.

## Module Structure — Class Diagram

```mermaid
classDiagram
    direction TB

    class generate_data {
        +generate_reference_data(n_samples, seed) DataFrame
        +generate_current_data(n_samples, seed, drift) DataFrame
        +main()
    }
    note for generate_data "src/generate_data.py\nNumPy RNG-based synthetic data\nPoisson + Normal distributions"

    class monitor {
        +FEATURE_COLUMNS : list~str~
        +REPORT_PATH : Path
        +load_datasets(reference_path, current_path) tuple~DataFrame, DataFrame~
        +run_drift_report(reference, current) dict
        +extract_drift_score(report_dict) float
        +main() float
    }
    note for monitor "src/monitor.py\nEvidently DataDriftPreset\nReturns drift_share ∈ [0.0, 1.0]"

    class retrain_trigger {
        +DRIFT_THRESHOLD : float = 0.3
        +MODELS_DIR : Path
        +train_model(data_path) RandomForestClassifier
        +save_model(model) Path
        +heal() void
    }
    note for retrain_trigger "src/retrain_trigger.py\nscikit-learn RandomForest\nUTC-timestamped .pkl output"

    class features {
        +user : Entity
        +user_transactions_source : FileSource
        +user_transaction_features : FeatureView
    }
    note for features "feature_repo/features.py\nFeast entity + feature view\nTTL: 1 day"

    retrain_trigger --> monitor : imports main() as run_monitor
    monitor --> generate_data : consumes output CSVs (file I/O)
    retrain_trigger --> generate_data : reads reference.csv for training
    features --> generate_data : schema aligned (same 3 columns)
```

## Call Graph — Execution Sequence

```mermaid
sequenceDiagram
    participant GHA as GitHub Actions
    participant RT as retrain_trigger.heal()
    participant MON as monitor.main()
    participant LD as monitor.load_datasets()
    participant DR as monitor.run_drift_report()
    participant EX as monitor.extract_drift_score()
    participant TM as retrain_trigger.train_model()
    participant SM as retrain_trigger.save_model()
    participant FS as Filesystem

    GHA->>RT: python -m src.retrain_trigger
    RT->>MON: run_monitor()

    MON->>LD: load_datasets()
    LD->>FS: read reference.csv
    FS-->>LD: DataFrame (1000 × 3)
    LD->>FS: read current.csv
    FS-->>LD: DataFrame (500 × 3)
    LD-->>MON: (reference, current)

    MON->>DR: run_drift_report(ref, cur)
    Note over DR: Evidently Report<br/>DataDriftPreset<br/>KS / Wasserstein tests
    DR->>FS: write drift_report.json
    DR-->>MON: report_dict

    MON->>EX: extract_drift_score(report_dict)
    Note over EX: Find DatasetDriftMetric<br/>→ result.drift_share
    EX-->>MON: drift_score (float)
    MON-->>RT: drift_score

    alt drift_score > 0.3
        Note over RT: "Drift Detected!<br/>Triggering retraining..."
        RT->>TM: train_model()
        TM->>FS: read reference.csv
        FS-->>TM: DataFrame (1000 × 4)
        Note over TM: RandomForestClassifier<br/>n_estimators=100<br/>fit(X, y)
        TM-->>RT: trained model

        RT->>SM: save_model(model)
        Note over SM: timestamp = UTC now<br/>fraud_model_YYYYMMDDTHHMMSSz.pkl
        SM->>FS: write .pkl
        SM-->>RT: model path
    else drift_score ≤ 0.3
        Note over RT: "Model is healthy."
    end
```

## File I/O Map

```mermaid
flowchart LR
    subgraph Inputs
        REF[reference.csv<br/>1000 rows × 4 cols]
        CUR[current.csv<br/>500 rows × 4 cols]
    end

    subgraph Processing
        MON[monitor.py<br/>Evidently DataDrift]
        RT[retrain_trigger.py<br/>RandomForest fit]
    end

    subgraph Outputs
        RPT[drift_report.json<br/>Full Evidently report]
        MDL[fraud_model_&lt;ts&gt;.pkl<br/>Serialized classifier]
    end

    REF --> MON
    CUR --> MON
    MON --> RPT
    MON -.->|drift_score| RT
    REF --> RT
    RT --> MDL
```
