# Design Documentation Index

## C4 Model Diagrams

All diagrams use **Mermaid.js** syntax and render natively on GitHub.

| Level | File | What It Shows |
|---|---|---|
| **L1 — Context** | [diagrams/c4-1-context.md](diagrams/c4-1-context.md) | System boundary, actors (ML Engineer, Fraud Analyst), external systems (Transaction Source, GitHub Actions, Artifact Storage) |
| **L2 — Container** | [diagrams/c4-2-container.md](diagrams/c4-2-container.md) | Deployable units: Data Generator, Feast, Drift Monitor, Retrain Trigger, Data Store, Model Store. Data flows and latency estimates |
| **L3 — Component** | [diagrams/c4-3-component.md](diagrams/c4-3-component.md) | Internal components of Monitor (loader, reporter, extractor), Trigger (heal loop, trainer, saver), and Feast (entity, source, feature view) |
| **L4 — Code** | [diagrams/c4-4-code.md](diagrams/c4-4-code.md) | Module class diagram, full execution sequence diagram, file I/O map |

## Architecture Decision Records (ADRs)

| ADR | Title | Key Trade-off |
|---|---|---|
| [ADR-001](adrs/ADR-001-feast-feature-store.md) | Feast for Feature Store | Schema-as-code and migration path vs. extra dependency weight and no real-time serving |
| [ADR-002](adrs/ADR-002-evidently-drift-detection.md) | Evidently AI for Drift Detection | Batteries-included drift presets vs. heavier dependency and opinionated test selection |
| [ADR-003](adrs/ADR-003-threshold-retraining-strategy.md) | Static Threshold Retraining | Simplicity and debuggability vs. no adaptive behavior or cooldown logic |
| [ADR-004](adrs/ADR-004-scikit-learn-random-forest.md) | scikit-learn RandomForest | Zero extra deps and fast training vs. lower peak accuracy and no incremental learning |
| [ADR-005](adrs/ADR-005-local-filesystem-storage.md) | Local Filesystem Storage | Zero infrastructure vs. no cross-run state and 90-day artifact retention limit |
| [ADR-006](adrs/ADR-006-github-actions-orchestration.md) | GitHub Actions Orchestration | Native CI/CD integration vs. no DAG semantics and no persistent state |
