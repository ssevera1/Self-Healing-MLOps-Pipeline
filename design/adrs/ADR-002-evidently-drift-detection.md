# ADR-002: Chosen Evidently AI for Drift Detection Over Statistical Tests from Scratch

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The self-healing pipeline must detect when the distribution of incoming transaction
features has shifted enough to degrade model performance. Options considered:

| Option | Description |
|---|---|
| **Evidently AI** | Open-source ML monitoring library with pre-built drift presets, statistical tests (KS, Wasserstein, chi-squared), and structured JSON reports |
| **Manual scipy.stats** | Hand-rolled KS tests or PSI calculations using scipy |
| **NannyML** | Specialized in performance estimation without ground truth |
| **Whylogs / WhyLabs** | Profile-based monitoring with statistical drift detection |

## Decision

**Use Evidently AI's `DataDriftPreset`** as the drift detection engine.

## Rationale

1. **Batteries included** — `DataDriftPreset` automatically selects the right statistical
   test per column type (KS for numerical, chi-squared for categorical) and computes a
   dataset-level `drift_share` metric. No manual test selection needed.
2. **Structured output** — The `.as_dict()` method returns a well-structured JSON report
   that can be persisted, parsed, and consumed by downstream automation (the retrain trigger).
3. **Interpretable** — Evidently produces per-column drift scores and p-values, making it
   easy for ML engineers to diagnose *which* features drifted and *by how much*.
4. **Active ecosystem** — Evidently has strong community adoption, regular releases, and
   integration with dashboards (Evidently UI, Grafana) for future observability.

## Trade-offs Accepted

- **Heavier than scipy** — Evidently pulls in plotly, requests, and other dependencies
  (~100 MB). Acceptable for a CI/CD pipeline that installs from scratch each run.
- **Opinionated test selection** — Evidently auto-selects statistical tests. If we needed
  a specific test (e.g., Population Stability Index), we'd need to configure custom metrics.
  For the current 3-column numerical schema, the defaults (KS/Wasserstein) are appropriate.
- **Batch-only** — Evidently compares two static DataFrames. It doesn't support streaming
  drift detection. This aligns with our 6-hour batch cycle design.

## Consequences

- `monitor.py` depends on `evidently.report.Report` and `evidently.metric_preset.DataDriftPreset`.
- The drift score is `drift_share` (fraction of columns detected as drifted, 0.0–1.0),
  not a raw p-value. The retrain threshold (0.3) is calibrated against this metric.
- The full JSON report is saved to `data/drift_report.json` and uploaded as a GitHub
  Actions artifact for auditability.
