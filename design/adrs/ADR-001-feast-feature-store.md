# ADR-001: Chosen Feast for Feature Store Over Custom Feature Pipelines

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The fraud detection pipeline needs a structured way to define, store, and serve
transaction-based features (e.g., `user_transaction_count`). Options considered:

| Option | Description |
|---|---|
| **Feast** | Open-source feature store with entity tracking, offline/online serving, TTL, and versioned feature definitions |
| **Custom pandas pipelines** | Ad-hoc scripts that load CSVs and compute features inline |
| **Tecton / Hopsworks** | Managed/enterprise feature platforms with richer ML lifecycle features |

## Decision

**Use Feast with the local provider** (file-based offline store, SQLite online store).

## Rationale

1. **Schema as code** — Feature definitions in `features.py` are version-controlled,
   reviewable, and serve as a single source of truth for column names, types, and TTLs.
2. **Decoupled feature definitions from consumption** — The monitor and trainer both
   reference the same feature schema without duplicating column lists in ad-hoc code.
3. **Minimal infrastructure** — The local provider requires no external services (no
   Redis, no DynamoDB). SQLite + Parquet is sufficient for this batch-oriented pipeline.
4. **Migration path** — Feast supports GCP, AWS, and Snowflake providers. Moving to
   production means changing `feature_store.yaml`, not rewriting feature logic.

## Trade-offs Accepted

- **No real-time feature serving** — The local SQLite online store can't handle
  sub-millisecond serving at scale. Acceptable because this pipeline is batch-oriented
  (6-hour cycle), not a real-time inference service.
- **Extra dependency** — Feast adds ~50 MB of transitive dependencies (PyArrow, protobuf,
  etc.). Justified by the schema enforcement and future migration path it provides.
- **Feature materialization not wired** — The current pipeline generates synthetic data
  directly as CSVs rather than materializing through Feast. This is intentional for the
  MVP; production would run `feast materialize` to populate the online store.

## Consequences

- Feature column names (`user_transaction_count`, `user_transaction_amount_avg`,
  `user_transaction_amount_max`) are defined once in `feature_repo/features.py` and
  referenced by convention in `monitor.py` and `retrain_trigger.py`.
- Adding a new feature requires updating `features.py` and both consumer modules.
- The `feature_store.yaml` is the configuration entry point for changing storage backends.
