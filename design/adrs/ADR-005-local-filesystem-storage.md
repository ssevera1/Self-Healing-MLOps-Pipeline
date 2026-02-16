# ADR-005: Chosen Local Filesystem Over Cloud Object Storage for Artifacts

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The pipeline produces data artifacts (CSVs, drift reports) and model artifacts (.pkl
files) that need to be stored between pipeline stages. Options considered:

| Option | Description |
|---|---|
| **Local filesystem** | Plain files in `data/` and `models/` directories |
| **AWS S3 / GCS** | Cloud object storage with versioning, lifecycle policies, and IAM |
| **MLflow Model Registry** | Dedicated ML artifact store with model versioning, staging, and metadata |
| **DVC (Data Version Control)** | Git-like versioning for data and model files, backed by remote storage |

## Decision

**Use local filesystem storage** with `data/` for runtime artifacts and `models/` for
model versions. Both directories are gitignored. GitHub Actions Artifacts are used for
persistence across CI/CD runs.

## Rationale

1. **Zero infrastructure** — No cloud accounts, IAM roles, bucket policies, or external
   services to configure. The pipeline runs entirely on the GitHub Actions runner's disk.
2. **Transparent debugging** — Files are plain CSVs and pickles. Engineers can inspect
   them locally with standard tools (pandas, Python REPL, `jq` for JSON).
3. **GitHub Artifacts as persistence** — The workflow uploads `drift_report.json` and
   `models/*.pkl` as GitHub Actions Artifacts, providing 90-day retention with download
   links. This bridges the gap between ephemeral runner storage and permanent archival.
4. **Timestamp-based versioning** — Model files are named `fraud_model_YYYYMMDDTHHMMSSz.pkl`,
   giving a natural version history without needing a registry database.

## Trade-offs Accepted

- **No cross-run state on the runner** — Each GitHub Actions run starts with a fresh
  filesystem. The pipeline must regenerate reference data each run (or fetch it from
  an external source). Currently the synthetic data generator handles this.
- **No model staging/promotion** — There's no concept of "staging" vs "production" model
  versions. Every retrained model is saved with equal status. Production would need
  MLflow or a custom registry to manage model lifecycle states.
- **90-day artifact retention** — GitHub Actions Artifacts expire after 90 days by default.
  For long-term model auditing, artifacts should be copied to durable storage (S3, GCS).
- **No concurrent access control** — Local filesystem has no locking. If two pipeline runs
  overlap (unlikely with 6-hour cron), they could race on file writes. The cron schedule
  and single-runner configuration mitigate this.

## Consequences

- `data/` and `models/` are in `.gitignore` — artifacts never enter version control.
- `retrain_trigger.py` creates `models/` on first use (`MODELS_DIR.mkdir(parents=True, exist_ok=True)`).
- The GitHub Actions workflow has explicit `upload-artifact` steps for persistence.
- Migrating to cloud storage requires changing file paths in `monitor.py` and
  `retrain_trigger.py` to use `boto3` / `gcsfs` / equivalent clients.
