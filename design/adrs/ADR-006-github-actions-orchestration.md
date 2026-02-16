# ADR-006: Chosen GitHub Actions Over Airflow/Kubeflow for Pipeline Orchestration

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The self-healing pipeline needs an orchestrator to run the monitor and retrain cycle
on a schedule. Options considered:

| Option | Description |
|---|---|
| **GitHub Actions** | Built-in CI/CD with cron scheduling, artifact uploads, and zero additional infrastructure |
| **Apache Airflow** | DAG-based workflow orchestration, rich scheduling, requires a persistent server |
| **Kubeflow Pipelines** | Kubernetes-native ML pipelines, strong ML lifecycle support, heavy infrastructure |
| **Prefect / Dagster** | Modern Python-native orchestrators with UI, observability, and cloud offerings |

## Decision

**Use GitHub Actions with a cron schedule (`0 */6 * * *`)** and `workflow_dispatch`
for manual triggers.

## Rationale

1. **Zero infrastructure overhead** — The code is already on GitHub. Actions requires no
   additional servers, databases, or Kubernetes clusters. The runner is provided free
   (within limits) by GitHub.
2. **Native artifact handling** — `actions/upload-artifact@v4` provides built-in artifact
   persistence with download links, eliminating the need for a separate artifact store.
3. **Familiar CI/CD model** — The team already uses GitHub for version control. Adding
   Actions is a natural extension, not a new tool to learn and operate.
4. **Cron + manual trigger** — The `schedule` + `workflow_dispatch` combination covers
   both automated periodic runs and ad-hoc debugging runs with no extra config.
5. **Built-in secrets management** — If the pipeline later needs cloud credentials or
   API keys, GitHub Actions Secrets provides encrypted storage with native injection.

## Trade-offs Accepted

- **No DAG semantics** — GitHub Actions jobs are linear steps, not a DAG. The current
   pipeline is sequential (generate → monitor → retrain), so this is fine. If the pipeline
   grows to have parallel branches or complex dependencies, Airflow-style DAGs would be
   more expressive.
- **No persistent state between runs** — Each workflow run starts from scratch. There's no
   built-in way to track drift scores over time or maintain a model registry. This is
   mitigated by uploading artifacts and could be extended with a lightweight database.
- **6-hour minimum practical cadence** — While cron supports minute-level granularity,
   GitHub Actions runners have cold-start latency (~30s) and limited free minutes (2000/month
   for free tier). Running more frequently than every 6 hours would consume the budget quickly.
- **No visual pipeline graph** — Airflow and Kubeflow provide visual DAG views. GitHub
   Actions has a step-by-step log view but no graphical pipeline representation. The C4
   diagrams in `design/` compensate for this.

## Consequences

- The workflow is defined in `.github/workflows/mlops.yml` (single file).
- Pipeline modifications require editing the YAML and pushing to `main`.
- Monitoring pipeline health means checking GitHub Actions run history.
- Migrating to Airflow/Kubeflow later would mean translating the linear step sequence
  into DAG tasks — straightforward given the current simplicity.
