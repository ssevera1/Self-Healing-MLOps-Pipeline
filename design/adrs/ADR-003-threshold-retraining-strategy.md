# ADR-003: Chosen Static Threshold Retraining Over Continuous/Adaptive Strategies

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The pipeline needs a decision mechanism to determine when retraining is necessary.
Options considered:

| Option | Description |
|---|---|
| **Static threshold (drift_share > 0.3)** | Simple comparison: if more than 30% of feature columns have drifted, retrain |
| **Sliding window with adaptive threshold** | Track drift scores over time, retrain when score exceeds a rolling mean + N standard deviations |
| **Performance-based trigger** | Monitor model accuracy/F1 on labeled data; retrain when performance drops below a target |
| **Scheduled retraining** | Retrain on every run regardless of drift, relying on fresh data to maintain quality |

## Decision

**Use a static threshold of 0.3 on the `drift_share` metric** (fraction of columns
that Evidently flags as drifted).

## Rationale

1. **Simplicity and debuggability** — A single constant (`DRIFT_THRESHOLD = 0.3`) is
   trivial to understand, audit, and tune. No hidden state, no windowing logic, no
   historical score database required.
2. **Meaningful threshold semantics** — With 3 feature columns, a drift_share of 0.33
   means 1 column drifted; 0.67 means 2; 1.0 means all 3. A threshold of 0.3 triggers
   retraining when *any* column drifts, which is the conservative choice for fraud
   detection where feature stability directly impacts prediction quality.
3. **Avoids unnecessary retraining** — Unlike scheduled retraining, this approach only
   retrains when distributional evidence warrants it, saving compute and preventing
   model churn on stable data.
4. **No labeled data dependency** — Performance-based triggers require ground truth labels
   (which fraud label often arrives days/weeks late). Drift detection works on feature
   distributions alone, enabling faster response.

## Trade-offs Accepted

- **No adaptive behavior** — The threshold doesn't learn from past drift events. If the
  data distribution gradually shifts (concept drift), the static threshold may trigger
  too frequently or not frequently enough. Mitigation: ML engineers can tune the constant.
- **No hysteresis** — The pipeline will retrain on every 6-hour cycle where drift exceeds
  0.3, even if the previous cycle already retrained. For the current MVP this is acceptable;
  production should add cooldown logic (e.g., "don't retrain if last retrain was < 12h ago").
- **Threshold tuning is manual** — Choosing 0.3 was a judgment call calibrated for 3
  columns. If the feature set grows to 50 columns, 0.3 would mean 15 columns drifted,
  which may be too permissive. The threshold must be re-evaluated when the schema changes.

## Consequences

- `DRIFT_THRESHOLD` is defined as a module-level constant in `src/retrain_trigger.py`.
- Changing the threshold requires a code change and redeployment (intentional — threshold
  changes should be reviewed and version-controlled).
- The pipeline prints the threshold and current score on every run for observability.
