# ADR-004: Chosen scikit-learn RandomForest Over Gradient Boosting or Deep Learning

**Status:** Accepted
**Date:** 2026-02-16
**Deciders:** ML Engineering Team

## Context

The fraud detection model needs a classification algorithm. Options considered:

| Option | Description |
|---|---|
| **RandomForestClassifier (scikit-learn)** | Ensemble of decision trees, robust to overfitting, no GPU required |
| **XGBoost / LightGBM** | Gradient boosted trees, often higher accuracy, additional native dependencies |
| **PyTorch / TensorFlow neural network** | Deep learning, highest potential accuracy on large datasets, heavy infrastructure |
| **Logistic Regression** | Simple linear model, fast to train, may underfit complex fraud patterns |

## Decision

**Use scikit-learn's `RandomForestClassifier` with 100 estimators.**

## Rationale

1. **Zero additional dependencies** — scikit-learn is already required for preprocessing
   and evaluation. No extra C++ libraries (XGBoost) or GPU drivers (PyTorch) needed.
2. **Pickle-serializable** — The model serializes cleanly with Python's `pickle` module,
   producing a self-contained `.pkl` file. No custom serialization or model registry needed.
3. **Fast retraining** — 100 trees on 1000 rows trains in < 1 second. This matters because
   retraining happens inside a CI/CD job with limited compute and wall-clock budget.
4. **Interpretable** — Feature importances are built-in (`model.feature_importances_`),
   which aids fraud analysts in understanding model decisions.
5. **Robust defaults** — RandomForest handles class imbalance (95/5 fraud ratio)
   reasonably well without extensive hyperparameter tuning.

## Trade-offs Accepted

- **Lower peak accuracy** — On large, complex fraud datasets, gradient boosting (XGBoost,
  LightGBM) or neural networks typically outperform RandomForest by 1–3% AUC. Acceptable
  for an MVP where the self-healing architecture matters more than model accuracy.
- **No incremental learning** — RandomForest must retrain from scratch on the full dataset.
  With 1000 rows this is trivial; at production scale (millions of rows), this becomes a
  bottleneck. Mitigation: swap to an incremental learner when data volume warrants it.
- **Model size** — A 100-tree RandomForest on 3 features produces a ~200 KB pickle file.
  At scale (more features, more trees), this could grow to 50+ MB. Currently negligible.

## Consequences

- `train_model()` in `src/retrain_trigger.py` is the single function responsible for
  model training. Swapping algorithms means changing only this function.
- The model artifact is a `.pkl` file. Consumers must use the same scikit-learn version
  for deserialization (pickle compatibility). Production should pin scikit-learn version.
- Hyperparameters (`n_estimators=100`, `random_state=42`) are hardcoded. A future
  improvement would be to expose these via config or run hyperparameter search.
