# Model Card

## Model
Scikit-learn `Pipeline(preprocessor -> estimator)`. Estimator chosen by
validation F2 among Dummy, LogisticRegression, RandomForest, ExtraTrees.

## Intended use
Advisory failure-risk estimation for industrial machines with sensor
features matching `config.FEATURE_COLUMNS`.

## Not intended for
- Exact failure-time prediction.
- RUL estimation.
- Autonomous control or safety interlocks.
- Deployment without human review of alerts.

## Training data
AI4I 2020. Stratified 70/15/15 split with seed 42.

## Threshold
Chosen on validation by maximizing F2. Frozen before test evaluation.
Stored in `models/artifacts/threshold.json`.

## Metrics
Produced on the untouched test split; stored in
`models/metrics/test_metrics.json`. Per-candidate validation comparison
in `models/metrics/model_comparison.csv`.

## Ethical considerations
Predictions are statistical, not guarantees. Alerts must be reviewed by
qualified personnel. Do not shut down machinery solely because of a
model score.

## Limitations
- Class imbalance makes accuracy misleading; PR metrics are primary.
- Simulation environment: no physical sensor stream.
- Reference ranges are statistical percentiles, not manufacturer limits.