# Methodology

## Splits
Stratified 70/15/15, seed 42 (two successive stratified splits with
proportion assertions).

## Preprocessing
Inside the sklearn pipeline only. Fitting happens on the training split.
Numeric: median impute + StandardScaler. Categorical: most-frequent impute
+ OneHotEncoder(handle_unknown="ignore").

## Threshold tuning
`precision_recall_curve` on validation probabilities. F2 maximized:
`F2 = 5*P*R / (4*P + R)`. Chosen threshold is persisted and frozen
before final test evaluation.

## Final evaluation
Refit best model on train + val, evaluate on test with the frozen threshold.

## No leakage guarantees
- Feature selection is a whitelist (`FEATURE_COLUMNS`).
- Blacklist (`LEAKAGE_COLS`) enforced by unit tests.
- Preprocessor fit only inside the pipeline on training data.
- Test data never influences threshold or model selection.