# Architecture

Data acquisition -> validation -> preprocessing (ColumnTransformer,
70/15/15 stratified split) -> training (4 candidates, F2 threshold tuning
on val) -> artifacts (final_model.joblib, threshold.json, metadata) ->
prediction engine -> monitoring controller -> alert engine -> SQLite ->
Streamlit dashboard.

## Separation of concerns
- src/data/          acquisition + validation only.
- src/preprocessing/ splits, transformer, EDA.
- src/models/        training and evaluation.
- src/prediction/    inference engine; depends on trained artifacts only.
- src/monitoring/    sensor stream + controller; depends on prediction.
- src/alerts/        rule engine; no I/O.
- src/storage/       SQLite only.
- app.py             orchestrates pages; never contains model logic.