# AI-Powered Predictive Maintenance for Industrial Equipment

Internship-ready predictive maintenance system for the UCI AI4I dataset.

## Quickstart (Windows 11 / Python 3.13)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python scripts\download_data.py
    python scripts\validate_data.py
    python scripts\preprocess_data.py
    python scripts\train_model.py
    python scripts\evaluate_model.py
    pytest -q
    streamlit run app.py

## What this project does NOT do
- Predict exact failure timestamps.
- Compute Remaining Useful Life.
- Connect to physical IoT hardware (Live Monitoring is a simulation).

## ML methodology
- Target: `Machine failure`.
- Features: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- Excluded (leakage): `UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- Split: 70/15/15 stratified, seed 42.
- Candidates: Dummy, LogisticRegression, RandomForest, ExtraTrees.
- Threshold: tuned on validation by maximizing F2, frozen before test evaluation.

## Layout
    app.py                Streamlit dashboard (7 pages)
    config/config.py      Paths, schema, constants
    data/                 raw/, processed/, alerts/ (gitignored)
    models/artifacts/     final_model.joblib, threshold.json, metadata
    models/metrics/       comparison, test_metrics, EDA figures
    scripts/              download, validate, preprocess, train, evaluate
    src/                  data, preprocessing, models, prediction, monitoring, alerts, storage
    tests/                pytest suite
    docs/                 architecture, model card, dataset card, install/user/methodology

## Tests
`pytest -q`. Tests requiring a trained model skip cleanly until you run `scripts\train_model.py`.

## License
MIT - see `LICENSE`.