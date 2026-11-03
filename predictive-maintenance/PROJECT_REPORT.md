# AI-Powered Predictive Maintenance for Industrial Equipment
### Internship Project Report

> Every numeric result in this report is populated from your run of
> `scripts/train_model.py` and `models/metrics/test_metrics.json`.
> Do not publish this report with placeholder metrics.

## 1. Abstract
We develop a predictive maintenance system over the UCI AI4I dataset.
A class-balanced classifier estimates machine-failure probability from
operating measurements; an F2-tuned threshold is frozen before test
evaluation. A modular architecture (data -> preprocessing -> modeling ->
inference -> monitoring -> alerts -> dashboard) keeps every stage
replaceable and testable.

## 2. Objectives
1. Train an ML failure-prediction model on AI4I 2020.
2. Implement equipment-health monitoring (simulated).
3. Generate maintenance alerts and recommendations.
4. Deliver an interactive dashboard.

## 3. Scope
In-scope: model training, threshold tuning, batch/manual inference, simulated
live monitoring, alerts persisted in SQLite, Streamlit UI.
Out-of-scope: exact failure-time forecasting, RUL, physical IoT integration,
LLM/RAG augmentation.

## 4. Dataset
- Source: UCI ML Repository, dataset 601, AI4I 2020.
- Rows: ~10,000.
- Features: `Type`, `Air temperature [K]`, `Process temperature [K]`,
  `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- Target: `Machine failure` (binary).
- Excluded (leakage): `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`,
  `RNF`, and the target itself.

## 5. Preprocessing
`ColumnTransformer`:
- Numeric: `SimpleImputer(median)` -> `StandardScaler`.
- Categorical: `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")`.
Fitting happens inside the pipeline on the training split only.

## 6. Training procedure
- Stratified 70/15/15 split with `random_state=42`.
- Each candidate fit on train; probabilities on val; threshold tuned by
  maximizing F2.
- Best candidate by validation F2 selected.
- Best candidate refit on train+val with the frozen threshold.
- Final untouched test evaluation recorded to
  `models/metrics/test_metrics.json`.

## 7. Experimental results
Fill from `models/metrics/test_metrics.json` after training.

| metric | value |
|---|---|
| Precision | - |
| Recall | - |
| F1 | - |
| F2 | - |
| ROC-AUC | - |
| Avg Precision | - |
| Threshold | - |

Per-candidate validation metrics are in `models/metrics/model_comparison.csv`.

## 8. System architecture
See `docs/ARCHITECTURE.md`.

## 9. Prediction engine
`src/prediction/engine.py`. Loads `final_model.joblib` + `threshold.json`,
validates inputs strictly, returns `PredictionResult` with probability,
binary flag, risk level, operating-range notes, recommendation, timestamp.

## 10. Monitoring implementation
`src/monitoring/stream.py` replays `FEATURE_COLUMNS` from the processed CSV.
`MonitorController` exposes start/pause/stop/step/reset. Dashboard keeps
the controller in `st.session_state`. Explicitly labeled *simulation*.

## 11. Alert-generation mechanism
`src/alerts/engine.py` raises an alert when `failure_probability >= threshold`.
Cooldown suppression (default 300 s) applies per `(equipment_id, risk_level)`.
Persistence in SQLite via `src/storage/alerts_db.py` with acknowledgement
and CSV export.

## 12. Dashboard design
Seven pages: Overview, Live Monitoring (simulated), Manual Prediction,
Batch Analysis, Model Performance, Alerts & Maintenance History,
About & Dataset. Charts use Plotly; styling follows an industrial
monitoring aesthetic with metric cards and clear empty states.

## 13. System requirements
Windows 11 or Linux; Python 3.13; ~1 GB free disk; CPU only.

## 14. Installation
See `docs/INSTALLATION.md`.

## 15. Testing results
Run `pytest -q` and paste the summary line here. Tests requiring a
trained model skip cleanly until `scripts\train_model.py` has run.

## 16. Limitations
- Binary classification only; no temporal forecasting or RUL.
- Simulation, not real hardware.
- Statistical reference ranges, not manufacturer safety limits.
- Class imbalance makes ROC-AUC alone misleading; PR metrics primary.

## 17. Future enhancements
- Real PLC/OPC-UA sensor integration behind the existing stream interface.
- RUL modeling with a properly defined target and survival analysis.
- Alert escalations with on-call routing.
- Drift monitoring on live feature distributions.

## 18. References
- UCI ML Repository, dataset 601: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Streamlit Documentation: https://docs.streamlit.io