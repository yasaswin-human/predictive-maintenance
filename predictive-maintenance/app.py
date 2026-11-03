"""
AI-Powered Predictive Maintenance - Streamlit dashboard.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import (
    COMPARISON_PATH, FEATURE_COLUMNS, MAX_UPLOAD_MB,
    PROCESSED_CSV_PATH, RAW_CSV_PATH, TARGET_COL,
)
from src.alerts.engine import AlertEngine
from src.models.evaluator import compute_curves, load_test_metrics
from src.monitoring.controller import MonitorController
from src.monitoring.stream import SensorStream
from src.prediction.engine import PredictionEngine, PredictionError
from src.storage.alerts_db import (
    acknowledge_alert, fetch_alerts, save_alert,
)

st.set_page_config(
    page_title="Predictive Maintenance - AI4I",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model ...")
def get_engine() -> PredictionEngine:
    ref = PROCESSED_CSV_PATH if PROCESSED_CSV_PATH.exists() else RAW_CSV_PATH
    return PredictionEngine.from_artifacts(reference_csv=ref)


@st.cache_data(show_spinner=False)
def get_dataset_overview() -> dict:
    path = PROCESSED_CSV_PATH if PROCESSED_CSV_PATH.exists() else RAW_CSV_PATH
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out = {
        "rows": len(df),
        "columns": df.shape[1],
        "type_counts": df["Type"].value_counts().to_dict() if "Type" in df.columns else {},
    }
    if TARGET_COL in df.columns:
        out["target_counts"] = df[TARGET_COL].value_counts().to_dict()
    return out


def _need_model():
    try:
        return get_engine()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python scripts/train_model.py` first, "
            "then refresh this page."
        )
        st.stop()


PAGES = [
    "Overview",
    "Live Monitoring",
    "Manual Prediction",
    "Batch Analysis",
    "Model Performance",
    "Alerts & Maintenance History",
    "About & Dataset",
]

st.sidebar.title("Predictive Maintenance")
st.sidebar.caption("AI4I - Industrial equipment health")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    "Advisory decision-support tool. Not a certified industrial safety controller."
)


if page == "Overview":
    st.title("Industrial Equipment Health - Overview")
    st.caption("AI-Powered Predictive Maintenance - AI4I dataset")

    overview = get_dataset_overview()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows", overview.get("rows", "-"))
    c2.metric("Features used", len(FEATURE_COLUMNS))
    tc = overview.get("target_counts", {})
    if tc:
        c3.metric("Healthy records", int(tc.get(0, 0)))
        c4.metric("Failure records", int(tc.get(1, 0)))

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("Failure distribution")
        if tc:
            df_bar = pd.DataFrame({
                "class": ["Healthy", "Failure"],
                "count": [int(tc.get(0, 0)), int(tc.get(1, 0))],
            })
            fig = px.bar(df_bar, x="class", y="count", color="class",
                         color_discrete_sequence=["#4CAF50", "#E53935"])
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Machine Type distribution")
        tcnt = overview.get("type_counts", {})
        if tcnt:
            fig = px.pie(names=list(tcnt.keys()), values=list(tcnt.values()),
                         color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model performance summary (test split)")
    metrics = load_test_metrics()
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        m2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        m3.metric("F2", f"{metrics.get('f2', 0):.3f}")
        m4.metric("Average Precision", f"{metrics.get('avg_precision', 0):.3f}")
        st.caption(f"Model: {metrics.get('name', '-')} - threshold {metrics.get('threshold', 0):.3f}")
    else:
        st.info("No test metrics yet - run `python scripts/train_model.py`.")


elif page == "Live Monitoring":
    st.title("Live Monitoring (Simulated)")
    st.warning(
        "Simulation only. Readings are replayed from the AI4I dataset - "
        "this is not a physical IoT deployment.",
    )
    engine = _need_model()

    if "monitor" not in st.session_state:
        st.session_state.monitor = MonitorController(
            engine=engine, stream=SensorStream(shuffle=True)
        )
        st.session_state.alert_engine = AlertEngine()

    monitor: MonitorController = st.session_state.monitor
    alert_engine: AlertEngine = st.session_state.alert_engine

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Start", use_container_width=True):
        monitor.start()
    if b2.button("Pause", use_container_width=True):
        monitor.pause()
    if b3.button("Stop", use_container_width=True):
        monitor.stop()
    if b4.button("Step", use_container_width=True):
        r = monitor.step()
        if r:
            a = alert_engine.evaluate(r)
            if a:
                save_alert(a)

    st.caption(
        f"Position: {monitor.state.position} / {monitor.state.total} | "
        f"running={monitor.state.running} | paused={monitor.state.paused}"
    )

    speed = st.slider("Playback speed (readings/sec)", 1, 20, 3)
    auto = st.toggle("Auto-advance", value=monitor.state.running and not monitor.state.paused)
    if auto:
        for _ in range(speed):
            r = monitor.step()
            if r is None:
                break
            a = alert_engine.evaluate(r)
            if a:
                save_alert(a)
        time.sleep(1.0)

    latest = monitor.state.readings[-1] if monitor.state.readings else None
    if latest:
        st.subheader("Current reading")
        prob_col, risk_col = st.columns(2)
        prob_col.metric("Failure probability", f"{latest.failure_probability:.3f}")
        risk_col.metric("Risk level", latest.risk_level)
        st.dataframe(pd.DataFrame([latest.input]), use_container_width=True)
        if latest.operating_range_notes:
            st.info("\n\n".join(latest.operating_range_notes))
        st.markdown(f"**Recommendation:** {latest.recommendation}")

    if monitor.state.readings:
        st.subheader("Recent trend")
        hist = pd.DataFrame(
            [{"t": i, "prob": r.failure_probability}
             for i, r in enumerate(monitor.state.readings[-100:])]
        )
        fig = px.line(hist, x="t", y="prob",
                      labels={"t": "reading #", "prob": "Failure probability"})
        fig.add_hline(y=engine.threshold, line_dash="dash", line_color="red",
                      annotation_text=f"threshold={engine.threshold:.3f}")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)


elif page == "Manual Prediction":
    st.title("Manual Prediction")
    engine = _need_model()

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        machine_type = c1.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = c2.number_input("Air temperature [K]", value=298.0,
                                    min_value=200.0, max_value=400.0, step=0.1)
        c3, c4 = st.columns(2)
        proc_temp = c3.number_input("Process temperature [K]", value=308.0,
                                     min_value=200.0, max_value=400.0, step=0.1)
        rpm = c4.number_input("Rotational speed [rpm]", value=1500,
                               min_value=0, max_value=10000, step=10)
        c5, c6 = st.columns(2)
        torque = c5.number_input("Torque [Nm]", value=40.0,
                                  min_value=0.0, max_value=200.0, step=0.5)
        tool_wear = c6.number_input("Tool wear [min]", value=100,
                                     min_value=0, max_value=5000, step=1)
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        row = {
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }
        try:
            result = engine.predict_one(row)
        except PredictionError as e:
            st.error(str(e))
        else:
            a, b, c = st.columns(3)
            a.metric("Failure probability", f"{result.failure_probability:.3f}")
            b.metric("Predicted class", "FAILURE" if result.failure_predicted else "HEALTHY")
            c.metric("Risk level", result.risk_level)
            st.caption(f"Threshold: {result.threshold:.3f}")
            if result.operating_range_notes:
                st.info("\n\n".join(result.operating_range_notes))
            st.success(result.recommendation)


elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.caption(
        "Upload a CSV with one row per reading. Required columns: "
        + ", ".join(FEATURE_COLUMNS) + ". The target column is not required."
    )
    engine = _need_model()
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
            else:
                try:
                    preds = engine.predict_batch(df)
                except PredictionError as e:
                    st.error(str(e))
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("Records processed", len(preds))
                    c2.metric("Flagged as failure risk",
                              int(preds["failure_predicted"].sum()))
                    fig = px.histogram(preds, x="risk_level", color="risk_level",
                                       color_discrete_map={"LOW": "#4CAF50",
                                                           "MEDIUM": "#FFB300",
                                                           "HIGH": "#E53935"})
                    fig.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(preds, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        data=preds.to_csv(index=False).encode(),
                        file_name="predictions.csv", mime="text/csv",
                    )
    else:
        if RAW_CSV_PATH.exists():
            st.info(
                f"Tip: `{RAW_CSV_PATH.name}` already contains the required "
                f"columns and can be uploaded directly as a batch input."
            )


elif page == "Model Performance":
    st.title("Model Performance")
    metrics = load_test_metrics()
    if not metrics:
        st.info("Run `python scripts/train_model.py` first.")
    else:
        st.subheader("Final test metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{metrics['precision']:.3f}")
        c2.metric("Recall", f"{metrics['recall']:.3f}")
        c3.metric("F1", f"{metrics['f1']:.3f}")
        c4.metric("F2", f"{metrics['f2']:.3f}")
        c5, c6, c7 = st.columns(3)
        c5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        c6.metric("Avg precision", f"{metrics.get('avg_precision', 0):.3f}")
        c7.metric("Threshold", f"{metrics['threshold']:.3f}")

        st.subheader("Confusion matrix (test split)")
        cm = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        fig = px.imshow(
            cm, text_auto=True,
            x=["Pred healthy", "Pred failure"],
            y=["Actual healthy", "Actual failure"],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model comparison (validation split)")
        if COMPARISON_PATH.exists():
            comp = pd.read_csv(COMPARISON_PATH)
            show_cols = ["name", "threshold", "precision", "recall", "f1", "f2",
                         "roc_auc", "avg_precision"]
            st.dataframe(comp[show_cols], use_container_width=True)
            fig = px.bar(comp, x="name", y=["f2", "f1", "precision", "recall"],
                         barmode="group")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("ROC and PR curves (test split)")
        path = PROCESSED_CSV_PATH if PROCESSED_CSV_PATH.exists() else RAW_CSV_PATH
        if Path(path).exists():
            from src.preprocessing.splits import stratified_three_way_split
            df = pd.read_csv(path)
            _, _, test_df = stratified_three_way_split(df)
            curves = compute_curves(test_df)
            if curves.get("roc"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curves["roc"]["fpr"], y=curves["roc"]["tpr"],
                                         mode="lines", name="ROC"))
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash"))
                fig.update_layout(height=340, xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)
            if curves.get("pr"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curves["pr"]["recall"],
                                         y=curves["pr"]["precision"],
                                         mode="lines", name="PR"))
                fig.update_layout(height=340, xaxis_title="Recall",
                                  yaxis_title="Precision")
                st.plotly_chart(fig, use_container_width=True)


elif page == "Alerts & Maintenance History":
    st.title("Alerts & Maintenance History")
    c1, c2 = st.columns(2)
    risk_filter = c1.selectbox("Risk level", ["ALL", "HIGH", "MEDIUM", "LOW"])
    days = c2.slider("Last N days", 1, 30, 7)
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()

    df = fetch_alerts(risk_level=risk_filter, since=since)
    if df.empty:
        st.info("No alerts yet. Use Live Monitoring to generate alerts.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Export alerts CSV",
            data=df.to_csv(index=False).encode(),
            file_name="alerts.csv", mime="text/csv",
        )
        ids = ["-"] + df["alert_id"].head(50).tolist()
        to_ack = st.selectbox("Acknowledge alert ID", ids)
        if to_ack != "-" and st.button("Acknowledge"):
            if acknowledge_alert(to_ack):
                st.success(f"Acknowledged {to_ack}")
                st.rerun()


elif page == "About & Dataset":
    st.title("About this project")
    st.markdown(
        """
### Objective
Develop an AI-based predictive maintenance system that predicts potential
machine failures from industrial sensor measurements, monitors equipment
health, generates maintenance alerts, and presents actionable information
through an interactive dashboard.

### Dataset
The [AI4I Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
from the UCI Machine Learning Repository. Approximately 10,000 observations
of industrial machine operating conditions with binary failure labels and
five failure-mode indicators.

**Leakage rule enforced:** the model never sees `UDI`, `Product ID`,
`Machine failure`, or the failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`,
`RNF`). Only the operating measurements (`Type`, air/process temperature,
rotational speed, torque, tool wear) are used as inputs.

### Model architecture
- Scikit-learn `ColumnTransformer` (median-imputed + standardized numeric
  features; one-hot-encoded categorical `Type`).
- Candidates: Dummy, Logistic Regression, Random Forest, Extra Trees -
  all trained with `class_weight="balanced"` where applicable.
- Reproducible stratified split: 70% train / 15% validation / 15% test.
- Decision threshold chosen on validation data by maximizing F2
  (recall-weighted), then **frozen** before final test evaluation.
- Final model refit on train + validation.

### Monitoring simulation
Live Monitoring replays genuine sensor readings from the dataset. It is a
simulation, not a physical IoT deployment.

### Disclaimer
This is an advisory decision-support tool for an internship deliverable.
Predictions are statistical estimates, not certified industrial safety
signals. Do not shut down machinery solely because of an experimental
model prediction. Operating-range notes reflect statistical percentiles
of the training data, not manufacturer specifications.
        """
    )