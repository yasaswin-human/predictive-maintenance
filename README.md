# AI-Powered Predictive Maintenance for Industrial Equipment

> A machine-failure classification project using the UCI AI4I 2020 Predictive Maintenance Dataset. It combines a leakage-conscious machine-learning pipeline, a seven-page Streamlit dashboard, simulated sensor monitoring, and SQLite-backed maintenance alerts.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

> **Project status:** The implementation and experimental results must be verified locally before describing this project as fully tested or completed. Evaluation metrics are intentionally left pending until the actual training pipeline has run successfully.

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives and Scope](#objectives-and-scope)
3. [Key Features](#key-features)
4. [Dataset](#dataset)
5. [Machine Learning Methodology](#machine-learning-methodology)
6. [Evaluation Results](#evaluation-results)
7. [System Architecture](#system-architecture)
8. [Project Structure](#project-structure)
9. [Technology Stack](#technology-stack)
10. [Installation](#installation)
11. [Running the Project](#running-the-project)
12. [Dashboard Pages](#dashboard-pages)
13. [Testing](#testing)
14. [Generated Artifacts](#generated-artifacts)
15. [Limitations and Safety Notice](#limitations-and-safety-notice)
16. [Troubleshooting](#troubleshooting)
17. [Future Enhancements](#future-enhancements)
18. [License](#license)
19. [References](#references)

---

## 1. Overview

Unexpected industrial equipment failures can cause production interruptions, increased maintenance costs, and operational risks.

Traditional maintenance approaches generally follow two strategies:

- **Reactive maintenance:** Equipment is repaired after a failure occurs.
- **Preventive maintenance:** Equipment is serviced according to predetermined schedules.

Predictive maintenance takes a different approach. It analyzes equipment operating data to identify potential failure risk and help maintenance personnel make informed maintenance decisions.

This project develops an AI-powered predictive maintenance system that analyzes industrial sensor measurements and predicts potential machine failures using machine-learning techniques.

### Project workflow

1. Download the official UCI AI4I 2020 dataset or use a previously downloaded copy.
2. Validate the dataset structure and sensor measurements.
3. Preprocess the data and prevent target leakage.
4. Train and compare multiple machine-learning classifiers.
5. Select a model and optimize its decision threshold using validation data.
6. Evaluate the selected model on an untouched test dataset.
7. Save the trained model and evaluation artifacts.
8. Perform predictions through a reusable inference engine.
9. Demonstrate equipment monitoring using simulated sensor readings.
10. Generate maintenance alerts and store their history using SQLite.
11. Visualize equipment information and prediction results through a Streamlit dashboard.

The system performs binary classification of machine-failure risk using the current operating measurements. It does not predict the exact date or time of a future failure.

---

## 2. Objectives and Scope

### Objectives

- Develop a machine-learning model for predicting industrial equipment failures.
- Collect, validate, and preprocess equipment sensor measurements.
- Implement equipment-health monitoring using sensor data.
- Generate alerts when the model identifies elevated failure risk.
- Provide maintenance recommendations based on prediction results.
- Develop an interactive dashboard for equipment-health visualization.
- Evaluate the model using appropriate classification metrics.

### Scope

The current project focuses on a software-based predictive maintenance demonstration using a publicly available synthetic industrial dataset.

It covers:

- Dataset acquisition and preprocessing.
- Machine-learning model development.
- Failure-risk prediction.
- Simulated equipment monitoring.
- Maintenance alert generation.
- Persistent alert management.
- Interactive dashboard visualization.
- Technical documentation and evaluation.

Physical industrial sensors and factory control systems are not connected in the current implementation.

---

## 3. Key Features

### Machine-learning failure prediction
Uses multiple classification algorithms to estimate the probability of machine failure from sensor measurements.

### Data preprocessing
Validates the dataset, handles missing values, checks duplicates, and prepares numerical and categorical features.

### Data leakage prevention
Excludes identifiers, target information, and failure-mode columns from model inputs.

### Model comparison
Evaluates four classifiers and compares their validation performance.

### Decision threshold optimization
Selects a classification threshold using validation data and the F2 score.

### Model evaluation
Evaluates the final selected model on an untouched test dataset.

### Prediction engine
Provides reusable single-record and batch prediction functionality.

### Simulated equipment monitoring
Replays sensor readings from the AI4I dataset and generates predictions for each reading.

### Maintenance alert system
Generates advisory alerts when predicted failure probability crosses the selected threshold.

### SQLite alert storage
Maintains alert history, acknowledgement status, and CSV export functionality.

### Interactive dashboard
Provides seven dashboard sections for equipment monitoring, predictions, model performance, and maintenance history.

### Documentation and testing
Includes project documentation, installation instructions, and an automated pytest test suite.

---

## 4. Dataset

**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset

**Repository:** UCI Machine Learning Repository

**Dataset ID:** 601

**Task:** Binary classification — Machine failure prediction.

**Dataset link:**

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### Dataset characteristics

- Approximately 10,000 observations.
- Industrial milling-machine operating measurements.
- Numerical and categorical features.
- Binary machine-failure target.
- Five associated failure-mode indicators.

The dataset represents simulated industrial operating conditions. It is a synthetic benchmark dataset, not real measurements collected from an operational factory.

### Model input features

The model uses six input columns: one categorical feature and five numerical sensor measurements.

| Feature | Type | Description |
|---|---|---|
| `Type` | Categorical | Machine/product quality variant: L, M, or H |
| `Air temperature [K]` | Numerical | Air temperature in Kelvin |
| `Process temperature [K]` | Numerical | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Numerical | Rotational speed in RPM |
| `Torque [Nm]` | Numerical | Torque in Newton-metres |
| `Tool wear [min]` | Numerical | Accumulated tool wear in minutes |

### Prediction target

`Machine failure`

The model predicts whether a machine failure is associated with the supplied operating measurements.

### Excluded columns

The following columns must never be used as model inputs:

| Excluded column | Reason |
|---|---|
| `UDI` | Record identifier |
| `Product ID` | Product identifier |
| `Machine failure` | Prediction target |
| `TWF` | Tool wear failure indicator |
| `HDF` | Heat dissipation failure indicator |
| `PWF` | Power failure indicator |
| `OSF` | Overstrain failure indicator |
| `RNF` | Random failure indicator |

The failure-mode indicators are excluded because they contain information associated with the target and could create data leakage.

The model uses an explicit feature whitelist to select its input columns.

### Data preprocessing

The preprocessing pipeline performs:

1. Dataset schema validation.
2. Missing-value inspection.
3. Duplicate-record detection.
4. Target distribution analysis.
5. Numerical feature analysis.
6. Categorical feature analysis.
7. Feature selection.
8. Numerical and categorical transformation.

Preprocessing is implemented using scikit-learn pipelines and is fitted on the training data only.

---

## 5. Machine Learning Methodology

### Dataset splitting

The dataset is divided into three stratified partitions.

| Dataset partition | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Train machine-learning models |
| Validation | 15% | Select models and optimize thresholds |
| Testing | 15% | Final model evaluation |

Random seed: `42`

Stratification helps preserve the failure-class distribution across the partitions.

### Data preprocessing

The model uses a `ColumnTransformer` and `Pipeline`.

**Numerical preprocessing:**

- SimpleImputer using median strategy.
- StandardScaler.

**Categorical preprocessing:**

- SimpleImputer using most-frequent strategy.
- OneHotEncoder with `handle_unknown="ignore"`.

The preprocessing transformations are fitted within the model pipeline, preventing information from the validation and test partitions from leaking into training.

### Machine-learning algorithms

The project compares four classification algorithms.

| Model | Purpose |
|---|---|
| DummyClassifier | Baseline comparison |
| Logistic Regression | Linear classification model |
| Random Forest Classifier | Ensemble learning |
| Extra Trees Classifier | Randomized tree-ensemble learning |

Class weighting is applied to the applicable non-dummy classifiers to address class imbalance.

### Threshold optimization

A default decision threshold of 0.5 is not automatically assumed to be optimal.

The system selects a threshold using validation predictions by maximizing the F2 score.

The F2 score gives recall greater importance than precision.

\[
F_2 = \frac{5PR}{4P+R}
\]

Where:

- P = Precision
- R = Recall

The objective is to reduce missed failure predictions while considering the number of false-positive alerts.

The selected model is refitted using the combined training and validation partitions. The selected validation threshold is kept fixed before evaluation on the final test partition.

Because refitting can change probability estimates, the effect on threshold behaviour and probability calibration should be considered when interpreting results.

### Model evaluation metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- Average Precision
- Confusion Matrix

Accuracy alone is not sufficient for evaluating this imbalanced classification problem.

Particular attention is given to the failure class, including precision, recall, F1-score, F2-score, and the confusion matrix.

---

## 6. Evaluation Results

The following values must be populated using actual output from the training pipeline.

**Do not insert estimated or fabricated performance results.**

| Test-set metric | Result |
|---|---|
| Precision | Pending actual training run |
| Recall | Pending actual training run |
| F1-score | Pending actual training run |
| F2-score | Pending actual training run |
| ROC-AUC | Pending actual training run |
| Average Precision | Pending actual training run |
| Decision threshold | Pending actual training run |

### Model comparison

Candidate model validation results are saved in:

`models/metrics/model_comparison.csv`

### Final test results

Final held-out test evaluation is saved in:

`models/metrics/test_metrics.json`

These metrics should be used to complete the final internship report.

---

## 7. System Architecture

The project follows a modular architecture that separates dataset processing, machine learning, prediction, monitoring, alert management, and visualization.

### Architecture diagram

```text
               UCI AI4I 2020 Dataset
                         |
                         v
                Dataset Acquisition
                         |
                         v
                Data Validation
                         |
                         v
               Data Preprocessing
                         |
                         v
                  Exploratory
                 Data Analysis
                         |
                         v
             Stratified Data Splitting
                         |
                         v
               Model Training and
                    Comparison
                         |
                         v
              Validation Threshold
                  Optimization
                         |
                         v
                Final Model Saving
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         Manual       Batch CSV   Simulated
         Prediction   Prediction  Monitoring
             |           |           |
             +-----------+-----------+
                         |
                         v
                 Prediction Engine
                         |
                         v
                   Alert Engine
                         |
                         v
                   SQLite Storage
                         |
                         v
                Streamlit Dashboard
```

### Major components

| Component | Responsibility |
|---|---|
| Data acquisition | Downloads the official dataset |
| Data validation | Checks schema and target validity |
| Preprocessing | Cleans and transforms the measurements |
| Model training | Trains and compares classifiers |
| Model evaluation | Measures classification performance |
| Prediction engine | Generates failure predictions |
| Monitoring engine | Replays sensor readings |
| Alert engine | Generates maintenance alerts |
| SQLite storage | Stores alert history |
| Streamlit dashboard | Visualizes the system |

---

## 8. Project Structure

```text
predictive-maintenance/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── LICENSE
├── .gitignore
├── .env.example
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   └── alerts/
│
├── models/
│   ├── artifacts/
│   └── metrics/
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── validate_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── prediction/
│   ├── monitoring/
│   ├── alerts/
│   └── storage/
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   ├── test_alerts.py
│   └── test_monitoring.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_CARD.md
│   ├── MODEL_CARD.md
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   ├── METHODOLOGY.md
│   └── screenshots/
│
└── logs/
```

Each module is designed to have a separate responsibility.

The dashboard should orchestrate the user interface and call the appropriate backend services rather than contain duplicate model-training or prediction logic.

The exact directory contents should be verified against the actual repository.

---

## 9. Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python 3.13 |
| Data manipulation | Pandas |
| Numerical computation | NumPy |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Dashboard | Streamlit |
| Interactive visualization | Plotly |
| Data visualization | Matplotlib, Seaborn |
| Dataset download | Requests |
| Alert database | SQLite |
| Testing | Pytest |
| Development environment | VS Code |

The project is designed for CPU execution. A dedicated GPU is not required.

---

## 10. Installation

### Requirements

- Python 3.13 (recommended target version).
- Windows 11 with PowerShell.
- VS Code or another Python-compatible IDE.
- Internet connection for the initial dataset download.
- Approximately 1 GB of free disk space recommended.
- No GPU required.

Compatibility with other operating systems and Python versions should be verified separately.

### Clone the repository

```powershell
git clone <your-repository-url> predictive-maintenance

cd predictive-maintenance
```

### Create a virtual environment

```powershell
py -3.13 -m venv .venv
```

### Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## 11. Running the Project

Run all commands from the project root directory.

### Step 1 — Download the dataset

```powershell
python scripts\download_data.py
```

This script downloads and extracts the official dataset or uses a previously downloaded copy.

### Step 2 — Validate the dataset

```powershell
python scripts\validate_data.py
```

This checks the required columns, dataset structure, target values, and data validity.

A typical original AI4I dataset contains approximately:

- 10,000 records.
- 14 columns.
- An imbalanced machine-failure target.

Use the actual validation output as the authoritative result.

### Step 3 — Preprocess the dataset

```powershell
python scripts\preprocess_data.py
```

This generates the processed dataset and exploratory data analysis artifacts.

### Step 4 — Train the models

```powershell
python scripts\train_model.py
```

This executes the model-training and comparison workflow, selects the final candidate, optimizes its validation threshold, and generates the model artifacts and final test metrics.

### Step 5 — Evaluate the model

```powershell
python scripts\evaluate_model.py
```

This displays the saved final evaluation results.

### Step 6 — Run automated tests

```powershell
pytest -q
```

### Step 7 — Launch the dashboard

```powershell
streamlit run app.py
```

Open the URL displayed in the terminal:

```text
http://localhost:8501
```

**Important:** Train the model before launching dashboard sections that require saved model artifacts.

### If the UCI download fails

1. Open the official UCI dataset page.
2. Download the AI4I 2020 dataset archive.
3. Extract the archive.
4. Locate `ai4i2020.csv`.
5. Place it inside:

```text
data/raw/ai4i2020.csv
```

6. Run the validation and preprocessing commands again.

The system must not silently substitute randomly generated data and report it as official dataset results.

---

## 12. Dashboard Pages

The application contains seven dashboard sections.

| Page | Description |
|---|---|
| Overview | Dataset information, failure distribution, machine-type distribution, and available evaluation metrics |
| Live Monitoring | Simulated sensor stream, monitoring controls, prediction probability, and alerts |
| Manual Prediction | Enter sensor measurements and receive failure prediction results |
| Batch Analysis | Upload a CSV containing multiple sensor readings and download predictions |
| Model Performance | Display test metrics, confusion matrix, candidate comparison, ROC curve, and precision-recall curve |
| Alerts & Maintenance History | View, filter, acknowledge, and export maintenance alerts |
| About & Dataset | Project objectives, dataset details, methodology, and limitations |

### Live Monitoring

The Live Monitoring feature uses historical sensor measurements from the AI4I dataset.

Available controls are intended to include:

- Start monitoring.
- Pause monitoring.
- Stop monitoring.
- Step through readings.
- Reset the stream.
- Adjust playback speed.
- Inspect current sensor measurements.
- View prediction trends.
- Review generated alerts.

**This is simulated monitoring, not a physical IoT deployment.**

The project does not currently connect to PLC, SCADA, OPC-UA, or MQTT hardware.

---

## 13. Testing

Testing is performed using pytest.

Run:

```powershell
pytest -q
```

### Test coverage

| Test file | Testing purpose |
|---|---|
| `test_data.py` | Dataset validation, missing files, empty files, and schema errors |
| `test_preprocessing.py` | Feature selection, leakage prevention, preprocessing, and unseen categories |
| `test_prediction.py` | Prediction input validation, single prediction, and batch prediction |
| `test_alerts.py` | Alert generation and duplicate suppression |
| `test_monitoring.py` | Sensor-stream iteration, reset, and missing-column handling |

Some prediction tests may be skipped if the trained model artifacts are unavailable.

A skipped test must not be reported as a passed test.

### Integration testing

After obtaining the dataset, run:

```powershell
python scripts\download_data.py

python scripts\validate_data.py

python scripts\preprocess_data.py

python scripts\train_model.py

python scripts\evaluate_model.py

pytest -q

streamlit run app.py
```

Record the actual test results and resolve critical errors before declaring the system operational.

---

## 14. Generated Artifacts

The following files are generated during the workflow, depending on successful execution.

| Artifact | Purpose |
|---|---|
| `data/raw/ai4i2020.csv` | Original dataset |
| `data/processed/processed.csv` | Processed dataset |
| `models/artifacts/final_model.joblib` | Trained preprocessing and classification pipeline |
| `models/artifacts/threshold.json` | Selected model threshold |
| `models/artifacts/model_metadata.json` | Model and feature metadata |
| `models/metrics/model_comparison.csv` | Validation model comparison |
| `models/metrics/test_metrics.json` | Final test evaluation |
| `models/metrics/eda_stats.json` | Exploratory statistics |
| `models/metrics/*.png` | EDA and evaluation charts |
| `data/alerts/alerts.sqlite` | Persistent alert database |

Large raw datasets, runtime databases, virtual environments, and temporary files should generally be excluded from Git commits.

The scripts should be used to recreate generated outputs where appropriate.

---

## 15. Limitations and Safety Notice

This project is an experimental advisory decision-support application, not a certified industrial safety controller.

### Synthetic dataset

The AI4I 2020 dataset contains simulated industrial data. Model performance on this dataset does not prove equivalent performance on real industrial machinery.

### No exact failure-time prediction

The model predicts the failure class and estimated probability associated with current measurements. It does not estimate the exact time or date when failure will happen.

### No Remaining Useful Life estimation

Remaining Useful Life (RUL) prediction is outside the current project's scope.

### No physical IoT connection

The monitoring feature replays historical dataset readings. Actual industrial sensor integration is not currently implemented.

### No failure-mode diagnosis

The model performs binary classification and does not determine which specific failure mode will occur.

### Statistical operating ranges

Percentile-based operating-range notes are statistical observations. They are not manufacturer-approved operating limits or certified safety thresholds.

### Advisory maintenance recommendations

Predictions and alerts are intended to support maintenance decisions. They must be checked by qualified personnel.

**Do not shut down or control machinery based solely on the output of this experimental model.**

Real-world deployment would require representative industrial data, external validation, appropriate engineering review, and safety assessment.

---

## 16. Troubleshooting

| Problem | Suggested solution |
|---|---|
| Missing trained model | Run `python scripts\train_model.py` before using inference features. |
| Dataset download failure | Download the dataset manually from UCI and place it in `data/raw/`. |
| Streamlit command not found | Activate the virtual environment and install the requirements. |
| `ModuleNotFoundError` for `src` or `config` | Run commands from the project root and check package files. |
| Prediction tests skipped | Confirm that model artifacts have been generated. |
| Dashboard metrics unavailable | Run training and evaluation and verify the metrics directory. |
| Invalid batch CSV | Check that all six required model input columns are present with valid values. |
| Dependency installation errors | Check the active Python version and package error messages. |

When investigating errors, preserve the full traceback and record the Python version and command used.

---

## 17. Future Enhancements

The following are possible future extensions and are not part of the current demonstrated functionality.

- **Physical IoT integration:** Integrate appropriate sensor sources using MQTT or OPC-UA.
- **Remaining Useful Life prediction:** Develop a separate RUL estimation model using an appropriate target and dataset.
- **Drift monitoring:** Detect changes in incoming sensor distributions.
- **Advanced alert routing:** Add email or messaging notifications.
- **Model version management:** Track model versions and support rollback.
- **Explainable AI:** Investigate SHAP-based feature importance and individual prediction explanations.
- **External validation:** Test the model using representative industrial operating data.
- **Operational monitoring:** Evaluate probability calibration, false-alarm rates, and model performance over time.

---

## 18. License

This project is distributed under the MIT License.

See the [`LICENSE`](LICENSE) file for the complete license terms.

The UCI dataset retains its own attribution and licensing conditions, which must be followed separately.

---

## 19. References

1. UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset.  
   https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

2. Dataset DOI:  
   https://doi.org/10.24432/C5HS5C

3. S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, 2020 International Conference on Artificial Intelligence for Industries (AI4I), 2020.

4. Scikit-learn Documentation:  
   https://scikit-learn.org/stable/user_guide.html

5. Streamlit Documentation:  
   https://docs.streamlit.io

6. Plotly Python Documentation:  
   https://plotly.com/python/

---

## Project Information

**Project Title:** AI-Powered Predictive Maintenance for Industrial Equipment

**Domain:** Artificial Intelligence / Machine Learning / Industrial IoT

**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset

**Primary Language:** Python

**Dashboard Framework:** Streamlit

**Machine-learning Framework:** Scikit-learn

**Project Type:** Internship Project

**Development Environment:** Windows 11 / VS Code

### Final verification checklist

Before submitting this project as a completed internship deliverable:

- [ ] Official dataset downloaded and validated.
- [ ] Preprocessing and EDA completed.
- [ ] Four candidate models trained and compared.
- [ ] Real test metrics generated and inserted above.
- [ ] Trained model successfully loaded for inference.
- [ ] Manual prediction tested.
- [ ] Batch prediction tested.
- [ ] Simulated monitoring tested.
- [ ] Alert generation and persistence tested.
- [ ] All seven dashboard pages checked.
- [ ] Automated tests executed and results recorded.
- [ ] Genuine dashboard screenshots added.
- [ ] Complete project report finalized.
- [ ] Installation instructions verified on the target machine.

**This repository should describe only functionality that is present and verified in the actual implementation.**
