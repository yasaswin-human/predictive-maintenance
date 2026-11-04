# User Guide — AI-Powered Predictive Maintenance for Industrial Equipment

This guide explains how to install, configure, train, test, and operate the AI-Powered Predictive Maintenance system on Windows 11.

The project uses the UCI AI4I 2020 Predictive Maintenance Dataset to develop a machine-failure classification model. It provides manual predictions, batch predictions, simulated sensor monitoring, maintenance alerts, and a Streamlit dashboard.

> **Important:** Live Monitoring currently replays historical sensor readings. It is a simulation, not a connection to physical industrial equipment. Predictions are advisory and are not certified industrial safety signals.

---

## 1. Understanding the two data workflows

There are two different ways to provide data to the system.

| Workflow | Required data | Purpose |
|---|---|---|
| Model training | Labelled data containing `Machine failure` | Trains and evaluates the machine-learning model |
| Model prediction | Six machine sensor input features | Uses the trained model to predict failure probability |

**Important:** Uploading a CSV through the Batch Analysis dashboard does not train or update the model. It only generates predictions using the existing trained model.

To train or retrain the model, follow Section 5.

---

## 2. System requirements

- Windows 11
- Python 3.13 (recommended)
- PowerShell
- VS Code or another Python IDE
- Internet connection for the initial dataset download
- Approximately 1 GB of free disk space recommended
- No GPU required

Run all commands from the project root directory, which contains `app.py`, `requirements.txt`, and the `scripts` folder.

---

## 3. Installation and environment setup

### Step 1: Clone the repository

Replace `<your-repository-url>` with your actual GitHub repository URL.

```powershell
git clone <your-repository-url> predictive-maintenance

cd predictive-maintenance
```

Alternatively, extract the project ZIP and open the extracted folder in VS Code.

### Step 2: Check Python

```powershell
python --version
```

The recommended environment is Python 3.13.

Alternatively:

```powershell
py -3.13 --version
```

### Step 3: Create a virtual environment

```powershell
py -3.13 -m venv .venv
```

### Step 4: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, allow activation for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

Make sure the virtual environment is activated before continuing.

---

## 4. Dataset acquisition and preprocessing

### Dataset information

**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset

**Dataset ID:** 601

Official source:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains approximately 10,000 observations of simulated industrial operating conditions.

### Step 1: Download the dataset

```powershell
python scripts\download_data.py
```

Expected raw dataset location:

```text
data/raw/ai4i2020.csv
```

The download script should use the official dataset or a cached copy.

### Step 2: Validate the dataset

```powershell
python scripts\validate_data.py
```

The validation process checks the required schema, target values, and data quality.

The prediction target is:

```text
Machine failure
```

### Step 3: Preprocess the dataset

```powershell
python scripts\preprocess_data.py
```

This step performs the configured preprocessing and exploratory analysis.

Expected processed dataset:

```text
data/processed/processed.csv
```

EDA statistics and visualizations are saved under:

```text
models/metrics/
```

### If the UCI download fails

1. Open the official UCI dataset page.
2. Download the dataset archive manually.
3. Extract the archive.
4. Locate `ai4i2020.csv`.
5. Place the file inside `data/raw/`.
6. Continue with dataset validation.

Do not replace the official dataset with randomly generated data and describe it as official AI4I data.

---

## 5. Model training and retraining

### Step 1: Start model training

After downloading, validating, and preprocessing the dataset, execute:

```powershell
python scripts\train_model.py
```

The training pipeline compares four machine-learning classifiers:

1. DummyClassifier
2. LogisticRegression
3. RandomForestClassifier
4. ExtraTreesClassifier

### Training workflow

The described implementation follows these steps:

1. Creates stratified training, validation, and test partitions.
2. Fits preprocessing and classification pipelines using training data.
3. Evaluates candidate models on validation data.
4. Selects a decision threshold by maximizing the validation F2 score.
5. Selects the candidate model using validation performance.
6. Refits the selected pipeline on training and validation data.
7. Evaluates the final model on the untouched test partition.
8. Saves the selected model and evaluation artifacts.

The intended split is:

| Partition | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Random seed: `42`

### Important feature selection rules

The six model input features are:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The following columns must not be used as model inputs:

- `UDI`
- `Product ID`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The failure-mode columns are excluded to prevent target leakage.

### Step 2: Inspect evaluation results

```powershell
python scripts\evaluate_model.py
```

Expected artifacts include:

```text
models/
├── artifacts/
│   ├── final_model.joblib
│   ├── threshold.json
│   └── model_metadata.json
│
└── metrics/
    ├── model_comparison.csv
    ├── test_metrics.json
    └── eda_stats.json
```

The final evaluation should include:

- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- Average Precision
- Confusion Matrix
- Selected decision threshold

Use the actual generated metrics in your project report. Do not manually invent or alter results.

### Retraining with new labelled data

The current training workflow is designed around the AI4I dataset schema.

For another labelled dataset:

1. Make sure you have permission to use the data.
2. Include all six required model input columns.
3. Include the binary `Machine failure` target.
4. Ensure the feature definitions and units are compatible.
5. Back up the existing data and model artifacts.
6. Update the dataset path or acquisition configuration as appropriate.
7. Run validation.
8. Run preprocessing.
9. Run model training.
10. Evaluate the newly trained model.

Do not use unlabelled sensor readings as supervised training examples. The failure classifier needs known labels.

**Training data and prediction data are not interchangeable.**

---

## 6. Automated testing

Run:

```powershell
pytest -q
```

The test suite is intended to cover:

| Test file | Purpose |
|---|---|
| `test_data.py` | Dataset validation |
| `test_preprocessing.py` | Feature selection and leakage prevention |
| `test_prediction.py` | Single and batch prediction |
| `test_alerts.py` | Alert generation and duplicate suppression |
| `test_monitoring.py` | Sensor-stream iteration and reset |

Some prediction tests may be skipped if trained model artifacts are unavailable.

A skipped test is not a passed test. Train the model and rerun the tests to exercise the inference functionality.

Record the actual terminal test results.

---

## 7. Launching the Streamlit dashboard

After training the model and generating the required artifacts:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal. Normally:

http://localhost:8501

Keep the terminal running while using the dashboard.

Press `Ctrl+C` in the terminal to stop the application.

---

## 8. Using the dashboard

The application contains seven sections.

### Page 1: Overview

Displays available dataset information, machine-failure distribution, machine-type distribution, and saved model evaluation results.

### Page 2: Live Monitoring

Demonstrates simulated machine monitoring by replaying historical sensor readings.

Described controls include:

- Start
- Pause
- Stop
- Step
- Reset
- Playback speed adjustment

The application passes readings through the prediction engine and can generate alerts when the configured probability threshold is crossed.

**This is simulated monitoring, not physical IoT monitoring.**

### Page 3: Manual Prediction

Enter the current operating measurements of a machine.

The six required inputs are:

| Input | Example |
|---|---:|
| Type | `M` |
| Air temperature [K] | `300.0` |
| Process temperature [K] | `310.0` |
| Rotational speed [rpm] | `1500` |
| Torque [Nm] | `40.0` |
| Tool wear [min] | `100` |

These are illustrative input values, not guaranteed operating values for a particular machine.

After submission, the inference engine should return the estimated failure probability, predicted class, risk classification, operating-range observations, and maintenance recommendation, according to the implemented interface.

Use actual measurements and correct units when testing.

### Page 4: Batch Analysis

This page allows multiple machine readings to be processed using a CSV file.

The CSV must contain exactly these six required model input columns:

```csv
Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
M,300.0,310.0,1500,40.0,100
L,301.0,311.0,1400,45.0,120
H,299.5,309.5,1600,35.0,80
```

Save the example as:

```text
machine_readings.csv
```

Then:

1. Open Batch Analysis.
2. Upload the CSV file.
3. Allow the application to validate its columns and measurements.
4. Review the generated predictions.
5. Inspect the risk distribution.
6. Download the prediction results if the export control is available.

**Do not include `Machine failure` or failure-mode columns in prediction CSV uploads.**

The model must predict the target rather than receive it as an input.

### Page 5: Model Performance

Displays available evaluation metrics, confusion matrix, candidate model comparison, ROC curve, and precision-recall curve.

Check the displayed dataset partition for every metric. Final test metrics should represent the untouched test partition.

### Page 6: Alerts & Maintenance History

Displays the available alert history and supports filtering, acknowledgement, and CSV export where implemented.

SQLite is used for alert persistence.

The configured cooldown mechanism is intended to reduce repeated alerts for the same equipment and risk level.

### Page 7: About & Dataset

Displays the project objectives, dataset source, methodology, and limitations.

---

## 9. Understanding predictions and alerts

The inference engine produces a failure probability and applies the saved decision threshold to produce the binary prediction.

The described application uses three risk categories:

- **LOW:** Probability below the configured medium-risk boundary.
- **MEDIUM:** Probability at or above half the selected threshold but below the threshold.
- **HIGH:** Probability at or above the selected threshold.

These categories are application-defined. They are not manufacturer-certified equipment severity ratings.

Operating-range notes based on dataset percentiles are statistical observations, not certified equipment safety limits.

Maintenance recommendations are advisory. Qualified maintenance personnel should verify sensor readings and operating conditions before taking action.

---

## 10. Generated project outputs

| Location | Purpose |
|---|---|
| `data/raw/` | Original downloaded dataset |
| `data/processed/` | Processed dataset |
| `models/artifacts/` | Trained model, threshold, and metadata |
| `models/metrics/` | Evaluation metrics, EDA statistics, and charts |
| `data/alerts/` | SQLite alert history |
| `docs/` | Project documentation |
| `logs/` | Runtime logs, depending on logging configuration |

Some generated data, model artifacts, and runtime databases are excluded from Git through `.gitignore`.

After cloning the repository, download the data and train the model to recreate the required artifacts.

---

## 11. Troubleshooting

| Problem | Suggested solution |
|---|---|
| Python command not recognized | Install Python and configure PATH, or use `py -3.13`. |
| Virtual environment activation blocked | Use the process-scoped PowerShell execution-policy command in Section 3. |
| Package installation error | Verify the Python version and inspect the dependency error. |
| UCI download fails | Download the dataset manually and put the CSV in `data/raw/`. |
| Dataset validation fails | Check the filename, column names, target values, and CSV format. |
| Processed CSV missing | Run `python scripts\preprocess_data.py`. |
| Model artifact missing | Run the complete dataset and training workflow. |
| Streamlit command not recognized | Activate the virtual environment and install requirements. |
| `ModuleNotFoundError` for `src` or `config` | Run commands from the project root directory. |
| Batch CSV rejected | Verify all six feature-column names and input values. |
| Prediction tests skipped | Check whether the trained model artifacts exist. |
| Evaluation metrics unavailable | Train the model and check `models/metrics/test_metrics.json`. |

When reporting a problem, include the command used, complete traceback, Python version, and whether the virtual environment was activated.

---

## 12. Recommended first-time execution

For a fresh installation, follow this sequence:

```powershell
# Create the environment
py -3.13 -m venv .venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download and prepare the dataset
python scripts\download_data.py
python scripts\validate_data.py
python scripts\preprocess_data.py

# Train and evaluate
python scripts\train_model.py
python scripts\evaluate_model.py

# Run automated tests
pytest -q

# Start the dashboard
streamlit run app.py
```

Once the model has been trained successfully, you generally do not need to retrain it every time the dashboard starts. The dashboard should load the saved model artifacts.

---

## 13. Safety, dataset attribution, and licensing

This project is an experimental advisory decision-support application.

- The AI4I 2020 dataset contains synthetic benchmark data.
- The current monitoring workflow replays historical readings.
- The model does not predict the exact time or date of machine failure.
- The project does not implement Remaining Useful Life estimation.
- The system is not a certified industrial safety controller.
- Do not operate, stop, or control machinery based solely on experimental model predictions.
- Real-world deployment requires representative industrial data, external validation, and engineering safety assessment.

### Dataset reference

UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Dataset DOI:

https://doi.org/10.24432/C5HS5C

The project's original source code uses the MIT License. The dataset and third-party dependencies retain their respective licensing conditions.

---

## 14. Quick command reference

| Task | Command |
|---|---|
| Download dataset | `python scripts\download_data.py` |
| Validate dataset | `python scripts\validate_data.py` |
| Preprocess data | `python scripts\preprocess_data.py` |
| Train/retrain models | `python scripts\train_model.py` |
| Evaluate model | `python scripts\evaluate_model.py` |
| Run tests | `pytest -q` |
| Launch dashboard | `streamlit run app.py` |

---

**End of User Guide**
