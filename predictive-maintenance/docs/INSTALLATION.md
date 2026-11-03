# Installation (Windows 11)

1. Install Python 3.13 from python.org (check "Add to PATH").
2. In PowerShell from the project root:

       python -m venv .venv
       .\.venv\Scripts\Activate.ps1
       pip install --upgrade pip
       pip install -r requirements.txt
       python scripts\download_data.py
       python scripts\validate_data.py
       python scripts\preprocess_data.py
       python scripts\train_model.py
       python scripts\evaluate_model.py
       pytest -q
       streamlit run app.py

## Troubleshooting
- UCI download blocked: follow the printed manual procedure and place
  `ai4i2020.csv` in `data\raw\`.
- `streamlit` command not found: activate the venv first.
- `pytest` skips `test_prediction.py`: expected until `scripts\train_model.py` runs.