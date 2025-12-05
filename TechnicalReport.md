Legacy Reimbursement — Technical Report

Overview

Purpose: Predict reimbursement amounts from trip-level features to automate and standardize legacy reimbursement decisions.

Inputs: Raw JSON (data/public_cases.json) containing trip records. Records may be flat or nested (e.g., { "input": {...}, "expected_output": ... }).

Outputs: Trained model pipelines saved to artifacts/, processed CSVs in data/, and evaluation figures in reports/figures/.

Data and Preprocessing

Canonical fields expected

trip_duration_days

miles_traveled

total_receipts_amount

reimbursement_amount

Loader behavior

Accepts a list of records in JSON. Handles both flat records and nested records with input and expected_output.

Attempts to auto-map common synonyms (e.g., days, miles, total_amount, expected_output) to canonical names.

Provides an explicit --map canonical:actual_key CLI option for manual mapping when auto-detection fails.

Coerces numeric types and fills missing values with sensible defaults (0 or 0.0).

Shuffles data and splits into train (first 750 rows) and test (remaining rows).

Files

src/data_loader.py — robust loader and mapping logic.

data/public_cases.json — raw input.

data/train.csv, data/test.csv — processed outputs.

Modeling and Evaluation

Models trained

Linear Regression (baseline)

Ridge Regression

Random Forest

XGBoost (fallback to sklearn booster if XGBoost not installed)

SVR and MLP (additional experiments)

Optional stacking ensemble combining RF and XGBoost

Pipeline structure

Each saved artifact is a dictionary with keys: preproc and model.

Artifacts saved as artifacts/<model>_pipeline.joblib.

Metrics

MAE (mean absolute error)

RMSE computed explicitly as sqrt(mean_squared_error(...)) for sklearn compatibility

Figures generated

Predicted vs True scatter plots

Residual histograms

Feature importance or coefficient plots

Saved to reports/figures/ with filenames like <model>_pred_vs_true.png.

Artifacts, Notebooks, and Tests

Key scripts and notebooks

src/data_loader.py — data preparation

src/train.py — training orchestration and artifact saving

src/predict.py — CLI for single-record prediction

notebooks/01_EDA.ipynb — exploratory data analysis (auto-prepares data)

notebooks/02_Modeling.ipynb — model training and figure generation

tests/test_pipeline.py — pytest checks for artifacts and pipeline structure

scripts/run_nbconvert_safe.py — safe wrapper to execute notebooks with nbconvert

Testing notes

Tests ensure artifacts/ exists and that at least one *_pipeline.joblib artifact loads and contains preproc and model keys.

If unpickling fails with ModuleNotFoundError: No module named 'src', ensure tests run from the project root or add the project root to PYTHONPATH.

Reproduction Steps

Run these commands from the project root with the virtual environment activated.

Install dependencies

pip install -r requirements.txt
# or minimal set
pip install pandas numpy scikit-learn joblib matplotlib seaborn nbconvert nbclient xgboost

Prepare data

python -m src.data_loader data/public_cases.json
# If loader needs explicit mapping, use:
python -m src.data_loader data/public_cases.json 
  --map trip_duration_days:days --map miles_traveled:miles 
  --map total_receipts_amount:receipts --map reimbursement_amount:expected_output

Train models

python -m src.train data/train.csv data/test.csv

Execute notebooks and generate figures

python scripts/run_nbconvert_safe.py notebooks/01_EDA.ipynb notebooks/executed/01_EDA_executed.ipynb --timeout 600
python scripts/run_nbconvert_safe.py notebooks/02_Modeling.ipynb notebooks/executed/02_Modeling_executed.ipynb --timeout 3600

Predict from CLI

python src/predict.py 3 120 245.50
# prints a single numeric reimbursement amount

Run tests

pytest -q

Troubleshooting and Recommendations

Missing CSVs in notebooks

Cause: data/train.csv and data/test.csv not present.

Fix: run python -m src.data_loader data/public_cases.json or use the notebook’s first cell which attempts to create them.

Unpickling ModuleNotFoundError

Cause: artifact references src.* but project root not on sys.path.

Fix: run scripts/tests from project root or add project root to PYTHONPATH:

export PYTHONPATH=$(pwd)
# Windows PowerShell
$env:PYTHONPATH = (Get-Location)

nbconvert FileNotFoundError for output path

Cause: --output path parent directory missing.

Fix: create parent directory or use scripts/run_nbconvert_safe.py which creates it automatically.

Missing Python packages in notebooks

Fix: install missing packages into the same venv used to run Jupyter:

pip install seaborn matplotlib pandas numpy

Testing in CI

Recommendation: add a self-contained test that creates a tiny synthetic dataset, runs the loader and a quick model training, then asserts artifacts were created. This avoids relying on pre-existing artifacts in CI.

Portability

Consider exporting models to ONNX or saving only model parameters for cross-environment portability.

Pin dependency versions in requirements.txt to avoid unpickle/import mismatches.

Contact and Next Steps

Repository root contains all scripts, notebooks, and data.

To improve reproducibility, consider adding a GitHub Actions workflow that installs dependencies, runs the loader, trains a quick model, executes notebooks, and runs tests.

Prepared by: Rajesh Radhakrishnan, Mahaboob Basha Pension Hunnor & Kalaiselvan
Murugavelu Date: 4th December 2025