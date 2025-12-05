# DASC-6020-Team6-2025
This is having all the documentation, code repository associated to DASC 6020 for Team6 

# Legacy Reimbursement - ML Project

## Overview
Reverse-engineer a legacy travel reimbursement system using historical data. This repository contains:
- EDA notebooks
- Feature engineering pipeline
- Multiple model implementations (linear, tree, boosting, advanced)
- Evaluation and saved screenshots for each model
- Production CLI that accepts exactly three parameters and outputs a single reimbursement amount

## Repository structure

├─ data/ │  ├─ public_cases.json            # original 1000 examples (place here) │  ├─ train.csv                    # generated 750-sample CSV │  └─ test.csv                     # generated 250-sample CSV ├─ artifacts/                      # saved model pipelines (joblib) ├─ notebooks/ │  ├─ 01_EDA.ipynb │  └─ 02_Modeling.ipynb ├─ reports/ │  └─ figures/                     # generated screenshots saved here ├─ src/ │  ├─ init.py │  ├─ data_loader.py │  ├─ features.py │  ├─ models/ │  │  ├─ linear_models.py │  │  ├─ tree_models.py │  │  ├─ boosting_models.py │  │  ├─ advanced_models.py │  │  └─ ensemble.py │  ├─ train.py │  ├─ evaluate.py │  └─ predict.py ├─ tests/ │  ├─ test_pipeline.py │  └─ test_predict_cli.py ├─ requirements.txt └─ README.md



## Setup

1. Create and activate a Python 3.9+ virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```
   
 
 
## Data 

- Place public_cases.json into data/ (this file is not included in the repo). The JSON must contain records with keys:
- trip_duration_days (int)
- miles_traveled (int)
- total_receipts_amount (float)
- reimbursement_amount (float

##Prepare datasets

python -m src.data_loader data/public_cases.json
# This creates data/train.csv and data/test.csv

##Train models and save artifacts
python -m src.train data/train.csv data/test.csv


##Generate evaluation figures (screenshots)Run the snippet in notebooks/02_Modeling.ipynb or execute to generate PNGs in reports/figures/.
## Execute EDA
python scripts/run_nbconvert_safe.py notebooks/01_EDA.ipynb notebooks/executed/01_EDA_executed.ipynb --timeout 600

## Execute Modeling (this will train models and save figures)
python scripts/run_nbconvert_safe.py notebooks/02_Modeling.ipynb notebooks/executed/02_Modeling_executed.ipynb --timeout 3600

##then open and execute notebooks/01_EDA.ipynb and notebooks/02_Modeling.ipynb to generate figures
jupyter notebook


##Predict (production CLI)

python src/predict.py 3 120 245.50
# Output: 397.78

##or execute as
python -m src.train data/train.csv data/test.csv



##Tests
## Run unit tests:
pytest -q



