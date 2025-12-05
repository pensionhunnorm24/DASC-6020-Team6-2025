"""
train.py

Training orchestration for the Legacy Reimbursement project.

- Loads train/test CSVs
- Builds preprocessing pipeline
- Trains multiple models (linear, ridge, random forest, xgboost fallback)
- Evaluates models (MAE, RMSE)
- Saves pipeline artifacts (preproc + model) to artifacts/
- Robust to sklearn version differences (explicit RMSE computation)
- Includes basic error handling so one failing model doesn't stop the whole run
"""

from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import build_preprocessing
from src.models.linear_models import linear_regression, ridge_regression, lasso_regression
from src.models.tree_models import random_forest, decision_tree
from src.models.advanced_models import svr_model, mlp_model

# Try to import xgboost; if unavailable, fallback to sklearn's GradientBoostingRegressor
try:
    from src.models.boosting_models import xgboost_model
except Exception:
    # boosting_models provides fallback internally; re-import to be safe
    from src.models.boosting_models import xgboost_model

ARTIFACTS = Path(__file__).resolve().parents[1].joinpath("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def train_and_evaluate(train_csv: str | Path, test_csv: str | Path, poly_degree: int = 2):
    """
    Train a set of models and save pipelines.

    Parameters
    ----------
    train_csv : str | Path
        Path to training CSV (expects columns: trip_duration_days, miles_traveled, total_receipts_amount, reimbursement_amount)
    test_csv : str | Path
        Path to test CSV
    poly_degree : int
        Degree for polynomial feature expansion in preprocessing

    Returns
    -------
    dict
        Mapping model_name -> {'mae': float, 'rmse': float, 'status': 'ok'|'error', 'error': str (optional)}
    """
    start_time = time.time()

    # Load data
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Basic validation
    required_cols = {'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount'}
    if not required_cols.issubset(set(train.columns)) or not required_cols.issubset(set(test.columns)):
        raise ValueError(f"Train/test CSVs must contain columns: {required_cols}")

    X_train = train[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y_train = train['reimbursement_amount']
    X_test = test[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y_test = test['reimbursement_amount']

    # Build preprocessing pipeline and transform data
    preproc = build_preprocessing(poly_degree=poly_degree)
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    # Define models to train (at least 4 as per spec)
    models = {
        'linear': linear_regression(),
        'ridge': ridge_regression(alpha=1.0),
        'rf': random_forest(n_estimators=100),
        'xgb': xgboost_model(n_estimators=100),
        'svr': svr_model(C=1.0),
        'mlp': mlp_model(hidden_layer_sizes=(50, 50), max_iter=500)
    }

    results = {}

    for name, model in models.items():
        model_start = time.time()
        try:
            print(f"[TRAIN] Starting training for model: {name}")
            model.fit(X_train_t, y_train)
            preds = model.predict(X_test_t)

            # Metrics
            mae = float(mean_absolute_error(y_test, preds))
            # Compute RMSE explicitly for sklearn compatibility
            mse = float(mean_squared_error(y_test, preds))
            rmse = float(np.sqrt(mse))

            results[name] = {'mae': mae, 'rmse': rmse, 'status': 'ok'}
            # Save pipeline artifact (preproc + model)
            artifact_path = ARTIFACTS / f"{name}_pipeline.joblib"
            joblib.dump({'preproc': preproc, 'model': model}, artifact_path)
            elapsed = time.time() - model_start
            print(f"[TRAIN] Finished {name} (MAE={mae:.4f}, RMSE={rmse:.4f}) saved to {artifact_path} (took {elapsed:.1f}s)")
        except Exception as e:
            # Record error but continue with other models
            results[name] = {'status': 'error', 'error': str(e)}
            print(f"[ERROR] Model {name} failed: {e}", file=sys.stderr)

    total_elapsed = time.time() - start_time
    print(f"[DONE] Training completed in {total_elapsed:.1f}s. Artifacts saved to: {ARTIFACTS}")
    return results


def _print_usage_and_exit():
    print("Usage: python -m src.train <train_csv> <test_csv> [poly_degree]")
    print("Example: python -m src.train data/train.csv data/test.csv 2")
    raise SystemExit(1)


if __name__ == "__main__":
    # CLI entrypoint: python -m src.train <train_csv> <test_csv> [poly_degree]
    if len(sys.argv) < 3:
        _print_usage_and_exit()

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    poly_degree = 2
    if len(sys.argv) >= 4:
        try:
            poly_degree = int(sys.argv[3])
        except ValueError:
            print("poly_degree must be an integer; using default 2")

    try:
        res = train_and_evaluate(train_csv, test_csv, poly_degree=poly_degree)
        # Print summary
        print("\nSummary:")
        for name, info in res.items():
            if info.get('status') == 'ok':
                print(f" - {name}: MAE={info['mae']:.4f}, RMSE={info['rmse']:.4f}")
            else:
                print(f" - {name}: ERROR -> {info.get('error')}")
    except Exception as exc:
        print(f"[FATAL] Training failed: {exc}", file=sys.stderr)
        raise