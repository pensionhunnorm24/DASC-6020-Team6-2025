"""
predict.py

Production CLI that accepts exactly 3 parameters:
    trip_duration_days (int)
    miles_traveled (int)
    total_receipts_amount (float)

Prints a single numeric reimbursement amount rounded to 2 decimals.

This version ensures the project root is on sys.path before loading joblib artifacts,
so unpickling can import project modules referenced by saved pipelines.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure project root is on sys.path so pickled objects referencing src.* can be imported.
# Project root is the parent directory of this file's parent (i.e., repository root).
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT.joinpath("artifacts")


def load_pipeline(preferred: str = "xgb_pipeline.joblib"):
    """
    Load a saved pipeline artifact. If preferred_name not found,
    fallback to first available pipeline. Raises FileNotFoundError or RuntimeError on failure.
    """
    # Look for preferred artifact first
    preferred_path = ARTIFACTS / preferred
    if preferred_path.exists():
        try:
            return joblib.load(preferred_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline '{preferred_path}': {e}") from e

    # Fallback: any pipeline
    files = sorted(ARTIFACTS.glob("*_pipeline.joblib"))
    if not files:
        raise FileNotFoundError(f"No pipeline artifacts found in {ARTIFACTS}. Run training first.")
    # Try loading each until one succeeds
    last_exc = None
    for p in files:
        try:
            return joblib.load(p)
        except Exception as e:
            last_exc = e
            # continue to next file
    # If we reach here, none loaded successfully
    raise RuntimeError(f"Failed to load any pipeline from {ARTIFACTS}. Last error: {last_exc}")


def validate_and_parse_args(args):
    """Validate that exactly 3 args are provided and parse types."""
    if len(args) != 3:
        raise SystemExit("Error: Exactly 3 parameters required: trip_duration_days miles_traveled total_receipts_amount")
    try:
        trip_days = int(args[0])
        miles = int(args[1])
        receipts = float(args[2])
    except ValueError:
        raise SystemExit("Error: Parameter types must be int int float")
    return trip_days, miles, receipts


def main(argv):
    trip_days, miles, receipts = validate_and_parse_args(argv)

    # Ensure artifacts directory exists
    if not ARTIFACTS.exists():
        raise SystemExit(f"Error: artifacts directory not found at {ARTIFACTS}. Run training first.")

    # Load pipeline (this may raise FileNotFoundError or RuntimeError)
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        raise SystemExit(f"Error: {e}")
    except RuntimeError as e:
        raise SystemExit(f"Error loading model pipeline: {e}")

    # Expect pipeline to be a dict with 'preproc' and 'model'
    if not isinstance(pipeline, dict) or 'preproc' not in pipeline or 'model' not in pipeline:
        raise SystemExit("Error: Loaded artifact does not contain expected keys 'preproc' and 'model'.")

    preproc = pipeline['preproc']
    model = pipeline['model']

    # Build single-row DataFrame for preprocessing
    X = pd.DataFrame([{
        'trip_duration_days': trip_days,
        'miles_traveled': miles,
        'total_receipts_amount': receipts
    }])

    try:
        X_t = preproc.transform(X)
    except Exception as e:
        raise SystemExit(f"Error during preprocessing: {e}")

    try:
        pred = model.predict(X_t)[0]
    except Exception as e:
        raise SystemExit(f"Error during model prediction: {e}")

    # Post-processing: ensure numeric, round to 2 decimals
    try:
        pred_rounded = round(float(pred), 2)
    except Exception as e:
        raise SystemExit(f"Error converting prediction to float: {e}")

    # Print exactly one numeric value (no extra text)
    print(f"{pred_rounded:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])