"""
evaluate.py
Evaluation utilities and plotting functions.
Saves figures to reports/figures/ as PNG files (these are the screenshots).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

FIG_DIR = Path(__file__).resolve().parents[1].joinpath("reports", "figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_pred_vs_true(y_true, y_pred, model_name: str):
    """Scatter plot of predicted vs true values."""
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
    plt.xlabel("True reimbursement")
    plt.ylabel("Predicted reimbursement")
    plt.title(f"Pred vs True: {model_name}")
    plt.tight_layout()
    out = FIG_DIR / f"{model_name}_pred_vs_true.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_residuals_hist(y_true, y_pred, model_name: str):
    """Histogram of residuals (true - pred)."""
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30, color='C0', alpha=0.8)
    plt.xlabel("Residual (True - Pred)")
    plt.title(f"Residuals: {model_name} (MAE={mean_absolute_error(y_true,y_pred):.2f})")
    plt.tight_layout()
    out = FIG_DIR / f"{model_name}_residuals.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_feature_importance(importances, feature_names, model_name: str):
    """
    Bar chart of feature importances or coefficients.
    - importances: 1D array-like
    - feature_names: list of names (same length)
    """
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    plt.bar(range(len(importances)), np.array(importances)[idx], color='C1')
    plt.xticks(range(len(importances)), np.array(feature_names)[idx], rotation=45, ha='right')
    plt.title(f"Feature importance: {model_name}")
    plt.tight_layout()
    out = FIG_DIR / f"{model_name}_feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out