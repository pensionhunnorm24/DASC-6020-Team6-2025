"""
linear_models.py
Wrappers for linear models: LinearRegression, Ridge, Lasso.
Each function returns an sklearn estimator (or Pipeline if needed).
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

def linear_regression():
    """Simple linear regression pipeline (model only)."""
    return LinearRegression()

def ridge_regression(alpha: float = 1.0):
    """Ridge regression with L2 regularization."""
    return Ridge(alpha=alpha, random_state=42)

def lasso_regression(alpha: float = 0.1):
    """Lasso regression with L1 regularization."""
    return Lasso(alpha=alpha, random_state=42, max_iter=5000)