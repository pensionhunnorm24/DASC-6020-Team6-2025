"""
features.py
Feature engineering pipeline:
- Derived features: cost_per_mile, cost_per_day
- Polynomial features and scaling
- Exposes build_preprocessing() to return a sklearn Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class DerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that adds domain-derived features:
    - cost_per_mile = total_receipts_amount / miles_traveled
    - cost_per_day = total_receipts_amount / trip_duration_days
    Handles division-by-zero by filling with 0.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Avoid division by zero; replace zeros with NaN then fill
        X['cost_per_mile'] = X['total_receipts_amount'] / X['miles_traveled'].replace(0, np.nan)
        X['cost_per_mile'] = X['cost_per_mile'].fillna(0.0)
        X['cost_per_day'] = X['total_receipts_amount'] / X['trip_duration_days'].replace(0, np.nan)
        X['cost_per_day'] = X['cost_per_day'].fillna(0.0)
        return X

def build_preprocessing(poly_degree: int = 2):
    """
    Build preprocessing pipeline:
    - Derived features
    - Polynomial features (degree configurable)
    - Standard scaling
    Returns sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ('derive', DerivedFeatures()),
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('scale', StandardScaler())
    ])
    return pipeline