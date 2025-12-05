"""
tree_models.py
Decision tree and random forest regressors.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def decision_tree(max_depth: int | None = None):
    """Decision tree regressor."""
    return DecisionTreeRegressor(max_depth=max_depth, random_state=42)

def random_forest(n_estimators: int = 100):
    """Random forest regressor."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)