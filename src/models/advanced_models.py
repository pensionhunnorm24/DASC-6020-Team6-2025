"""
advanced_models.py
Support Vector Regression and MLP regressor wrappers.
"""

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def svr_model(C: float = 1.0, epsilon: float = 0.1):
    """Support Vector Regression (RBF kernel)."""
    return SVR(C=C, epsilon=epsilon, kernel='rbf')

def mlp_model(hidden_layer_sizes=(50,50), max_iter: int = 500):
    """Multi-layer Perceptron regressor."""
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=max_iter)