"""
boosting_models.py
XGBoost wrapper with fallback to sklearn's GradientBoostingRegressor if XGBoost not installed.
"""

try:
    from xgboost import XGBRegressor
    def xgboost_model(n_estimators: int = 100):
        return XGBRegressor(n_estimators=n_estimators, random_state=42, verbosity=0, n_jobs=1)
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    def xgboost_model(n_estimators: int = 100):
        # Fallback when xgboost is not available
        return GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)