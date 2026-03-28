"""
Model definitions.
"""

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import XGB_PARAMS


def make_elasticnet() -> Pipeline:
    """ElasticNet with train-only preprocessing and temporal CV."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=TimeSeriesSplit(n_splits=3),
            random_state=42,
        )),
    ])


def make_xgboost() -> XGBRegressor:
    """XGBoost with default regularised parameters."""
    return XGBRegressor(**XGB_PARAMS)
