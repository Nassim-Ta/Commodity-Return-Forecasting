"""Model definitions."""

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import XGB_PARAMS


def make_elasticnet():
    """ElasticNet baseline with time-series CV for l1_ratio."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=TimeSeriesSplit(n_splits=3),
            random_state=42,
        )),
    ])


def make_xgboost():
    """
    XGBoost with conservative regularisation. n_estimators is an upper bound —
    early stopping in the backtest loop picks the actual tree count.
    """
    return XGBRegressor(**XGB_PARAMS)
