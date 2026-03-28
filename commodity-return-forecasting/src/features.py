"""
Feature engineering pipeline. All transformations are backward-looking only.
"""

import numpy as np
import pandas as pd
from .config import (
    TARGETS, HORIZON_STEPS, ZSCORE_WINDOWS,
    VOLATILITY_WINDOW, N_TOP_FEATURES
)


def _get_top_features(df: pd.DataFrame, feature_cols: list[str], target: str, n: int) -> list[str]:
    """Return the n features most correlated (in absolute value) with target."""
    corrs = df[feature_cols].corrwith(df[target]).abs()
    return corrs.nlargest(n).index.tolist()


def add_rolling_zscores(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Rolling z-scores: (x - rolling_mean) / rolling_std.
    Captures whether current values are unusual relative to recent history.
    """
    new_cols = []
    for window in ZSCORE_WINDOWS:
        for col in feature_cols:
            roll_mean = df[col].rolling(window, min_periods=window // 2).mean()
            roll_std = df[col].rolling(window, min_periods=window // 2).std()
            col_name = f"{col}_zscore_{window}d"
            df[col_name] = (df[col] - roll_mean) / (roll_std + 1e-12)
            new_cols.append(col_name)
    return df, new_cols


def add_cross_sectional_ranks(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Percentile rank of each feature relative to all features on the same date.
    Robust to outliers and non-stationarity in levels.
    """
    new_cols = []
    ranks = df[feature_cols].rank(axis=1, pct=True)
    for col in feature_cols:
        col_name = f"{col}_xrank"
        df[col_name] = ranks[col]
        new_cols.append(col_name)
    return df, new_cols


def add_rolling_volatility(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Rolling standard deviation of top features per target."""
    new_cols = []
    seen = set()
    for target in TARGETS:
        top_feats = _get_top_features(df, feature_cols, target, N_TOP_FEATURES)
        for col in top_feats:
            col_name = f"{col}_vol_{VOLATILITY_WINDOW}d"
            if col_name not in seen:
                df[col_name] = df[col].rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW // 2).std()
                new_cols.append(col_name)
                seen.add(col_name)
    return df, new_cols


def add_feature_lags(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Lagged values of top features, shifted by at least the forecast horizon.
    Captures momentum and mean-reversion in inputs.
    """
    new_cols = []
    seen = set()
    for target in TARGETS:
        horizon = HORIZON_STEPS[target]
        top_feats = _get_top_features(df, feature_cols, target, N_TOP_FEATURES)
        for col in top_feats:
            for mult in [1, 2]:
                lag = horizon * mult
                col_name = f"{col}_lag{lag}"
                if col_name not in seen:
                    df[col_name] = df[col].shift(lag)
                    new_cols.append(col_name)
                    seen.add(col_name)
    return df, new_cols


def add_momentum(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Feature change over the forecast horizon."""
    new_cols = []
    seen = set()
    for target in TARGETS:
        horizon = HORIZON_STEPS[target]
        top_feats = _get_top_features(df, feature_cols, target, N_TOP_FEATURES)
        for col in top_feats:
            col_name = f"{col}_mom_{horizon}d"
            if col_name not in seen:
                df[col_name] = df[col] - df[col].shift(horizon)
                new_cols.append(col_name)
                seen.add(col_name)
    return df, new_cols


def add_target_lags(df: pd.DataFrame, target: str, horizon: int, offsets: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Horizon-shifted target lags: lag_k = target.shift(horizon + offset).
    Prevents overlap between return windows.
    """
    lag_cols = []
    for offset in offsets:
        lag = horizon + offset
        col_name = f"target_lag_{lag}"
        df[col_name] = df[target].shift(lag)
        lag_cols.append(col_name)
    return df, lag_cols


def engineer_all_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full feature engineering pipeline. Returns (df, all_new_columns).
    """
    df = df.copy()
    all_new = []

    df, cols = add_rolling_zscores(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_cross_sectional_ranks(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_rolling_volatility(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_feature_lags(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_momentum(df, feature_cols)
    all_new.extend(cols)

    print(f"Engineered {len(all_new)} new features (total: {len(feature_cols) + len(all_new)})")
    return df, all_new
