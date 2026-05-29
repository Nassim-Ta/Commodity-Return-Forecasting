"""
Feature engineering — all transforms are strictly backward-looking
(rolling windows, shifts), so no future leak in the transforms themselves.

Earlier version picked which features to enrich using full-sample
correlation with the target. That's look-ahead bias: the selection
depended on test-period targets. Removed it — we now transform
all 60 raw features uniformly and let the model sort out relevance.
"""

import numpy as np
import pandas as pd
from .config import ZSCORE_WINDOWS, VOLATILITY_WINDOW, HORIZON_STEPS, TARGETS


def add_rolling_zscores(df, feature_cols):
    """Trailing-window z-score: is the current value unusual vs recent history."""
    df = df.copy()
    new_cols = []
    chunks = {}

    for w in ZSCORE_WINDOWS:
        for col in feature_cols:
            mu = df[col].rolling(w, min_periods=w // 2).mean()
            sigma = df[col].rolling(w, min_periods=w // 2).std()
            name = f"{col}_zscore_{w}d"
            chunks[name] = (df[col] - mu) / (sigma + 1e-12)
            new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(chunks, index=df.index)], axis=1)
    return df, new_cols


def add_cross_sectional_ranks(df, feature_cols):
    """
    Percentile rank across features on the same date.

    This is a within-row rank across heterogeneous features, not a proper
    cross-asset rank (we only have one commodity). Mainly acts as a nonlinear
    rescaling that squashes outliers. Debatable whether it means much with
    anonymised features, but the tree model benefits empirically.
    """
    df = df.copy()
    new_cols = []
    ranks = df[feature_cols].rank(axis=1, pct=True)
    rank_data = {}

    for col in feature_cols:
        name = f"{col}_xrank"
        rank_data[name] = ranks[col]
        new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(rank_data, index=df.index)], axis=1)
    return df, new_cols


def add_rolling_volatility(df, feature_cols):
    """Rolling std — measures local instability of each feature."""
    df = df.copy()
    new_cols = []
    chunks = {}
    w = VOLATILITY_WINDOW

    for col in feature_cols:
        name = f"{col}_vol_{w}d"
        chunks[name] = df[col].rolling(w, min_periods=w // 2).std()
        new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(chunks, index=df.index)], axis=1)
    return df, new_cols


def add_feature_lags(df, feature_cols, horizon):
    """Lagged values shifted by multiples of the horizon — no overlap with prediction window."""
    df = df.copy()
    new_cols = []
    chunks = {}

    for col in feature_cols:
        for mult in [1, 2]:
            lag = horizon * mult
            name = f"{col}_lag{lag}"
            if name not in chunks:
                chunks[name] = df[col].shift(lag)
                new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(chunks, index=df.index)], axis=1)
    return df, new_cols


def add_momentum(df, feature_cols, horizon):
    """Change over the forecast horizon — basically feature-level returns."""
    df = df.copy()
    new_cols = []
    chunks = {}

    for col in feature_cols:
        name = f"{col}_mom_{horizon}d"
        if name not in chunks:
            chunks[name] = df[col] - df[col].shift(horizon)
            new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(chunks, index=df.index)], axis=1)
    return df, new_cols


def add_target_lags(df, target, horizon, offsets):
    """
    Lagged target values. Each lag = horizon + offset to avoid any overlap
    between the return window we're predicting and the one we use as input.
    """
    df = df.copy()
    lag_cols = []
    for offset in offsets:
        lag = horizon + offset
        name = f"target_lag_{lag}"
        df[name] = df[target].shift(lag)
        lag_cols.append(name)
    return df, lag_cols


def engineer_base_features(df, feature_cols):
    """
    Target-independent transforms: z-scores, ranks, volatility.
    Safe to run on the full dataset before the walk-forward split
    since nothing here touches the target column.
    """
    df = df.copy()
    all_new = []

    df, cols = add_rolling_zscores(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_cross_sectional_ranks(df, feature_cols)
    all_new.extend(cols)

    df, cols = add_rolling_volatility(df, feature_cols)
    all_new.extend(cols)

    print(f"Base features: {len(all_new)} new columns")
    return df, all_new


def engineer_target_features(df, feature_cols, target):
    """
    Horizon-dependent transforms (lags, momentum). Also safe to pre-compute
    since they're just shifts of feature values, but they need the target
    name to determine the horizon.
    """
    horizon = HORIZON_STEPS[target]
    df = df.copy()
    all_new = []

    df, cols = add_feature_lags(df, feature_cols, horizon)
    all_new.extend(cols)

    df, cols = add_momentum(df, feature_cols, horizon)
    all_new.extend(cols)

    print(f"Target-specific features ({target}, h={horizon}): {len(all_new)} new columns")
    return df, all_new
