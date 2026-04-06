"""
Feature engineering. Every transform here is strictly backward-looking
(rolling windows, shifts) so there's no future information leakage in the
transforms themselves. Target-dependent operations (like selecting which
features to enrich) are NOT done here — that would leak if computed on the
full dataset before the walk-forward split.

Previous version had a _get_top_features() that picked features by
full-sample correlation with the target, then only engineered those.
Removed that because it's a subtle but real form of look-ahead bias:
the selection was conditioned on test-period target values.
Now we just transform all 60 raw features uniformly.
"""

import numpy as np
import pandas as pd
from .config import ZSCORE_WINDOWS, VOLATILITY_WINDOW, HORIZON_STEPS, TARGETS


def add_rolling_zscores(df, feature_cols):
    """
    z-score relative to a trailing window.
    Captures whether current value is unusual vs recent history.
    """
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

    NOTE: this is a within-row rank across heterogeneous features, NOT a
    cross-asset rank (we only have one commodity). It acts as a nonlinear
    rescaling that squashes outliers. Interpretation is limited since the
    features are anonymised and may have very different natures. We keep it
    because it empirically helps the tree model, but it's debatable.
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
    """Rolling std of each feature. Measures local instability."""
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
    """
    Lagged values shifted by multiples of the horizon.
    Shift is >= horizon so there's no overlap with the prediction window.
    """
    df = df.copy()
    new_cols = []
    chunks = {}

    for col in feature_cols:
        for mult in [1, 2]:
            lag = horizon * mult
            name = f"{col}_lag{lag}"
            if name not in chunks:  # avoid duplicates across horizons
                chunks[name] = df[col].shift(lag)
                new_cols.append(name)

    df = pd.concat([df, pd.DataFrame(chunks, index=df.index)], axis=1)
    return df, new_cols


def add_momentum(df, feature_cols, horizon):
    """Change over the forecast horizon. Basically feature-level returns."""
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
    Lagged target values. Each lag = horizon + offset so there's
    no overlap between the return window we're predicting and the
    lagged return window we're using as input.
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
    because nothing here touches the target column.
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
    Horizon-dependent transforms: lags and momentum.
    Also safe to pre-compute (they're just shifts of feature values),
    but they depend on the specific horizon so we need the target name.
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
