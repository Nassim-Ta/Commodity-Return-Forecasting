"""Hyperparameters and constants."""

TARGETS = ["output1_1w", "output1_1m"]

# horizons in trading days
HORIZON_STEPS = {"output1_1w": 5, "output1_1m": 21}

# walk-forward
N_SPLITS = 5

# cost in bps per unit turnover
COST_BPS = 2.0

# feature engineering windows
ZSCORE_WINDOWS = [21, 63]
VOLATILITY_WINDOW = 21
TARGET_LAGS_OFFSETS = [0, 1, 2, 5, 10]

# depth 4 instead of 5 to limit overfitting on 300+ features
XGB_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "max_depth": 4,
    "subsample": 0.75,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "tree_method": "hist",
}

EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_FRAC = 0.15  # last 15% of train becomes validation

INITIAL_CAPITAL = 1.0
