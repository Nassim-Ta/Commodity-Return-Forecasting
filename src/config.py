"""
Configuration: constants, parameters, and paths.
"""

# --- Targets ---
TARGETS = ["output1_1w", "output1_1m"]

# --- Horizons (trading days) ---
HORIZON_STEPS = {"output1_1w": 5, "output1_1m": 21}

# --- Walk-forward ---
N_SPLITS = 5

# --- Transaction costs (basis points per unit turnover) ---
COST_BPS = 2.0

# --- Feature engineering ---
ZSCORE_WINDOWS = [21, 63]
VOLATILITY_WINDOW = 21
N_TOP_FEATURES = 10
TARGET_LAGS_OFFSETS = [0, 1, 2, 5, 10]  # added to horizon

# --- XGBoost defaults ---
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
}
