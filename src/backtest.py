"""
Walk-forward backtest with embargo, signal-proportional sizing, and risk metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import TARGETS, HORIZON_STEPS, N_SPLITS, COST_BPS, TARGET_LAGS_OFFSETS
from .data import load_data
from .features import engineer_all_features, add_target_lags
from .models import make_xgboost, make_elasticnet


def _non_overlap_indices(n: int, horizon: int) -> np.ndarray:
    """Indices for non-overlapping observations."""
    return np.arange(0, n, horizon)


def _compute_pnl(y_pred: np.ndarray, y_true: np.ndarray, cost_bps: float) -> dict:
    """
    Compute PnL with signal-proportional sizing.
    Position = clip(prediction / std(predictions), -1, 1).
    """
    pred_std = np.std(y_pred) + 1e-12
    position = np.clip(y_pred / pred_std, -1, 1)

    pnl = position * y_true
    turnover = np.abs(np.diff(position, prepend=0))
    cost = (cost_bps / 10_000) * turnover
    net_pnl = pnl - cost

    # Drawdown
    cum_pnl = np.cumsum(net_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = np.min(cum_pnl - running_max)

    return {
        "position": position,
        "net_pnl": net_pnl,
        "max_dd": max_dd,
        "avg_turnover": np.mean(turnover),
    }


def walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_factory=None,
    n_splits: int = N_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Walk-forward evaluation with embargo.

    Parameters
    ----------
    df : DataFrame with features, target, and 'Dates' column.
    feature_cols : list of feature column names (excluding target lags).
    target : target column name.
    model_factory : callable returning a fresh model. Defaults to XGBoost.
    n_splits : number of walk-forward folds.

    Returns
    -------
    fold_results : DataFrame with per-fold metrics (non-overlapping).
    oos_df : DataFrame with all OOS predictions and PnL.
    last_model : the model fitted on the last fold (for feature importance).
    """
    horizon = HORIZON_STEPS[target]

    df_wf = df.dropna(subset=[target]).copy()

    # Add target lags
    df_wf, lag_cols = add_target_lags(df_wf, target, horizon, TARGET_LAGS_OFFSETS)
    df_wf = df_wf.dropna(subset=lag_cols)

    all_feat = feature_cols + lag_cols
    X_all = df_wf[all_feat].copy()

    # Drop completely missing features to avoid SimpleImputer warnings.
    all_nan_cols = X_all.columns[X_all.isna().all()].tolist()
    if all_nan_cols:
        X_all = X_all.drop(columns=all_nan_cols)
        all_feat = [c for c in all_feat if c not in all_nan_cols]

    y_all = df_wf[target]
    dates_all = df_wf["Dates"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    oos_parts = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), 1):
        if len(train_idx) <= horizon:
            continue

        # Embargo: remove last `horizon` observations from train
        train_idx = train_idx[:-horizon]

        X_train = X_all.iloc[train_idx]
        y_train = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        # Drop columns that are entirely NaN in the training fold (sklearn imputer warning)
        train_non_na_cols = X_train.columns[X_train.notna().any()].tolist()
        if len(train_non_na_cols) < X_train.shape[1]:
            dropped = [c for c in X_train.columns if c not in train_non_na_cols]
            X_train = X_train[train_non_na_cols]
            X_test = X_test[train_non_na_cols]

        model = (model_factory or make_xgboost)()

        if hasattr(model, "fit"):
            # sklearn Pipeline or XGBRegressor
            model.fit(X_train, y_train, **({} if hasattr(model, "steps") else {"verbose": False}))
        y_pred = model.predict(X_test)

        # --- Non-overlapping metrics ---
        idx_no = _non_overlap_indices(len(y_test), horizon)
        y_test_no = y_test.values[idx_no]
        y_pred_no = y_pred[idx_no]

        ic = np.corrcoef(y_pred_no, y_test_no)[0, 1]
        hit_rate = (np.sign(y_pred_no) == np.sign(y_test_no)).mean()

        # PnL
        pnl_info = _compute_pnl(y_pred, y_test.values, COST_BPS)
        non_overlap_pnl = pnl_info["net_pnl"][idx_no]
        sharpe = (np.mean(non_overlap_pnl) / (np.std(non_overlap_pnl) + 1e-12)) * np.sqrt(252 / horizon)

        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "ic": ic,
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "max_dd": pnl_info["max_dd"],
        })

        oos_parts.append(pd.DataFrame({
            "date": dates_all.iloc[test_idx].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "position": pnl_info["position"],
            "net_pnl": pnl_info["net_pnl"],
        }))

    results_df = pd.DataFrame(fold_results)
    oos_df = pd.concat(oos_parts).reset_index(drop=True)
    return results_df, oos_df, model


def print_results(results: pd.DataFrame, target: str, model_name: str) -> None:
    """Pretty-print fold-level and average results."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} — {target} (horizon={HORIZON_STEPS[target]}d)")
    print("=" * 60)
    print(results[["fold", "ic", "hit_rate", "sharpe", "max_dd"]].round(4).to_string(index=False))
    print(f"\nMean: IC={results['ic'].mean():.3f}, "
          f"Hit={results['hit_rate'].mean():.1%}, "
          f"Sharpe={results['sharpe'].mean():.2f}, "
          f"MaxDD={results['max_dd'].mean():.4f}")


# --- Main entry point ---
if __name__ == "__main__":
    df, feature_cols = load_data()

    # --- ElasticNet baseline ---
    for target in TARGETS:
        results, oos, _ = walkforward(df, feature_cols, target, model_factory=make_elasticnet)
        print_results(results, target, "ElasticNet")

    # --- XGBoost on raw features ---
    for target in TARGETS:
        results, oos, _ = walkforward(df, feature_cols, target)
        print_results(results, target, "XGBoost (raw)")

    # --- XGBoost on engineered features ---
    df_eng, eng_cols = engineer_all_features(df, feature_cols)
    all_feature_cols = feature_cols + eng_cols

    for target in TARGETS:
        results, oos, model = walkforward(df_eng, all_feature_cols, target)
        print_results(results, target, "XGBoost (engineered)")
