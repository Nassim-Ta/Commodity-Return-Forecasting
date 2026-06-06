"""
Walk-forward backtest with embargo, early stopping, and causal position sizing.

Fixes from the initial version:
- feature engineering no longer selects features by full-sample correlation
  (that was a look-ahead leak)
- position sizing uses expanding-window std instead of full-fold std
  (old version used future predictions to normalise current ones)
- XGBoost gets early stopping on a chronological validation chunk
- added bootstrap CIs on IC
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from .config import (
    TARGETS, HORIZON_STEPS, N_SPLITS, COST_BPS,
    TARGET_LAGS_OFFSETS, EARLY_STOPPING_ROUNDS, EARLY_STOPPING_FRAC,
)
from .data import load_data
from .features import engineer_base_features, engineer_target_features, add_target_lags
from .models import make_xgboost, make_elasticnet


def _non_overlap_indices(n, horizon):
    return np.arange(0, n, horizon)


def _compute_pnl_causal(y_pred, y_true, cost_bps, warmup=20):
    """
    PnL with causal sizing. y_true is in percentage points (1.5 = 1.5%),
    so we divide by 100 for compounding.
    """
    n = len(y_pred)
    position = np.zeros(n)

    for t in range(n):
        if t < warmup:
            position[t] = np.sign(y_pred[t])
        else:
            # expanding-window normalisation — only uses past predictions
            pred_std = np.std(y_pred[:t + 1]) + 1e-12
            position[t] = np.clip(y_pred[t] / pred_std, -1, 1)

    port_ret = position * y_true / 100
    turnover = np.abs(np.diff(position, prepend=0))
    cost = (cost_bps / 10_000) * turnover
    net_ret = port_ret - cost

    nav = np.cumprod(1 + net_ret)
    peak = np.maximum.accumulate(nav)
    dd_pct = ((nav - peak) / peak) * 100
    max_dd_pct = np.min(dd_pct)

    return {
        "position": position,
        "net_ret": net_ret,
        "nav": nav,
        "max_dd": max_dd_pct,
        "avg_turnover": np.mean(turnover),
    }

def _bootstrap_ic(y_pred, y_true, n_boot=2000, seed=42):
    """Bootstrap 95% CI on Pearson IC."""
    rng = np.random.RandomState(seed)
    n = len(y_pred)
    ics = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ics[b] = np.corrcoef(y_pred[idx], y_true[idx])[0, 1]
    return np.mean(ics), np.percentile(ics, 2.5), np.percentile(ics, 97.5)


def walkforward(
    df,
    feature_cols,
    target,
    model_factory=None,
    n_splits=N_SPLITS,
    use_early_stopping=True,
):
    """
    Walk-forward CV with embargo. Returns (fold_results, oos_df, last_model).
    """
    horizon = HORIZON_STEPS[target]
    is_xgb = model_factory is None

    df_wf = df.dropna(subset=[target]).copy()

    # target lags shifted by >= horizon to avoid overlap
    df_wf, lag_cols = add_target_lags(df_wf, target, horizon, TARGET_LAGS_OFFSETS)
    df_wf = df_wf.dropna(subset=lag_cols)

    all_feat = feature_cols + lag_cols
    X_all = df_wf[all_feat].copy()

    all_nan = X_all.columns[X_all.isna().all()].tolist()
    if all_nan:
        X_all = X_all.drop(columns=all_nan)
        all_feat = [c for c in all_feat if c not in all_nan]

    y_all = df_wf[target]
    dates_all = df_wf["Dates"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    oos_parts = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), 1):
        if len(train_idx) <= horizon:
            continue

        # embargo: drop last `horizon` rows from training
        train_idx = train_idx[:-horizon]

        X_train = X_all.iloc[train_idx]
        y_train = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        valid_cols = X_train.columns[X_train.notna().any()].tolist()
        if len(valid_cols) < X_train.shape[1]:
            X_train = X_train[valid_cols]
            X_test = X_test[valid_cols]

        model = (model_factory or make_xgboost)()

        if is_xgb and use_early_stopping and not hasattr(model, "steps"):
            # chronological val set from the end of training
            n_train = len(X_train)
            n_val = max(int(n_train * EARLY_STOPPING_FRAC), horizon)
            split_point = n_train - n_val

            X_tr = X_train.iloc[:split_point]
            y_tr = y_train.iloc[:split_point]
            X_val = X_train.iloc[split_point:]
            y_val = y_train.iloc[split_point:]

            model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif hasattr(model, "steps"):
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)

        # non-overlapping metrics
        idx_no = _non_overlap_indices(len(y_test), horizon)
        y_test_no = y_test.values[idx_no]
        y_pred_no = y_pred[idx_no]

        ic = np.corrcoef(y_pred_no, y_test_no)[0, 1]
        hit_rate = (np.sign(y_pred_no) == np.sign(y_test_no)).mean()

        ic_mean, ic_lo, ic_hi = _bootstrap_ic(y_pred_no, y_test_no)

        pnl_info = _compute_pnl_causal(y_pred, y_test.values, COST_BPS)
        ret_no = pnl_info["net_ret"][idx_no]
        sharpe = (
            np.mean(ret_no) / (np.std(ret_no) + 1e-12)
        ) * np.sqrt(252 / horizon)

        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "ic": ic,
            "ic_95_lo": ic_lo,
            "ic_95_hi": ic_hi,
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "max_dd": pnl_info['max_dd'],
        })

        oos_parts.append(pd.DataFrame({
            "date": dates_all.iloc[test_idx].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "position": pnl_info["position"],
            "net_ret": pnl_info["net_ret"],
        }))

    results_df = pd.DataFrame(fold_results)
    oos_df = pd.concat(oos_parts).reset_index(drop=True)
    return results_df, oos_df, model


def print_results(results, target, model_name):
    """Print fold-level results with confidence intervals."""
    print(f"\n{'=' * 65}")
    print(f"{model_name} -- {target} (horizon={HORIZON_STEPS[target]}d)")
    print("=" * 65)

    display_cols = ["fold", "ic", "ic_95_lo", "ic_95_hi", "hit_rate", "sharpe", "max_dd"]
    print(results[display_cols].round(4).to_string(index=False))

    mean_ic = results["ic"].mean()
    std_ic = results["ic"].std()
    n = len(results)

    if n > 1 and std_ic > 0:
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    else:
        t_stat, p_val = np.nan, np.nan

    print(f"\nMean: IC={mean_ic:.3f} +/- {std_ic:.3f}, "
          f"Hit={results['hit_rate'].mean():.1%}, "
          f"Sharpe={results['sharpe'].mean():.2f}, "
          f"MaxDD={results['max_dd'].mean():.4f}%")
    print(f"IC t-stat={t_stat:.2f}, p={p_val:.4f} "
          f"({'significant' if p_val < 0.05 else 'NOT significant'} at 5%)")


# ---- main ----
if __name__ == "__main__":
    df_raw, raw_cols = load_data()

    # base features (z-scores, ranks, vol) — target-independent, no leakage
    df_base, base_cols = engineer_base_features(df_raw, raw_cols)
    base_feature_set = raw_cols + base_cols

    # --- ElasticNet baseline (on raw features only) ---
    for target in TARGETS:
        results, oos, _ = walkforward(
            df_raw, raw_cols, target,
            model_factory=make_elasticnet,
            use_early_stopping=False,
        )
        print_results(results, target, "ElasticNet")

    # --- XGBoost on raw features ---
    for target in TARGETS:
        results, oos, _ = walkforward(df_raw, raw_cols, target)
        print_results(results, target, "XGBoost (raw)")

    # --- XGBoost on engineered features ---
    for target in TARGETS:
        df_full, tgt_cols = engineer_target_features(df_base, raw_cols, target)
        full_feature_set = base_feature_set + tgt_cols

        results, oos, model = walkforward(df_full, full_feature_set, target)
        print_results(results, target, "XGBoost (engineered)")
