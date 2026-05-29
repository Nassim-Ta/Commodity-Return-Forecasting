"""Post-hoc analysis: feature importance, PnL plots, diagnostics."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from .config import HORIZON_STEPS, TARGETS


# ============================================================
# EDA plots
# ============================================================

def plot_target_distributions(df, targets):
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 4))
    for ax, target in zip(axes, targets):
        data = df[target].dropna()
        ax.hist(data, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{target}\nSkew={data.skew():.2f}, Kurt={data.kurtosis():.2f}")
    plt.tight_layout()
    plt.show()


def check_stationarity(df, targets):
    """ADF test on each target series."""
    for target in targets:
        series = df[target].dropna()
        adf_stat, adf_p, *_ = adfuller(series, maxlag=20)
        status = "Stationary" if adf_p < 0.05 else "Non-stationary"
        print(f"{target}: ADF stat={adf_stat:.3f}, p={adf_p:.2e} -> {status}")


def plot_missing_values(df, feature_cols):
    plt.figure(figsize=(14, 4))
    avail = df[feature_cols].notna().sum(axis=1)
    plt.fill_between(df["Dates"], avail, alpha=0.7)
    plt.title("Feature availability over time")
    plt.xlabel("Date")
    plt.ylabel(f"Features available (out of {len(feature_cols)})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    missing_pct = df[feature_cols].isna().mean().sort_values(ascending=False)
    print(f"Features with >50% missing: {(missing_pct > 0.5).sum()}")
    print(f"Features with >20% missing: {(missing_pct > 0.2).sum()}")
    print(f"Features with 0% missing:   {(missing_pct == 0).sum()}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(missing_pct)), missing_pct.values, width=1, alpha=0.7)
    ax.set_xlabel("Feature index (sorted)")
    ax.set_ylabel("Missing %")
    ax.set_title("Missing values per feature")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_target_correlations(df, targets, feature_cols):
    for target in targets:
        corrs = df[feature_cols].corrwith(df[target]).dropna().sort_values()
        print(f"\n{target} -- top 10 correlated features:")
        print(corrs.tail(10).round(4).to_string())
        print(f"\n{target} -- bottom 10:")
        print(corrs.head(10).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, target in zip(axes, targets):
        corrs = df[feature_cols].corrwith(df[target]).dropna()
        ax.hist(corrs, bins=40, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(f"Feature-target correlations -- {target}")
        ax.set_xlabel("Pearson r")
    plt.tight_layout()
    plt.show()


def plot_rolling_correlations(df, targets, feature_cols, window=252):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, target in zip(axes, targets):
        top_feats = (
            df[feature_cols].corrwith(df[target]).abs().nlargest(5).index.tolist()
        )
        for feat in top_feats:
            rc = df[feat].rolling(window).corr(df[target])
            ax.plot(df["Dates"], rc, alpha=0.6, label=feat[:15])
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"Rolling {window}d correlation -- {target}")
        ax.set_ylabel("Correlation")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation_matrix(df, feature_cols):
    corr = df[feature_cols].corr()
    n_high = 0
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.9:
                n_high += 1
    print(f"Feature pairs with |corr| > 0.9: {n_high}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
        xticklabels=False, yticklabels=False,
    )
    ax.set_title(f"Feature correlation matrix ({len(feature_cols)} features)")
    plt.tight_layout()
    plt.show()


# ============================================================
# Backtest analysis
# ============================================================

def plot_cumulative_pnl(oos, target, title_suffix=""):
    horizon = HORIZON_STEPS[target]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    oos_no = oos.iloc[::horizon].copy()
    oos_no["nav"] = np.cumprod(1 + oos_no["net_ret"].values)

    axes[0].plot(oos_no["date"], oos_no["nav"], lw=1.5, color="green")
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_title(f"{target} -- NAV{title_suffix}")
    axes[0].set_ylabel("NAV")
    axes[0].grid(alpha=0.3)

    nav = np.cumprod(1 + oos["net_ret"].values)
    peak = np.maximum.accumulate(nav)
    dd_pct = (nav - peak) / peak * 100

    axes[1].fill_between(oos["date"], dd_pct, 0, alpha=0.5, color="red")
    axes[1].set_title(f"{target} -- Drawdown (%)")
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_backtest_stats(oos, target):
    horizon = HORIZON_STEPS[target]
    idx_no = np.arange(0, len(oos), horizon)

    ret_no = oos["net_ret"].values[idx_no]
    y_pred_no = oos["y_pred"].values[idx_no]
    y_true_no = oos["y_true"].values[idx_no]

    nav = np.cumprod(1 + oos["net_ret"].values)
    peak = np.maximum.accumulate(nav)
    max_dd = np.min((nav - peak) / peak)

    sharpe = (np.mean(ret_no) / (np.std(ret_no) + 1e-12)) * np.sqrt(252 / horizon)
    ic = np.corrcoef(y_pred_no, y_true_no)[0, 1]
    hit = (np.sign(y_pred_no) == np.sign(y_true_no)).mean()
    ann_return = np.mean(ret_no) * 252 / horizon
    calmar = ann_return / (abs(max_dd) + 1e-12)
    turnover = np.abs(np.diff(oos["position"].values, prepend=0)).mean()

    winning = ret_no[ret_no > 0]
    losing = ret_no[ret_no < 0]
    win_loss = (
        np.mean(winning) / (abs(np.mean(losing)) + 1e-12) if len(losing) > 0 else np.inf
    )

    print(f"\n{target} (horizon={horizon}d):")
    print(f"  IC (non-overlap):       {ic:.3f}")
    print(f"  Hit rate:               {hit:.1%}")
    print(f"  Sharpe (ann.):          {sharpe:.2f}")
    print(f"  Max drawdown:           {max_dd:.1%}")
    print(f"  Calmar ratio:           {calmar:.2f}")
    print(f"  Win/Loss ratio:         {win_loss:.2f}")
    print(f"  Avg daily turnover:     {turnover:.2%}")
    print(f"  N obs (non-overlap):    {len(ret_no)}")

def plot_feature_importance(model, X_sample=None, n_top=20, title=""):
    """
    Uses SHAP if X_sample is provided, otherwise falls back to
    XGBoost's built-in gain metric.
    """
    try:
        if X_sample is None:
            raise ValueError("no sample data, skip shap")
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        importance = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=X_sample.columns,
        ).sort_values(ascending=False)
        method = "SHAP"
    except Exception:
        importance = pd.Series(
            model.feature_importances_,
            index=model.get_booster().feature_names,
        ).sort_values(ascending=False)
        method = "gain"

    fig, ax = plt.subplots(figsize=(10, 8))
    importance.head(n_top).plot(kind="barh", ax=ax)
    ax.set_title(f"Top {n_top} features ({method}) {title}")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance
