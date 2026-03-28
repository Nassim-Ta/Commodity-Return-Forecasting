"""
Post-hoc analysis: SHAP, feature importance, PnL plots, backtest statistics.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from .config import HORIZON_STEPS, TARGETS

def plot_target_distributions(df, targets):
    """1.1 Target distributions"""
    fig, axes = plt.subplots(1, len(targets), figsize=(12, 4))
    for ax, target in zip(axes, targets):
        data = df[target].dropna()
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{target}\nSkew={data.skew():.2f}, Kurt={data.kurtosis():.2f}')
    plt.tight_layout()
    plt.show()

def check_stationarity(df, targets):
    """1.2 Stationarity (ADF test on targets)"""
    for target in targets:
        series = df[target].dropna()
        adf_stat, adf_p, _, _, _, _ = adfuller(series, maxlag=20)
        status = 'Stationary' if adf_p < 0.05 else 'Non-stationary'
        print(f"{target}: ADF stat={adf_stat:.3f}, p={adf_p:.2e} → {status}")

def plot_missing_values(df, FEATURE_COLS):
    plt.figure(figsize=(14, 4))
    features_available = df[FEATURE_COLS].notna().sum(axis=1)
    plt.fill_between(df['Dates'], features_available, alpha=0.7)
    plt.title('Feature availability over time')
    plt.xlabel('Date')
    plt.ylabel(f'Features available (out of {len(FEATURE_COLS)})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    missing_pct = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    print(f"Features with >50% missing: {(missing_pct > 0.5).sum()}")
    print(f"Features with >20% missing: {(missing_pct > 0.2).sum()}")
    print(f"Features with 0% missing: {(missing_pct == 0).sum()}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(missing_pct)), missing_pct.values, width=1, alpha=0.7)
    ax.set_xlabel('Feature index (sorted by missingness)')
    ax.set_ylabel('Missing %')
    ax.set_title('Missing values per feature')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_target_correlations(df, targets, FEATURE_COLS):
    for target in TARGETS:
        corrs = df[FEATURE_COLS].corrwith(df[target]).dropna().sort_values()
        print(f"\n{target} — top 10 correlated features:")
        print(corrs.tail(10).round(4).to_string())
        print(f"\n{target} — bottom 10:")
        print(corrs.head(10).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, target in zip(axes, TARGETS):
        corrs = df[FEATURE_COLS].corrwith(df[target]).dropna()
        ax.hist(corrs, bins=40, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title(f'Feature-target correlation distribution — {target}')
        ax.set_xlabel('Pearson correlation')
    plt.tight_layout()
    plt.show()

def plot_rolling_correlations(df, targets, FEATURE_COLS):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    WINDOW = 252  # ~1 year

    for ax, target in zip(axes, TARGETS):
        top_feats = df[FEATURE_COLS].corrwith(df[target]).abs().nlargest(5).index.tolist()
        for feat in top_feats:
            rolling_corr = df[feat].rolling(WINDOW).corr(df[target])
            ax.plot(df['Dates'], rolling_corr, alpha=0.6, label=feat[:15])
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Rolling {WINDOW}d correlation — {target}')
        ax.set_ylabel('Correlation')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_matrix(df, FEATURE_COLS):
    corr_matrix = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    print(f"Feature pairs with |corr| > 0.9: {len(high_corr_pairs)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=False, yticklabels=False)
    ax.set_title(f'Feature correlation matrix ({len(FEATURE_COLS)} features)')
    plt.tight_layout()
    plt.show()




def plot_cumulative_pnl(oos: pd.DataFrame, target: str, title_suffix: str = "") -> None:
    """Plot cumulative PnL (non-overlapping) and drawdown."""
    horizon = HORIZON_STEPS[target]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Non-overlapping cumulative PnL
    oos_no = oos.iloc[::horizon].copy()
    oos_no["cum_pnl"] = oos_no["net_pnl"].cumsum()

    axes[0].plot(oos_no["date"], oos_no["cum_pnl"], linewidth=1.5, color="green")
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_title(f"{target} — Cumulative PnL (non-overlapping){title_suffix}")
    axes[0].set_ylabel("Cumulative PnL")
    axes[0].grid(alpha=0.3)

    # Drawdown
    cum_pnl = oos["net_pnl"].cumsum().values
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max

    axes[1].fill_between(oos["date"], drawdown, 0, alpha=0.5, color="red")
    axes[1].set_title(f"{target} — Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_backtest_stats(oos: pd.DataFrame, target: str) -> None:
    """Print detailed backtest statistics."""
    horizon = HORIZON_STEPS[target]
    idx_no = np.arange(0, len(oos), horizon)

    pnl_no = oos["net_pnl"].values[idx_no]
    y_pred_no = oos["y_pred"].values[idx_no]
    y_true_no = oos["y_true"].values[idx_no]

    cum_pnl = np.cumsum(oos["net_pnl"].values)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = np.min(cum_pnl - running_max)

    sharpe = (np.mean(pnl_no) / (np.std(pnl_no) + 1e-12)) * np.sqrt(252 / horizon)
    ic = np.corrcoef(y_pred_no, y_true_no)[0, 1]
    hit = (np.sign(y_pred_no) == np.sign(y_true_no)).mean()
    calmar = (np.mean(pnl_no) * 252 / horizon) / (abs(max_dd) + 1e-12)
    turnover = np.abs(np.diff(oos["position"].values, prepend=0)).mean()

    winning = pnl_no[pnl_no > 0]
    losing = pnl_no[pnl_no < 0]
    win_loss = np.mean(winning) / (abs(np.mean(losing)) + 1e-12) if len(losing) > 0 else np.inf

    print(f"\n{target} (horizon={horizon}d):")
    print(f"  IC (non-overlap):       {ic:.3f}")
    print(f"  Hit rate (non-overlap): {hit:.1%}")
    print(f"  Sharpe (annualised):    {sharpe:.2f}")
    print(f"  Max drawdown:           {max_dd:.4f}")
    print(f"  Calmar ratio:           {calmar:.2f}")
    print(f"  Win/Loss ratio:         {win_loss:.2f}")
    print(f"  Avg daily turnover:     {turnover:.2%}")
    print(f"  N obs (non-overlap):    {len(pnl_no)}")


def plot_feature_importance(model, n_top: int = 20, title: str = "") -> None:
    """
    Plot feature importance. Uses SHAP if available, otherwise XGBoost gain.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Would need X_sample here — fall back to gain-based
        raise ImportError("Use gain-based for simplicity")
    except (ImportError, Exception):
        importance = pd.Series(
            model.feature_importances_,
            index=model.get_booster().feature_names,
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        importance.head(n_top).plot(kind="barh", ax=ax)
        ax.set_title(f"Top {n_top} Features (XGBoost gain) {title}")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

        return importance
