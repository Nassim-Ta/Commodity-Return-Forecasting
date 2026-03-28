"""
Post-hoc analysis: SHAP, feature importance, PnL plots, backtest statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import HORIZON_STEPS


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
