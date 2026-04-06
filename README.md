# Commodity Return Forecasting

Forecasting 5-day and 21-day commodity returns using XGBoost on 60 anonymised features. The methodology uses walk-forward cross-validation with embargo, early stopping, and causal position sizing. All designed to prevent look-ahead bias at every stage of the pipeline.

## Key Results

> **Note**: results below are from the corrected pipeline. An earlier version had a subtle feature-selection leak (see [Methodology Notes](#methodology-notes) below) which inflated ICs. The numbers here are from the clean version.

| Forecast Horizon | Pearson IC | Hit Rate | Sharpe (ann.) | TC |
|------------------|-----------|----------|---------------|-----|
| 5-day (1-week)   | 0.464      | 65.8%     | 2.58           | 2 bps |
| 21-day (1-month) | 0.337       | 58.4%      | 0.97         | 2 bps |

*(Run `uv run python -m src.backtest` with data files in place to get updated numbers.)*

## Validation Framework

Walk-forward with 5 folds. For each fold:
1. Training data ends `h` days before the test period starts (embargo gap)
2. XGBoost uses early stopping on the last 15% of training data (held out chronologically) to pick the number of trees, instead of using a fixed count
3. All feature engineering (z-scores, lags, etc.) is backward-looking only
4. Position sizing uses an expanding-window std of past predictions. It does NOT use the full test fold's prediction distribution, which was a form of look-ahead
5. IC and Sharpe are computed on non-overlapping observations to avoid autocorrelation inflation
6. Bootstrap confidence intervals and a t-test on IC are reported for each fold

## Feature Engineering

We go from 60 to ~400 features using only past data:

- **Rolling z-scores** (21d and 63d windows): how far is the current value from its recent average
- **Cross-sectional percentile ranks**: nonlinear rescaling across features on the same date. Honestly this is debatable since we only have one asset. It's not a true cross-sectional rank. But it squashes outliers and the tree model seems to benefit from it
- **Rolling volatility** (21d): local feature instability
- **Lagged features**: shifted by the forecast horizon to avoid overlap with the prediction window
- **Momentum**: feature change over the forecast horizon

All transforms are applied uniformly to all 60 raw features. There is no target-based feature selection prior to the walk-forward split.

## Methodology Notes

The initial version of this project selected which features to enrich (volatility, lags, momentum) by computing full-sample correlations with the target. This is a form of look-ahead bias: the selection was conditioned on test-period target values, even though the transforms themselves were backward-looking. The current version removes this entirely: all features are transformed uniformly, and XGBoost's internal regularisation (colsample, depth limits, L1/L2) handles feature selection implicitly.

Similarly, the original position sizing normalised predictions by `np.std(y_pred)` computed over the entire test fold, meaning early positions depended on future predictions. This is now replaced by an expanding-window normalisation.

## Model Progression

| Architecture | IC (5d) | IC (21d) |
|---|---------|----------|
| ElasticNet (baseline) | 0.402 | 0.241 |
| XGBoost (raw features) | 0.472 | 0.381 |
| XGBoost (engineered) | 0.464 | 0.337 |

## Project Structure

```
commodity-return-forecasting/
├── pyproject.toml
├── src/
│   ├── config.py              # constants, hyperparams
│   ├── data.py                # csv loading and merge
│   ├── features.py            # feature transforms (all backward-looking)
│   ├── models.py              # ElasticNet + XGBoost definitions
│   ├── backtest.py            # walk-forward loop, PnL, statistics
│   └── analysis.py            # EDA plots, feature importance, diagnostics
├── notebooks/
│   └── exploration.ipynb      # EDA and interactive analysis
└── reports/
    ├── report.tex
    └── report.pdf
```

## Running

```bash
git clone https://github.com/Nassim-Ta/commodity-return-forecasting.git
cd commodity-return-forecasting
uv sync
```

> Raw data (`features.csv`, `output.csv`) is excluded from version control.

```bash
# full pipeline
uv run python -m src.backtest

# interactive EDA
uv run jupyter notebook notebooks/exploration.ipynb
```

## Report

Full writeup in [`reports/report.pdf`](reports/report.pdf).
