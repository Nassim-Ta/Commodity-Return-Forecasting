# Commodity Return Forecasting

Forecasting short-horizon commodity returns with walk-forward validation and look-ahead bias controls.

## Context

Personal research project, also submitted as a take-home challenge for a commodity trading role. The dataset is anonymised and not included in this repo.

## Pipeline

1. **Raw data**: 60 anonymised features, ~6500 daily observations
2. **Feature engineering**: 60 → 314 features — rolling z-scores (21d/63d), cross-sectional percentile ranks, rolling volatility, lagged values, momentum. All transforms are backward-looking only
3. **Models**: ElasticNet baseline, XGBoost with conservative regularisation and early stopping
4. **Validation**: walk-forward (5 folds, embargo gap of one forecast horizon, non-overlapping return periods). Position sizing uses expanding-window normalisation to avoid look-ahead in the backtest
5. **Costs**: 2 bps per unit turnover applied before computing all performance metrics

## Key issue found and fixed

Mid-project I discovered that the original feature-selection step was a form of look-ahead bias: it picked which features to enrich based on full-sample correlation with the target, meaning the selection was conditioned on test-period returns. Also fixed position sizing, which had been normalising by the full fold's prediction std (future information).

## Results

| Metric | 1-week horizon |
|--------|---------------|
| IC (Pearson) | 0.12 |
| Sharpe (ann.) | 1.45 |
| Hit rate | 57% |

After transaction costs (2 bps). Metrics computed on non-overlapping observations only.

## Stack

Python, XGBoost, scikit-learn, pandas, NumPy, matplotlib

## Usage

```bash
git clone https://github.com/Nassim-Ta/commodity-return-forecasting.git
cd commodity-return-forecasting
uv sync
uv run python -m src.backtest
```

Data files (`features.csv`, `output.csv`) are not included. See `data/README.md` for the expected format.
