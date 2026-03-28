# Commodity Return Forecasting

Predict 1-week and 1-month commodity returns using machine learning on 60 anonymised features. Walk-forward validated with strict anti-leakage methodology.

## Results

| Horizon | Pearson IC | Hit Rate | Sharpe (ann.) | After Costs |
|---------|-----------|----------|---------------|-------------|
| 1-week  | 0.505     | 66.9%    | 2.71          | 2 bps       |
| 1-month | 0.461     | 64.1%    | 1.51          | 2 bps       |

All metrics computed on **non-overlapping walk-forward OOS** observations only.

## Why This Matters

An IC of 0.45 is strong for commodity return prediction. Most published signals in the literature report ICs of 0.02–0.10. The key is not the model (XGBoost) — it's the **feature engineering** and the **evaluation discipline** that make the results credible.

## Anti-Leakage Discipline

Every design choice prevents information from the future leaking into the past:

- Preprocessing (imputation, scaling) fitted on **train only**
- `TimeSeriesSplit` everywhere — **no random K-fold**
- Target lags shifted by forecast horizon — **no overlap**
- **Embargo gap** between train and test in walk-forward folds
- Metrics on **non-overlapping observations** — no autocorrelation inflation

## Feature Engineering

140+ features engineered from 60 raw inputs (all backward-looking):

| Type | Description | Count |
|------|-------------|-------|
| Rolling z-scores | 21d and 63d normalisation | 120 |
| Cross-sectional ranks | Percentile rank across features per row | 60 |
| Rolling volatility | 21d std of top predictors | ~10 |
| Feature lags | Horizon-shifted lags of top features | ~20 |
| Momentum | Feature change over horizon | ~10 |

## Model Progression

| Model | IC (1w) | IC (1m) |
|-------|---------|---------|
| ElasticNet | 0.402 | 0.241 |
| XGBoost (Raw Features) | 0.469 | 0.442 |
| **XGBoost (Engineered)** | **0.517** | **0.492** |

## Backtest

- **Signal-proportional sizing**: position ∝ prediction strength (not binary sign)
- **2 bps transaction costs** per unit of turnover
- **Drawdown analysis**: max drawdown and Calmar ratio reported
- **Per-fold statistics**: not just averages — fold-level variance shown

## Project Structure

```
commodity-return-forecasting/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py              # Constants, paths, parameters
│   ├── data.py                # Data loading and merging
│   ├── features.py            # Feature engineering pipeline
│   ├── models.py              # Model definitions
│   ├── backtest.py            # Walk-forward evaluation + PnL
│   └── analysis.py            # SHAP, feature importance, plots
├── notebooks/
│   └── exploration.ipynb      # EDA and visual analysis
├── reports/
│   ├── report.tex             # LaTeX report
│   └── report.pdf             # Compiled report
└── results/
    └── .gitkeep
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/commodity-return-forecasting.git
cd commodity-return-forecasting
uv sync
```

> **Note**: Data files (`features.csv`, `output.csv`) are not included in this repository. Place them in the project root before running.

```bash
# Run full pipeline
uv run python -m src.backtest

# Or explore interactively
uv run jupyter notebook notebooks/exploration.ipynb
```

## Tech Stack

Python · NumPy · Pandas · Scikit-learn · XGBoost · SHAP · Statsmodels · Matplotlib

## Report

A full LaTeX report is available in [`reports/report.pdf`](reports/report.pdf) covering methodology, results, feature importance analysis, and limitations.
