# Commodity Return Forecasting

This project forecasts 5-day and 21-day commodity returns using machine learning. We use XGBoost to extract patterns from 60 anonymized features. The methodology relies on a strict walk-forward cross-validation setup to ensure that no future information leaks into the past.

## Empirical Results

| Forecast Horizon | Pearson IC | Hit Rate | Sharpe (ann.) | Transaction Costs |
|------------------|-----------|----------|---------------|-------------------|
| 5-day (1-week)   | 0.505     | 66.9%    | 2.71          | 2 bps             |
| 21-day (1-month) | 0.461     | 64.1%    | 1.51          | 2 bps             |

An out-of-sample Information Coefficient (IC) of 0.505 shows a strong predictive signal. This high performance comes from careful feature engineering and strict testing methods, rather than just the choice of the machine learning model.

## Validation Framework

To prevent look-ahead bias, our testing method is strictly chronological. All data preprocessing, such as filling missing values and scaling, is calculated using only the training data. We do not use random data shuffling. Target variables are shifted by the exact forecast horizon to avoid overlap. We also leave a gap of days between the training and testing sets in every fold. Finally, all performance metrics are calculated on non-overlapping periods to give a realistic view of future performance.

## Feature Engineering

We expanded the dataset from 60 to 314 features using only past data. These new features help the model understand the recent market context. We calculated 21-day and 63-day rolling z-scores to measure how far values are from their recent average. We also used percentile ranks to reduce the impact of extreme outliers. Finally, we added 21-day rolling standard deviations to measure local volatility, along with lagged features and momentum (the change in a feature over the forecast horizon).

## Model Progression

The table below shows how adding engineered features improves model accuracy. The scores represent the average out-of-sample Information Coefficient across all test folds.

| Architecture | IC (5d) | IC (21d) |
|---|---------|----------|
| ElasticNet (Linear Baseline) | 0.402 | 0.241 |
| XGBoost (Raw Features) | 0.469 | 0.442 |
| XGBoost (Engineered Space) | 0.517 | 0.492 |

## Portfolio Simulation & Risk Dynamics

In our backtest, the size of each trade is proportional to the strength of the model's prediction. We apply a trading cost of 2 basis points for every transaction. We track maximum drawdowns and Calmar ratios to evaluate risk. The results show that while the 21-day model makes accurate predictions, it suffers from massive drawdowns. This highlights the need for a strict stop-loss and volatility targeting strategy before trading this signal in real life.

## Project Structure

```
commodity-return-forecasting/
├── pyproject.toml
├── src/
│   ├── config.py              # Constants, paths, hyperparameters
│   ├── data.py                # Data ingestion and preprocessing
│   ├── features.py            # Feature engineering transformations
│   ├── models.py              # Architecture definitions
│   ├── backtest.py            # Walk-forward validation & PnL simulation
│   └── analysis.py            # SHAP values, metrics, plotting
├── notebooks/
│   └── exploration.ipynb      # EDA and statistical testing
└── reports/
    ├── report.tex             # LaTeX quantitative report
    └── report.pdf             # Compiled document
```

## Execution Instructions

```bash
git clone https://github.com/Nassim-Ta/commodity-return-forecasting.git
cd commodity-return-forecasting
uv sync
```

> **Note**: Raw datasets (`features.csv`, `output.csv`) are structurally excluded from version control. Ensure they are placed in the project root prior to execution.

```bash
# Execute the full validation and simulation pipeline
uv run python -m src.backtest

# Launch interactive exploratory analysis
uv run jupyter notebook notebooks/exploration.ipynb
```

## Documentation

A comprehensive quantitative report detailing the statistical methodology, feature importance analysis, and structural limitations is available in [`reports/report.pdf`](reports/report.pdf).
