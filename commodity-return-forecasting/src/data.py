"""
Data loading and initial validation.
"""

import pandas as pd
from .config import TARGETS


def load_data(features_path: str = "features.csv", output_path: str = "output.csv") -> tuple[pd.DataFrame, list[str]]:
    """
    Load and merge features and targets. Returns (df, feature_columns).
    """
    features_df = pd.read_csv(features_path, parse_dates=["Dates"])
    output_df = pd.read_csv(output_path, parse_dates=["Dates"])

    df = (
        features_df
        .merge(output_df, on="Dates", how="inner")
        .sort_values("Dates")
        .reset_index(drop=True)
    )

    feature_cols = [c for c in df.columns if c not in ["Dates"] + TARGETS]

    print(f"Period:       {df['Dates'].min().date()} → {df['Dates'].max().date()}")
    print(f"Observations: {len(df):,}")
    print(f"Features:     {len(feature_cols)}")
    print(f"Missing:      {df[feature_cols].isna().sum().sum() / df[feature_cols].size:.1%}")

    return df, feature_cols
