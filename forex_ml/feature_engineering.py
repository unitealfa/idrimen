"""
Feature Engineering Module
Generates normalized ML features from M1 candles
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import PIP_MULTIPLIERS, FEATURE_COLUMNS


def compute_log_returns(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """
    Compute logarithmic returns for various periods
    """
    df = df.copy()
    
    # Log returns
    df["log_return_1"] = np.log(df[column] / df[column].shift(1))
    df["log_return_3"] = np.log(df[column] / df[column].shift(3))
    df["log_return_5"] = np.log(df[column] / df[column].shift(5))
    df["log_return_10"] = np.log(df[column] / df[column].shift(10))
    
    return df


def compute_price_differences(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """
    Compute price differences
    """
    df = df.copy()
    
    df["price_diff_1"] = df[column] - df[column].shift(1)
    df["price_diff_5"] = df[column] - df[column].shift(5)
    
    return df


def compute_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price acceleration (change in velocity)
    """
    df = df.copy()
    
    # Velocity = price_diff_1
    # Acceleration = change in velocity
    velocity = df["price_diff_1"]
    df["price_acceleration"] = velocity - velocity.shift(1)
    
    return df


def compute_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directional trend features
    """
    df = df.copy()
    
    # Dominant direction over last 3 minutes
    # Sum of returns, then sign
    cumulative_3 = df["log_return_1"].rolling(3).sum()
    df["direction_3"] = np.sign(cumulative_3)
    
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility expansion/contraction features
    """
    df = df.copy()
    
    # Rolling volatility
    vol_3 = df["log_return_1"].rolling(3).std()
    vol_10 = df["log_return_1"].rolling(10).std()
    
    # Ratio: expansion if > 1, contraction if < 1
    df["volatility_ratio"] = vol_3 / (vol_10 + 1e-10)
    
    return df


def normalize_zscore_rolling(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Normalize using rolling z-score (prevents look-ahead bias)
    
    Args:
        series: Feature series to normalize
        window: Rolling window size
        
    Returns:
        Normalized series
    """
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    
    # Avoid division by zero
    std = std.replace(0, 1e-10)
    
    return (series - mean) / std


def build_features(m1_candles: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Build all features from M1 candles
    
    Args:
        m1_candles: DataFrame with M1 OHLC data
        symbol: Currency pair (for pip conversion)
        
    Returns:
        DataFrame with all features (normalized)
    """
    df = m1_candles.copy()
    
    # Step 1: Compute raw features
    df = compute_log_returns(df)
    df = compute_price_differences(df)
    df = compute_acceleration(df)
    df = compute_direction_features(df)
    df = compute_volatility_features(df)
    
    # tick_count and tick_volatility already exist from data_pipeline
    # range_pips already computed
    
    # Step 2: Fill NaN with 0 for early rows
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Step 3: Normalize all features using rolling z-score
    features_normalized = pd.DataFrame()
    features_normalized["timestamp"] = df["timestamp"]
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            features_normalized[col] = normalize_zscore_rolling(df[col])
        else:
            features_normalized[col] = 0
    
    # Step 4: Replace any remaining NaN/inf
    features_normalized = features_normalized.replace([np.inf, -np.inf], 0)
    features_normalized = features_normalized.fillna(0)
    
    # Keep original close for label generation
    features_normalized["close"] = df["close"]
    features_normalized["high"] = df["high"]
    features_normalized["low"] = df["low"]
    
    return features_normalized


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature matrix from DataFrame
    
    Args:
        df: DataFrame with features
        
    Returns:
        NumPy array of shape (n_samples, n_features)
    """
    return df[FEATURE_COLUMNS].values


if __name__ == "__main__":
    # Test feature engineering
    from data_pipeline import load_tick_data, aggregate_to_m1
    from config import DATA_FILE
    
    ticks = load_tick_data(DATA_FILE)
    m1 = aggregate_to_m1(ticks, "EURUSD")
    features = build_features(m1, "EURUSD")
    
    print("\nFeature sample:")
    print(features[FEATURE_COLUMNS].head(10))
    print(f"\nFeature shape: {get_feature_matrix(features).shape}")
