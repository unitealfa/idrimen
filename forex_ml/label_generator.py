"""
Label Generator Module
Creates prediction targets without look-ahead bias
"""

import pandas as pd
import numpy as np

from config import PIP_MULTIPLIERS, PREDICTION_HORIZON, DIRECTION_THRESHOLD_PIPS


def generate_labels(df: pd.DataFrame, symbol: str, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Generate prediction labels for supervised learning
    
    IMPORTANT: Labels use FUTURE data - careful with alignment!
    
    Args:
        df: DataFrame with M1 features (must have close, high, low)
        symbol: Currency pair (for pip conversion)
        horizon: Prediction horizon in minutes (default 10)
        
    Returns:
        DataFrame with labels added
    """
    df = df.copy()
    pip_mult = PIP_MULTIPLIERS.get(symbol, 10000)
    threshold = DIRECTION_THRESHOLD_PIPS
    
    # =============================================
    # 1. Future Price (at t+horizon)
    # =============================================
    df["future_price"] = df["close"].shift(-horizon)
    
    # =============================================
    # 2. Price Change (for direction)
    # =============================================
    df["price_change"] = df["future_price"] - df["close"]
    df["price_change_pips"] = df["price_change"] * pip_mult
    
    # =============================================
    # 3. Direction Classification
    # =============================================
    # UP = 2, FLAT = 1, DOWN = 0
    conditions = [
        (df["price_change_pips"] > threshold),   # UP
        (df["price_change_pips"] < -threshold),  # DOWN
    ]
    choices = [2, 0]  # UP=2, DOWN=0
    df["direction"] = np.select(conditions, choices, default=1)  # FLAT=1
    
    # =============================================
    # 4. Maximum Up/Down Variation over next horizon minutes
    # =============================================
    # Use rolling max/min on high/low shifted back
    max_high = df["high"].rolling(horizon).max().shift(-horizon)
    min_low = df["low"].rolling(horizon).min().shift(-horizon)
    
    df["max_up_pips"] = (max_high - df["close"]) * pip_mult
    df["max_down_pips"] = (df["close"] - min_low) * pip_mult
    
    # Clip to reasonable values
    df["max_up_pips"] = df["max_up_pips"].clip(0, 100)
    df["max_down_pips"] = df["max_down_pips"].clip(0, 100)
    
    return df


def prepare_labels_for_training(df: pd.DataFrame) -> dict:
    """
    Prepare labels in format expected by multi-output model
    
    Args:
        df: DataFrame with labels
        
    Returns:
        Dictionary with label arrays
    """
    # Remove rows with NaN labels (last 'horizon' rows)
    df_clean = df.dropna(subset=["future_price", "direction", "max_up_pips", "max_down_pips"])
    
    # Direction as one-hot encoding
    direction_onehot = pd.get_dummies(df_clean["direction"], prefix="dir")
    # Ensure all 3 classes exist
    for col in ["dir_0", "dir_1", "dir_2"]:
        if col not in direction_onehot.columns:
            direction_onehot[col] = 0
    direction_onehot = direction_onehot[["dir_0", "dir_1", "dir_2"]]
    
    labels = {
        "price": df_clean["future_price"].values.reshape(-1, 1),
        "direction": direction_onehot.values,
        "max_up": df_clean["max_up_pips"].values.reshape(-1, 1),
        "max_down": df_clean["max_down_pips"].values.reshape(-1, 1)
    }
    
    return labels, df_clean.index


def decode_direction(prediction: np.ndarray) -> str:
    """
    Decode direction from model output
    
    Args:
        prediction: Softmax output [p_down, p_flat, p_up]
        
    Returns:
        String: "DOWN", "FLAT", or "UP"
    """
    direction_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
    return direction_map[np.argmax(prediction)]


def get_direction_from_prices(price_before: float, price_after: float, 
                               symbol: str, threshold: float = DIRECTION_THRESHOLD_PIPS) -> str:
    """
    Determine actual direction from price movement
    
    Args:
        price_before: Price at prediction time
        price_after: Actual price after horizon
        symbol: Currency pair
        threshold: Pip threshold for UP/DOWN
        
    Returns:
        Direction string
    """
    pip_mult = PIP_MULTIPLIERS.get(symbol, 10000)
    change_pips = (price_after - price_before) * pip_mult
    
    if change_pips > threshold:
        return "UP"
    elif change_pips < -threshold:
        return "DOWN"
    else:
        return "FLAT"


if __name__ == "__main__":
    # Test label generation
    from data_pipeline import load_tick_data, aggregate_to_m1
    from feature_engineering import build_features
    from config import DATA_FILE
    
    ticks = load_tick_data(DATA_FILE)
    m1 = aggregate_to_m1(ticks, "EURUSD")
    features = build_features(m1, "EURUSD")
    labeled = generate_labels(features, "EURUSD")
    
    print("\nLabel distribution:")
    print(labeled["direction"].value_counts())
    
    print("\nSample labels:")
    print(labeled[["close", "future_price", "price_change_pips", "direction", 
                   "max_up_pips", "max_down_pips"]].head(20))
