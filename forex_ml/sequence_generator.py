"""
Sequence Generator Module
Creates temporal sequences for LSTM input
"""

import numpy as np
import pandas as pd
from typing import Tuple

from config import SEQUENCE_LENGTH, FEATURE_COLUMNS


def create_sequences(features: np.ndarray, 
                     labels: dict,
                     valid_indices: np.ndarray,
                     seq_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, dict]:
    """
    Create temporal sequences from feature matrix
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        labels: Dictionary with label arrays
        valid_indices: Indices where labels are valid
        seq_length: Length of each sequence (default 30)
        
    Returns:
        X: Sequences of shape (n_sequences, seq_length, n_features)
        y: Dictionary with aligned label arrays
    """
    X = []
    y_price = []
    y_direction = []
    y_max_up = []
    y_max_down = []
    
    # Convert valid_indices to list of positions
    valid_set = set(valid_indices)
    
    for i in range(seq_length, len(features)):
        # Check if current index has valid labels
        if i not in valid_set:
            continue
            
        # Get the position in labels array
        label_pos = np.where(valid_indices == i)[0]
        if len(label_pos) == 0:
            continue
        label_pos = label_pos[0]
        
        # Extract sequence
        sequence = features[i - seq_length:i]
        X.append(sequence)
        
        # Extract corresponding labels
        y_price.append(labels["price"][label_pos])
        y_direction.append(labels["direction"][label_pos])
        y_max_up.append(labels["max_up"][label_pos])
        y_max_down.append(labels["max_down"][label_pos])
    
    X = np.array(X)
    y = {
        "price": np.array(y_price),
        "direction": np.array(y_direction),
        "max_up": np.array(y_max_up),
        "max_down": np.array(y_max_down)
    }
    
    return X, y


def create_sequences_simple(df: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, dict]:
    """
    Simplified sequence creation from a prepared DataFrame
    
    Args:
        df: DataFrame with features and labels
        seq_length: Sequence length
        
    Returns:
        X, y for model training
    """
    # Drop rows with NaN labels
    df_clean = df.dropna(subset=["future_price", "direction"]).copy()
    
    # Get feature matrix
    features = df_clean[FEATURE_COLUMNS].values
    
    # Prepare labels
    direction_onehot = pd.get_dummies(df_clean["direction"].astype(int), prefix="dir")
    for col in ["dir_0", "dir_1", "dir_2"]:
        if col not in direction_onehot.columns:
            direction_onehot[col] = 0
    direction_onehot = direction_onehot[["dir_0", "dir_1", "dir_2"]]
    
    X = []
    y_price = []
    y_direction = []
    y_max_up = []
    y_max_down = []
    
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i])
        y_price.append(df_clean["future_price"].iloc[i])
        y_direction.append(direction_onehot.iloc[i].values)
        y_max_up.append(df_clean["max_up_pips"].iloc[i])
        y_max_down.append(df_clean["max_down_pips"].iloc[i])
    
    X = np.array(X)
    y = {
        "price": np.array(y_price).reshape(-1, 1),
        "direction": np.array(y_direction),
        "max_up": np.array(y_max_up).reshape(-1, 1),
        "max_down": np.array(y_max_down).reshape(-1, 1)
    }
    
    print(f"Created {len(X)} sequences of shape {X.shape}")
    
    return X, y


def create_prediction_sequence(df: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Create a single sequence for live prediction
    
    Args:
        df: DataFrame with recent features (must have at least seq_length rows)
        seq_length: Sequence length
        
    Returns:
        Sequence of shape (1, seq_length, n_features)
    """
    if len(df) < seq_length:
        raise ValueError(f"Need at least {seq_length} rows, got {len(df)}")
    
    # Take the last seq_length rows
    features = df[FEATURE_COLUMNS].values[-seq_length:]
    
    # Reshape for model input
    return features.reshape(1, seq_length, len(FEATURE_COLUMNS))


if __name__ == "__main__":
    # Test sequence generation
    from data_pipeline import load_tick_data, aggregate_to_m1
    from feature_engineering import build_features
    from label_generator import generate_labels
    from config import DATA_FILE
    
    ticks = load_tick_data(DATA_FILE)
    m1 = aggregate_to_m1(ticks, "EURUSD")
    features = build_features(m1, "EURUSD")
    labeled = generate_labels(features, "EURUSD")
    
    X, y = create_sequences_simple(labeled)
    
    print(f"\nX shape: {X.shape}")
    print(f"y['price'] shape: {y['price'].shape}")
    print(f"y['direction'] shape: {y['direction'].shape}")
