"""
Configuration file for Forex ML Prediction System
Contains constants, thresholds, and pip multipliers
"""

# ============================================
# CURRENCY PAIRS
# ============================================
SYMBOLS = [
    "EURUSD",
    "GBPUSD", 
    "USDJPY",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "EURGBP"
]

# ============================================
# PIP MULTIPLIERS (for pip calculation)
# ============================================
# For JPY pairs: 1 pip = 0.01 (multiplier = 100)
# For others: 1 pip = 0.0001 (multiplier = 10000)
PIP_MULTIPLIERS = {
    "EURUSD": 10000,
    "GBPUSD": 10000,
    "USDJPY": 100,
    "USDCHF": 10000,
    "EURJPY": 100,
    "GBPJPY": 100,
    "EURGBP": 10000
}

# ============================================
# MODEL PARAMETERS
# ============================================
SEQUENCE_LENGTH = 30          # 30 minutes of history
PREDICTION_HORIZON = 10       # Predict 10 minutes ahead
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001

# LSTM Architecture
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# ============================================
# TRADING THRESHOLDS
# ============================================
DIRECTION_THRESHOLD_PIPS = 3.0    # Pips threshold for UP/DOWN
CONFIDENCE_THRESHOLD = 0.6        # Min confidence for BUY/SELL
REWARD_RISK_RATIO = 1.5           # Max up must be 1.5x max down

# ============================================
# FILE PATHS
# ============================================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "forex_live_data.csv")
NEW_DATA_FILE = os.path.join(BASE_DIR, "forex_new_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ARCHIVES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archives")

# ============================================
# FEATURE NAMES
# ============================================
FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_3", 
    "log_return_5",
    "log_return_10",
    "price_diff_1",
    "price_diff_5",
    "tick_count",
    "range_pips",
    "tick_volatility",
    "price_acceleration",
    "direction_3",
    "volatility_ratio"
]

NUM_FEATURES = len(FEATURE_COLUMNS)
