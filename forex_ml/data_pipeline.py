"""
Data Pipeline Module
Handles loading tick data and converting to M1 OHLC candles
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from config import SYMBOLS, PIP_MULTIPLIERS


def load_tick_data(csv_path: str) -> pd.DataFrame:
    """
    Load tick data from CSV file
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with parsed timestamps
    """
    # Read CSV (no header in first row based on the data)
    df = pd.read_csv(csv_path, header=None)
    
    # Set column names
    df.columns = ["time"] + SYMBOLS
    
    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"])
    
    # Remove rows with missing data
    df = df.dropna()
    
    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)
    
    print(f"Loaded {len(df)} ticks from {df['time'].min()} to {df['time'].max()}")
    
    return df


def aggregate_to_m1(ticks: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Aggregate tick data to M1 (1-minute) OHLC candles
    
    Args:
        ticks: DataFrame with tick data
        symbol: Currency pair symbol
        
    Returns:
        DataFrame with M1 OHLC candles
    """
    # Create a copy and set time as index
    df = ticks[["time", symbol]].copy()
    df = df.set_index("time")
    
    # Resample to 1-minute
    ohlc = df[symbol].resample("1min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_count": "count"
    })
    
    # Calculate micro-structure features from ticks
    tick_volatility = df[symbol].resample("1min").std()
    ohlc["tick_volatility"] = tick_volatility
    
    # Range in pips
    pip_mult = PIP_MULTIPLIERS.get(symbol, 10000)
    ohlc["range_pips"] = (ohlc["high"] - ohlc["low"]) * pip_mult
    
    # Drop rows with NaN (empty minutes)
    ohlc = ohlc.dropna()
    
    # Reset index to make time a column
    ohlc = ohlc.reset_index()
    ohlc = ohlc.rename(columns={"time": "timestamp"})
    
    print(f"  {symbol}: {len(ohlc)} M1 candles created")
    
    return ohlc


def prepare_all_symbols(ticks: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare M1 candles for all symbols
    
    Args:
        ticks: DataFrame with all tick data
        
    Returns:
        Dictionary mapping symbol to M1 DataFrame
    """
    candles = {}
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        candles[symbol] = aggregate_to_m1(ticks, symbol)
    
    return candles


def get_recent_ticks(csv_path: str, minutes: int = 35) -> pd.DataFrame:
    """
    Load only the recent ticks needed for prediction
    
    Args:
        csv_path: Path to CSV file
        minutes: Number of minutes of history to load
        
    Returns:
        DataFrame with recent ticks
    """
    # Load last N lines efficiently
    df = load_tick_data(csv_path)
    
    # Filter to last X minutes
    cutoff = df["time"].max() - pd.Timedelta(minutes=minutes)
    recent = df[df["time"] >= cutoff]
    
    return recent


if __name__ == "__main__":
    # Test the pipeline
    from config import DATA_FILE
    
    ticks = load_tick_data(DATA_FILE)
    candles = prepare_all_symbols(ticks)
    
    print("\nSample EURUSD candles:")
    print(candles["EURUSD"].head())
