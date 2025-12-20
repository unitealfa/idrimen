"""
Live Prediction Module - Mode 2 (PyTorch Version)
Makes predictions every 10 minutes and validates them
"""

import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SYMBOLS, DATA_FILE, NEW_DATA_FILE, MODELS_DIR,
    SEQUENCE_LENGTH, PREDICTION_HORIZON, PIP_MULTIPLIERS,
    CONFIDENCE_THRESHOLD, REWARD_RISK_RATIO, DIRECTION_THRESHOLD_PIPS,
    NUM_FEATURES
)
from data_pipeline import load_tick_data, aggregate_to_m1
from feature_engineering import build_features
from label_generator import get_direction_from_prices
from sequence_generator import create_prediction_sequence
from lstm_model import load_model


# Global tracking
predictions_history = {}
stats = {"success": 0, "error": 0, "flat": 0}


def decode_direction(direction_probs):
    """Decode direction from model output"""
    direction_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
    return direction_map[np.argmax(direction_probs)]


def load_all_models():
    """Load all trained models"""
    models = {}
    
    for symbol in SYMBOLS:
        model_path = os.path.join(MODELS_DIR, f"{symbol}_model.pt")
        
        if os.path.exists(model_path):
            try:
                models[symbol] = load_model(model_path, NUM_FEATURES)
                print(f"   ✅ {symbol} modèle chargé")
            except Exception as e:
                print(f"   ❌ {symbol} erreur: {e}")
        else:
            print(f"   ⚠️ {symbol} pas de modèle trouvé")
    
    return models


def get_current_prices():
    """Get current prices from most recent data"""
    try:
        df = pd.read_csv(DATA_FILE, header=None, names=["time"] + SYMBOLS)
        df = df.dropna().tail(1)
        
        if len(df) > 0:
            prices = {symbol: df[symbol].iloc[-1] for symbol in SYMBOLS}
            return prices
    except Exception as e:
        pass
    
    return {}


def make_prediction(model, symbol: str, recent_data: pd.DataFrame) -> dict:
    """Make a prediction for a single symbol"""
    pip_mult = PIP_MULTIPLIERS.get(symbol, 10000)
    
    try:
        # Aggregate to M1
        m1 = aggregate_to_m1(recent_data, symbol)
        
        if len(m1) < SEQUENCE_LENGTH:
            return None
        
        # Build features
        features = build_features(m1, symbol)
        
        # Create sequence
        X = create_prediction_sequence(features)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Predict
        model.eval()
        with torch.no_grad():
            price_pred, direction_probs, _, max_up, max_down = model(X_tensor)
        
        # Convert to numpy
        price_pred = price_pred.numpy()[0][0]
        direction_probs = direction_probs.numpy()[0]
        max_up = max_up.numpy()[0][0]
        max_down = max_down.numpy()[0][0]
        
        # Decode direction
        direction = decode_direction(direction_probs)
        confidence = float(np.max(direction_probs))
        
        # Current price
        current_price = features["close"].iloc[-1]
        
        # Trading signal
        signal = determine_signal(direction, confidence, max_up, max_down)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": float(price_pred),
            "direction": direction,
            "direction_probs": direction_probs.tolist(),
            "max_up_pips": float(max_up),
            "max_down_pips": float(max_down),
            "signal": signal,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        print(f"   ⚠️ Erreur prédiction {symbol}: {e}")
        return None


def determine_signal(direction: str, confidence: float, max_up: float, max_down: float) -> str:
    """Determine trading signal based on prediction"""
    if direction == "UP" and confidence > CONFIDENCE_THRESHOLD:
        if max_up > max_down * REWARD_RISK_RATIO:
            return "BUY"
    elif direction == "DOWN" and confidence > CONFIDENCE_THRESHOLD:
        if max_down > max_up * REWARD_RISK_RATIO:
            return "SELL"
    
    return "HOLD"


def validate_predictions(current_time: datetime, current_prices: dict):
    """Validate predictions made 10 minutes ago"""
    global stats
    
    validation_cutoff = current_time - timedelta(minutes=PREDICTION_HORIZON)
    
    for pred_time, predictions in list(predictions_history.items()):
        if pred_time <= validation_cutoff:
            print(f"\n{'='*50}")
            print(f"📋 VALIDATION des prédictions de {pred_time.strftime('%H:%M:%S')}")
            print("="*50)
            
            for symbol, pred in predictions.items():
                if symbol not in current_prices:
                    continue
                
                actual_price = current_prices[symbol]
                actual_direction = get_direction_from_prices(
                    pred["current_price"], 
                    actual_price, 
                    symbol,
                    DIRECTION_THRESHOLD_PIPS
                )
                
                pip_mult = PIP_MULTIPLIERS.get(symbol, 10000)
                actual_change_pips = (actual_price - pred["current_price"]) * pip_mult
                
                if pred["direction"] == actual_direction:
                    if actual_direction == "FLAT":
                        print(f"   ➖ {symbol}: FLAT (mouvement {actual_change_pips:+.1f} pips)")
                        stats["flat"] += 1
                    else:
                        print(f"   ✅ RÉUSSI - {symbol}")
                        print(f"      Prédit: {pred['direction']} | Réel: {actual_direction} ({actual_change_pips:+.1f} pips)")
                        stats["success"] += 1
                else:
                    print(f"   ❌ ERREUR - {symbol}")
                    print(f"      Prédit: {pred['direction']} | Réel: {actual_direction} ({actual_change_pips:+.1f} pips)")
                    stats["error"] += 1
            
            del predictions_history[pred_time]


def display_stats():
    """Display current statistics"""
    total = stats["success"] + stats["error"]
    
    if total > 0:
        win_rate = stats["success"] / total
        print(f"\n📈 STATISTIQUES: {stats['success']}✅ / {stats['error']}❌ / {stats['flat']}➖")
        print(f"   Taux de réussite: {win_rate:.1%}")


def run_prediction_loop():
    """Main prediction loop - runs every 10 minutes"""
    print("\n" + "="*60)
    print("   🔮 MODE PRÉDICTION ACTIVÉ (PyTorch)")
    print("="*60)
    print("\n   Prédictions toutes les 10 minutes avec validation")
    print("   Appuyez Ctrl+C pour arrêter")
    
    # Load models
    print("\n📂 Chargement des modèles...")
    models = load_all_models()
    
    if not models:
        print("\n❌ Aucun modèle trouvé! Lancez d'abord l'entraînement (Option 1).")
        input("\nAppuyez sur Entrée pour continuer...")
        return
    
    available_symbols = list(models.keys())
    print(f"\n   Paires disponibles: {', '.join(available_symbols)}")
    
    try:
        while True:
            current_time = datetime.now()
            
            # 1. Validate past predictions
            current_prices = get_current_prices()
            if predictions_history and current_prices:
                validate_predictions(current_time, current_prices)
            
            display_stats()
            
            # 2. Load recent data
            print(f"\n{'='*60}")
            print(f"📊 NOUVELLES PRÉDICTIONS - {current_time.strftime('%H:%M:%S')}")
            print("="*60)
            
            try:
                ticks = load_tick_data(DATA_FILE)
                cutoff = ticks["time"].max() - pd.Timedelta(minutes=40)
                recent_ticks = ticks[ticks["time"] >= cutoff]
            except Exception as e:
                print(f"   ❌ Erreur chargement données: {e}")
                time.sleep(60)
                continue
            
            # 3. Make predictions
            current_predictions = {}
            
            for symbol in available_symbols:
                prediction = make_prediction(models[symbol], symbol, recent_ticks)
                
                if prediction:
                    current_predictions[symbol] = prediction
                    
                    print(f"\n   📊 {symbol}:")
                    print(f"      Prix actuel: {prediction['current_price']:.5f}")
                    print(f"      Prix prédit (+10min): {prediction['predicted_price']:.5f}")
                    print(f"      Direction: {prediction['direction']} (confiance: {prediction['confidence']:.0%})")
                    print(f"      Variation max: ↑{prediction['max_up_pips']:.1f} pips / ↓{prediction['max_down_pips']:.1f} pips")
                    
                    signal = prediction["signal"]
                    if signal == "BUY":
                        print(f"      🟢 Signal: {signal}")
                    elif signal == "SELL":
                        print(f"      🔴 Signal: {signal}")
                    else:
                        print(f"      ⚪ Signal: {signal}")
            
            if current_predictions:
                predictions_history[current_time] = current_predictions
            
            # 4. Wait for next cycle
            next_time = current_time + timedelta(minutes=PREDICTION_HORIZON)
            print(f"\n⏳ Prochaine prédiction à {next_time.strftime('%H:%M:%S')}")
            print("   (Ctrl+C pour arrêter)")
            
            for _ in range(PREDICTION_HORIZON * 2):
                time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("   🛑 ARRÊT DU MODE PRÉDICTION")
        print("="*60)
        
        display_stats()
        
        print("\n" + "="*60)
        input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    run_prediction_loop()
