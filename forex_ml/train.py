"""
Training Module - Mode 1 (PyTorch Version)
Trains models on existing data and sets up new data collection
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SYMBOLS, DATA_FILE, NEW_DATA_FILE, MODELS_DIR, ARCHIVES_DIR,
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from data_pipeline import load_tick_data, aggregate_to_m1
from feature_engineering import build_features
from label_generator import generate_labels
from sequence_generator import create_sequences_simple
from lstm_model import (
    build_lstm_model, save_model, CombinedLoss, 
    train_epoch, validate_epoch, EarlyStopping
)


def create_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=False):
    """Create PyTorch DataLoader from numpy arrays"""
    
    X_tensor = torch.FloatTensor(X)
    
    # Direction needs to be class indices (not one-hot)
    direction_indices = np.argmax(y['direction'], axis=1)
    
    # Create tensors
    price_tensor = torch.FloatTensor(y['price'])
    direction_tensor = torch.LongTensor(direction_indices)
    max_up_tensor = torch.FloatTensor(y['max_up'])
    max_down_tensor = torch.FloatTensor(y['max_down'])
    
    # Create a custom dataset that returns dict
    class ForexDataset(torch.utils.data.Dataset):
        def __init__(self, X, price, direction, max_up, max_down):
            self.X = X
            self.price = price
            self.direction = direction
            self.max_up = max_up
            self.max_down = max_down
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], {
                'price': self.price[idx],
                'direction': self.direction[idx],
                'max_up': self.max_up[idx],
                'max_down': self.max_down[idx]
            }
    
    dataset = ForexDataset(X_tensor, price_tensor, direction_tensor, 
                           max_up_tensor, max_down_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_symbol(ticks, symbol):
    """Train model for a single symbol"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Aggregate to M1
    m1 = aggregate_to_m1(ticks, symbol)
    
    if len(m1) < SEQUENCE_LENGTH + 20:
        raise ValueError(f"Pas assez de données ({len(m1)} bougies)")
    
    # Build features
    print("   📐 Construction des features...")
    features = build_features(m1, symbol)
    
    # Generate labels
    print("   🏷️ Génération des labels...")
    labeled = generate_labels(features, symbol)
    
    # Create sequences
    print("   📊 Création des séquences...")
    X, y = create_sequences_simple(labeled)
    
    if len(X) < 50:
        raise ValueError(f"Pas assez de séquences ({len(X)})")
    
    # Split data chronologically (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    
    y_train = {k: v[:split_idx] for k, v in y.items()}
    y_val = {k: v[split_idx:] for k, v in y.items()}
    
    print(f"   📈 Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Create dataloaders
    train_loader = create_dataloader(X_train, y_train, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, shuffle=False)
    
    # Build model
    print("   🧠 Construction du modèle LSTM...")
    model = build_lstm_model(num_features=X.shape[2])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping()
    
    # Training loop
    print(f"   🏋️ Entraînement ({EPOCHS} epochs max)...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"      ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss, val_acc


def run_training():
    """
    Main training function - trains models for all currency pairs
    """
    print("\n" + "="*60)
    print("   🔄 DÉMARRAGE DE L'ENTRAÎNEMENT (PyTorch)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    
    # =============================================
    # 1. Load existing data
    # =============================================
    print("\n📂 Chargement des données...")
    
    try:
        ticks = load_tick_data(DATA_FILE)
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return
    
    # Check for new data file and merge
    if os.path.exists(NEW_DATA_FILE):
        try:
            new_ticks = pd.read_csv(NEW_DATA_FILE, header=None)
            if len(new_ticks) > 10:
                new_ticks.columns = ticks.columns
                new_ticks["time"] = pd.to_datetime(new_ticks["time"])
                ticks = pd.concat([ticks, new_ticks], ignore_index=True)
                ticks = ticks.sort_values("time").drop_duplicates()
                print(f"   📊 Fusionné avec {len(new_ticks)} nouvelles données")
        except Exception as e:
            print(f"   ⚠️ Impossible de fusionner nouvelles données: {e}")
    
    print(f"   📊 Total: {len(ticks)} ticks")
    
    # =============================================
    # 2. Train model for each symbol
    # =============================================
    trained_count = 0
    failed_symbols = []
    
    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"🎯 Entraînement {symbol}")
        print("="*50)
        
        try:
            model, val_loss, val_acc = train_symbol(ticks, symbol)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{symbol}_model.pt")
            save_model(model, model_path)
            
            print(f"\n   ✅ {symbol} terminé!")
            print(f"      Val Loss: {val_loss:.4f}")
            print(f"      Val Direction Accuracy: {val_acc:.2%}")
            
            trained_count += 1
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            failed_symbols.append(symbol)
            continue
    
    # =============================================
    # 3. Archive training data
    # =============================================
    print("\n" + "="*50)
    print("📦 Archivage des données...")
    
    archive_name = f"forex_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    archive_path = os.path.join(ARCHIVES_DIR, archive_name)
    ticks.to_csv(archive_path, index=False)
    print(f"   ✅ Données archivées: {archive_name}")
    
    # =============================================
    # 4. Reset new data file
    # =============================================
    pd.DataFrame(columns=["time"] + SYMBOLS).to_csv(NEW_DATA_FILE, index=False)
    print(f"   ✅ Nouveau fichier créé: forex_new_data.csv")
    
    # =============================================
    # Summary
    # =============================================
    print("\n" + "="*60)
    print("   📊 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*60)
    print(f"\n   ✅ Modèles entraînés: {trained_count}/{len(SYMBOLS)}")
    
    if failed_symbols:
        print(f"   ⚠️ Échecs: {', '.join(failed_symbols)}")
    
    print(f"\n   📁 Modèles sauvegardés dans: {MODELS_DIR}")
    print(f"   📁 Nouvelles données iront dans: forex_new_data.csv")
    print("\n" + "="*60)
    
    input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    run_training()
