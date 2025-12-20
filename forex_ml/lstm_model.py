"""
LSTM Model Architecture - PyTorch Version
Multi-output model for price, direction, and max variation prediction
"""

import torch
import torch.nn as nn
import numpy as np
import os

from config import (
    SEQUENCE_LENGTH, NUM_FEATURES,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, BATCH_SIZE
)


class ForexLSTM(nn.Module):
    """
    Multi-output LSTM model for Forex prediction
    
    Outputs:
    - price: future price prediction (1 value)
    - direction: DOWN/FLAT/UP classification (3 classes)
    - max_up: maximum upward movement in pips (1 value)
    - max_down: maximum downward movement in pips (1 value)
    """
    
    def __init__(self, input_size=NUM_FEATURES, hidden1=LSTM_UNITS_1, 
                 hidden2=LSTM_UNITS_2, dense_size=DENSE_UNITS, dropout=DROPOUT_RATE):
        super(ForexLSTM, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Shared dense layer
        self.shared_dense = nn.Linear(hidden2, dense_size)
        self.relu = nn.ReLU()
        
        # Output heads
        self.price_head = nn.Linear(dense_size, 1)
        self.direction_head = nn.Linear(dense_size, 3)  # 3 classes
        self.max_up_head = nn.Linear(dense_size, 1)
        self.max_down_head = nn.Linear(dense_size, 1)
        
        # Softmax for direction
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Take last timestep
        out = out[:, -1, :]
        
        # Shared representation
        shared = self.relu(self.shared_dense(out))
        
        # Output heads
        price = self.price_head(shared)
        direction_logits = self.direction_head(shared)
        direction = self.softmax(direction_logits)
        max_up = self.max_up_head(shared)
        max_down = self.max_down_head(shared)
        
        return price, direction, direction_logits, max_up, max_down


def build_lstm_model(seq_length=SEQUENCE_LENGTH, num_features=NUM_FEATURES):
    """Build and return the model"""
    model = ForexLSTM(input_size=num_features)
    return model


class CombinedLoss(nn.Module):
    """Combined loss for multi-output model"""
    
    def __init__(self, price_weight=0.3, direction_weight=0.4, 
                 max_up_weight=0.15, max_down_weight=0.15):
        super(CombinedLoss, self).__init__()
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.max_up_weight = max_up_weight
        self.max_down_weight = max_down_weight
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        price_pred, _, direction_logits, max_up_pred, max_down_pred = predictions
        price_true, direction_true, max_up_true, max_down_true = targets
        
        # Individual losses
        price_loss = self.mse(price_pred, price_true)
        direction_loss = self.ce(direction_logits, direction_true)
        max_up_loss = self.mse(max_up_pred, max_up_true)
        max_down_loss = self.mse(max_down_pred, max_down_true)
        
        # Combined loss
        total_loss = (
            self.price_weight * price_loss +
            self.direction_weight * direction_loss +
            self.max_up_weight * max_up_loss +
            self.max_down_weight * max_down_loss
        )
        
        return total_loss, {
            'price': price_loss.item(),
            'direction': direction_loss.item(),
            'max_up': max_up_loss.item(),
            'max_down': max_down_loss.item()
        }


def save_model(model, path):
    """Save model to file"""
    torch.save(model.state_dict(), path)
    print(f"   ✅ Modèle sauvegardé: {path}")


def load_model(path, num_features=NUM_FEATURES):
    """Load model from file"""
    model = ForexLSTM(input_size=num_features)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        price_true = y_batch['price'].to(device)
        direction_true = y_batch['direction'].to(device)
        max_up_true = y_batch['max_up'].to(device)
        max_down_true = y_batch['max_down'].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(X_batch)
        targets = (price_true, direction_true, max_up_true, max_down_true)
        
        loss, _ = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Direction accuracy
        _, direction_probs, _, _, _ = predictions
        pred_classes = direction_probs.argmax(dim=1)
        correct += (pred_classes == direction_true).sum().item()
        total += len(direction_true)
    
    return total_loss / len(dataloader), correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            price_true = y_batch['price'].to(device)
            direction_true = y_batch['direction'].to(device)
            max_up_true = y_batch['max_up'].to(device)
            max_down_true = y_batch['max_down'].to(device)
            
            predictions = model(X_batch)
            targets = (price_true, direction_true, max_up_true, max_down_true)
            
            loss, _ = criterion(predictions, targets)
            total_loss += loss.item()
            
            # Direction accuracy
            _, direction_probs, _, _, _ = predictions
            pred_classes = direction_probs.argmax(dim=1)
            correct += (pred_classes == direction_true).sum().item()
            total += len(direction_true)
    
    return total_loss / len(dataloader), correct / total


if __name__ == "__main__":
    # Test model
    print("Testing PyTorch LSTM model...")
    
    model = build_lstm_model()
    print(f"\nModel architecture:\n{model}")
    
    # Test with dummy data
    X_dummy = torch.randn(10, SEQUENCE_LENGTH, NUM_FEATURES)
    price, direction, _, max_up, max_down = model(X_dummy)
    
    print(f"\nTest predictions:")
    print(f"  Price shape: {price.shape}")
    print(f"  Direction shape: {direction.shape}")
    print(f"  Max up shape: {max_up.shape}")
    print(f"  Max down shape: {max_down.shape}")
    print("\n✅ Model test passed!")
