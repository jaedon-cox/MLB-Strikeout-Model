#!/usr/bin/env python3
"""
Pitcher strikeout model trainer.
Reads the CSV produced by pitcher_data.py and trains a neural network.
"""

# =============================================================================
# PARAMETERS — adjust these before running
# =============================================================================

DATA_FILE     = 'pitcher_training_data.csv'
MODEL_FILE    = 'pitcher_model.pt'
SCALER_FILE   = 'scaler.pkl'

LAYER_1       = 128
LAYER_2       = 64
LAYER_3       = 32
EPOCHS        = 100
PATIENCE      = 10
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
TEST_SIZE     = 0.15
RANDOM_STATE  = 48 # Jacob Degrom's jersey number lol

# =============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_FILE)

print(f"{'=' * 60}")
print(f"DATA LOADED")
print(f"{'=' * 60}")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"\nTarget variable (strikeouts):")
print(df['strikeouts'].describe())

# Drop identifying columns — no predictive value
drop_cols = ['game_id', 'game_date', 'pitcher_id', 'pitcher_name', 'team', 'opponent']
df = df.drop(columns=drop_cols)

print(f"\nAfter dropping identifying columns:")
print(f"Remaining columns : {len(df.columns)}")
print(f"Feature columns   : {[c for c in df.columns if c != 'strikeouts']}")

# =============================================================================
# SPLIT X AND y
# =============================================================================

X = df.drop(columns=['strikeouts']).values
y = df['strikeouts'].values.reshape(-1, 1)

# =============================================================================
# TRAIN / VALIDATION SPLIT
# =============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print(f"\n{'=' * 60}")
print(f"TRAIN / VALIDATION SPLIT")
print(f"{'=' * 60}")
print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# =============================================================================
# NORMALIZE — fit on training data only
# =============================================================================

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

joblib.dump(scaler, SCALER_FILE)
print(f"\nScaler saved to: {SCALER_FILE}")

# =============================================================================
# CONVERT TO TENSORS AND CREATE DATALOADERS
# =============================================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"\n{'=' * 60}")
print(f"DATALOADERS READY")
print(f"{'=' * 60}")
print(f"Training batches  : {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Input features    : {X_train_t.shape[1]}")

# =============================================================================
# DEFINE MODEL
# =============================================================================

class StrikeoutModel(nn.Module):
    def __init__(self, input_size):
        super(StrikeoutModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, LAYER_1),
            nn.ReLU(),
            nn.Linear(LAYER_1, LAYER_2),
            nn.ReLU(),
            nn.Linear(LAYER_2, LAYER_3),
            nn.ReLU(),
            nn.Linear(LAYER_3, 1)
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train_t.shape[1]
model      = StrikeoutModel(input_size)
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n{'=' * 60}")
print(f"MODEL ARCHITECTURE")
print(f"{'=' * 60}")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# =============================================================================
# TRAINING LOOP
# =============================================================================

best_val_loss    = float('inf')
patience_counter = 0

print(f"\n{'=' * 60}")
print(f"TRAINING")
print(f"{'=' * 60}")

for epoch in range(EPOCHS):

    # --- Training ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss        = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)
            val_loss   += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1:>3}/{EPOCHS}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

    # --- Save best model ---
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"           ✓ Best model saved (val loss: {best_val_loss:.4f})")

    # --- Early stopping ---
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered — no improvement for {PATIENCE} epochs")
            break

print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"Best validation loss : {best_val_loss:.4f}")
print(f"Best RMSE            : {best_val_loss ** 0.5:.4f} strikeouts")
print(f"Model saved to       : {MODEL_FILE}")
