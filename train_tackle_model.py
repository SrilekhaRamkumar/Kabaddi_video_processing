"""
Phase 2.2: Training Script for Lightweight Tackle Detection Model

This script trains the LightweightTackleModel on labeled context-action pairs.

Input: CSV file with columns:
  - contact_conf: confidence of raider-defender contact [0, 1]
  - nearby_count: number of defenders within 1.1m [0, 7]
  - raider_speed: raider velocity magnitude [0, 0.5]
  - containment_angle: max encirclement angle [0, 1]
  - label: 1 if tackle occurred, 0 otherwise

NOTE: This is a template. You'll need to generate the CSV from Court_code2.py logs.

Usage:
    python train_tackle_model.py --csv path/to/tackle_training_data.csv --output models/tackle_model.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from tackle_model import LightweightTackleModel, TackleDataset


def train_tackle_model(csv_path, output_path="models/tackle_model.pt", epochs=50, batch_size=32, lr=0.001):
    """
    Train the lightweight tackle model.
    
    Args:
        csv_path: Path to training CSV
        output_path: Where to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    print(f"Loading training data from {csv_path}...")
    try:
        dataset = TackleDataset(csv_path, normalize=True)
        print(f"  Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = LightweightTackleModel(input_dim=4, dropout=0.1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {num_params} parameters")
    
    # Loss & optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...\n")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)  # Already returns shape [batch]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
            # Accuracy (threshold at 0.5)
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Learning rate schedule
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (val_loss={best_val_loss:.4f})")
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, output_path)
    print(f"✓ Model saved to {output_path}\n")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight tackle detection model")
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--output", type=str, default="models/tackle_model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PHASE 2.2: TRAINING LIGHTWEIGHT TACKLE DETECTION MODEL")
    print("="*70)
    
    train_tackle_model(
        csv_path=args.csv,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
