"""
Phase 2.2: Lightweight ML Model for Tackle Detection

Input Features (from scene graph context):
  - contact_confidence: strength of raider-defender contact
  - nearby_defender_count: number of defenders within 1.1m
  - raider_speed: magnitude of raider velocity vector
  - containment_angle: max encirclement angle from containment factors

Output:
  - P(tackle): Probability that defenders successfully caught the raider

Architecture:
  - 2 hidden layers (16 → 8 units)
  - ReLU activation, minimal dropout
  - ~300 parameters total
  - <1ms inference per frame
  
This model learns when contact actually translates to a successful tackle,
fixing the #1 failure mode: geometric tackle logic overfitting on defender grouping.
"""

import torch
import torch.nn as nn
import numpy as np


class LightweightTackleModel(nn.Module):
    """
    Lightweight neural network for tackle detection from graph features.
    Input: [contact_conf, nearby_count, raider_speed, containment_angle]
    Output: P(tackle)
    """
    
    def __init__(self, input_dim=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
            nn.Sigmoid(),  # Output: P(tackle) ∈ [0, 1]
        )
        
    def forward(self, features):
        """
        Args:
            features: Tensor of shape [batch, 4] or [4] (single sample)
        
        Returns:
            tackle_prob: Probability of successful tackle
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(features, (list, tuple)):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        return self.net(features).squeeze(-1)
    
    def predict_single(self, contact_conf, nearby_count, raider_speed, containment_angle):
        """
        Convenience method for single-frame inference during reasoning.
        
        Args:
            contact_conf (float): Contact confidence [0, 1]
            nearby_count (int): Number of nearby defenders
            raider_speed (float): Raider velocity magnitude
            containment_angle (float): Max containment angle [0, 1]
        
        Returns:
            tackle_prob (float): P(tackle) ∈ [0, 1]
        """
        features = torch.tensor(
            [contact_conf, nearby_count / 7.0, raider_speed / 0.5, containment_angle],
            dtype=torch.float32
        )
        with torch.no_grad():
            prob = self.forward(features).item()
        return float(prob)


def create_tackle_model(pretrained_path=None):
    """
    Factory function to instantiate and optionally load a pretrained tackle model.
    
    Args:
        pretrained_path (str, optional): Path to checkpoint. If provided, load weights.
    
    Returns:
        model: LightweightTackleModel instance
    """
    model = LightweightTackleModel()
    
    if pretrained_path:
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained tackle model from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {pretrained_path}: {e}")
    
    return model


# ============================================================================
# Training utilities (for Phase 2.2 training script)
# ============================================================================

class TackleDataset(torch.utils.data.Dataset):
    """
    Dataset for tackle detection training.
    
    Format: CSV with columns:
        contact_conf, nearby_count, raider_speed, containment_angle, label
    
    label: 1 if tackle occurred, 0 otherwise
    """
    
    def __init__(self, csv_path, normalize=True):
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.normalize = normalize
        
        if normalize:
            # Normalize numeric features to [0, 1]
            self.df["nearby_count"] = self.df["nearby_count"] / 7.0  # Max ~7 defenders
            self.df["raider_speed"] = self.df["raider_speed"] / 0.5  # Typical max speed
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor([
            row["contact_conf"],
            row["nearby_count"],
            row["raider_speed"],
            row["containment_angle"],
        ], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return features, label


def get_tackle_dataloader(csv_path, batch_size=32, shuffle=True, normalize=True):
    """
    Create a DataLoader for tackle detection training.
    """
    dataset = TackleDataset(csv_path, normalize=normalize)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Quick test
    print("\nTesting LightweightTackleModel...")
    
    model = create_tackle_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test single inference
    prob = model.predict_single(
        contact_conf=0.65,
        nearby_count=3,
        raider_speed=0.30,
        containment_angle=0.72,
    )
    print(f"Example tackle probability: {prob:.4f}")
    
    # Test batch inference
    batch = torch.randn(8, 4)
    with torch.no_grad():
        batch_probs = model(batch)
    print(f"Batch prediction shape: {batch_probs.shape}")
    print(f"Batch predictions: {batch_probs.numpy()}")
    
    print("✓ Model tests passed!\n")
