import torch
import torch.nn as nn
from torchvision import models


class TouchWindowClassifier(nn.Module):
    """
    Frame encoder + temporal average pooling baseline for touch confirmation.
    Input shape: [B, T, C, H, W]
    """

    def __init__(self, num_classes=2, dropout=0.2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, clip):
        batch_size, time_steps, channels, height, width = clip.shape
        flattened = clip.view(batch_size * time_steps, channels, height, width)
        frame_features = self.backbone(flattened)
        frame_features = frame_features.view(batch_size, time_steps, -1)
        pooled = frame_features.mean(dim=1)
        return self.head(pooled)
