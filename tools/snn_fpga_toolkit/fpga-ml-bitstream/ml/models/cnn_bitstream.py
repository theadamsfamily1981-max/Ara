"""
cnn_bitstream.py

Simple CNN for classifying bitstream images.

Goal:
    - Input: grayscale image (H x W), 1 channel.
    - Output: logits for K classes (default K=2: clean vs trojan).

Claude:
    - Implement in PyTorch.
"""

from typing import Tuple
import torch
from torch import nn


class BitstreamCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Claude: feel free to tweak architecture; keep it small/fast.

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
