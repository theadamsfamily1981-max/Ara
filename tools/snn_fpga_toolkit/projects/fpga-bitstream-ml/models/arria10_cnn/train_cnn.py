#!/usr/bin/env python3
"""
ResNet-50 Training for Arria 10 Bitstream Trojan Detection

Trains a Convolutional Neural Network on entropy map images to classify
bitstreams as clean (0) or infected (1) with Hardware Trojans.

Architecture: ResNet-50 (pretrained on ImageNet, fine-tuned)
Input: Grayscale entropy maps (1-channel images)
Output: Binary classification (Trojan probability)

Performance expectations (from research):
- Baseline accuracy: 85-95% on synthetic data
- False positive rate: <5%
- Inference time: ~200ms per bitstream (CPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
import sys

from dataset import Arria10BitstreamDataset, get_default_transforms, create_dataloaders


class TrojanDetectorCNN(nn.Module):
    """ResNet-50 based binary classifier for Hardware Trojan detection."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze ResNet backbone (only train classifier head)
        """
        super().__init__()

        # Load ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Modify first conv layer for grayscale input (1 channel instead of 3)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,  # 1 input channel (grayscale)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize new conv1 from pretrained weights (average across RGB)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),  # Binary output
            nn.Sigmoid()  # Output probability in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, H, W)

        Returns:
            Tensor of shape (batch_size, 1) with Trojan probabilities
        """
        return self.backbone(x)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # Shape: (batch, 1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate for one epoch."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Collect for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    # Compute confusion matrix
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    tp = ((all_predictions == 1) & (all_labels == 1)).sum().item()
    tn = ((all_predictions == 0) & (all_labels == 0)).sum().item()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum().item()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }


def train_model(
    data_root: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'auto'
) -> dict:
    """
    Complete training pipeline.

    Args:
        data_root: Root directory with clean/ and infected/ subdirs
        output_path: Path to save trained model (.pt)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam optimizer
        val_split: Fraction of data for validation
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze ResNet backbone
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        Training history dictionary
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print(f"Training on device: {device}")
    print(f"Using {'pretrained' if pretrained else 'random'} ResNet-50")
    print(f"Backbone: {'frozen' if freeze_backbone else 'trainable'}")
    print()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_root,
        batch_size=batch_size,
        val_split=val_split
    )

    print(f"Dataset loaded:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print()

    # Create model
    model = TrojanDetectorCNN(pretrained=pretrained, freeze_backbone=freeze_backbone)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    print("Starting training...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])

        # Print summary
        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
        print(f"  Val Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}% | F1: {val_metrics['f1']:.2f}%")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'history': history
            }, output_path)

            print(f"  ✓ Best model saved (val_acc: {best_val_acc:.2f}%)")

        print("=" * 70)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Model saved to: {output_path}")

    return history


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train CNN for FPGA bitstream Trojan detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data',
        required=True,
        help='Root directory with clean/ and infected/ entropy images'
    )

    parser.add_argument(
        '--output',
        default='arria10_cnn.pt',
        help='Output model path (default: arria10_cnn.pt)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split fraction (default: 0.2)'
    )

    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use ImageNet pretrained weights'
    )

    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze ResNet backbone (only train classifier head)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use (default: auto)'
    )

    args = parser.parse_args()

    # Train
    history = train_model(
        data_root=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        device=args.device
    )

    # Save history
    history_path = Path(args.output).with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved to: {history_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
