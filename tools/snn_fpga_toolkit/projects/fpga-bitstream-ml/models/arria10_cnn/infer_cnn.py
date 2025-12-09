#!/usr/bin/env python3
"""
CNN Inference for Single Bitstream Analysis

Loads a trained ResNet-50 model and classifies a single entropy map image
as clean (0) or infected (1) with Hardware Trojan.

Usage:
    python infer_cnn.py --model arria10_cnn.pt --image entropy.png
    python infer_cnn.py --model arria10_cnn.pt --image entropy.png --threshold 0.9
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import json
import sys
from pathlib import Path

from train_cnn import TrojanDetectorCNN


def load_model(model_path: str, device: str = 'auto') -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to .pt checkpoint
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        Tuple of (model, device, checkpoint_info)
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = TrojanDetectorCNN(pretrained=False)  # Weights will be loaded from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Extract info
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_acc': checkpoint.get('val_acc', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown')
    }

    return model, device, info


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess entropy map image.

    Args:
        image_path: Path to entropy PNG

    Returns:
        Tensor of shape (1, 1, 224, 224) ready for model input
    """
    # Load image
    image = Image.open(image_path).convert('L')  # Grayscale

    # Apply transforms (same as validation in training)
    transform = transforms.Compose([
        transforms.CenterCrop(224),  # Crop to 224x224
        transforms.ToTensor(),  # Convert to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Transform and add batch dimension
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 1, 224, 224)

    return tensor


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Run inference on a single image.

    Args:
        model: Trained TrojanDetectorCNN
        image_tensor: Preprocessed image tensor
        device: Device to run on

    Returns:
        Tuple of (probability, prediction)
        - probability: Float in [0, 1] (Trojan confidence)
        - prediction: 0 (clean) or 1 (infected)
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)  # Shape: (1, 1)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0

    return probability, prediction


def analyze_bitstream(
    model_path: str,
    image_path: str,
    threshold: float = 0.5,
    device: str = 'auto',
    verbose: bool = True
) -> dict:
    """
    Complete bitstream analysis pipeline.

    Args:
        model_path: Path to trained model (.pt)
        image_path: Path to entropy map image
        threshold: Classification threshold (default 0.5)
        device: Device to use
        verbose: Print detailed results

    Returns:
        Analysis results dictionary
    """
    if verbose:
        print("Loading model...")

    # Load model
    model, device, model_info = load_model(model_path, device)

    if verbose:
        print(f"  Model: {model_path}")
        print(f"  Trained for: {model_info['epoch']} epochs")
        print(f"  Val accuracy: {model_info['val_acc']}")
        print(f"  Device: {device}")
        print()

    # Load and preprocess image
    if verbose:
        print("Loading bitstream entropy map...")

    image_tensor = preprocess_image(image_path)

    if verbose:
        print(f"  Image: {image_path}")
        print(f"  Tensor shape: {image_tensor.shape}")
        print()

    # Predict
    if verbose:
        print("Running inference...")

    probability, prediction = predict(model, image_tensor, device)

    # Determine label
    if prediction == 0:
        label = "CLEAN"
        confidence = 1.0 - probability
    else:
        label = "INFECTED"
        confidence = probability

    # Prepare results
    results = {
        'image_path': str(image_path),
        'model_path': str(model_path),
        'trojan_probability': float(probability),
        'prediction': int(prediction),
        'label': label,
        'confidence': float(confidence),
        'threshold': threshold,
        'model_info': model_info
    }

    # Print results
    if verbose:
        print()
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Prediction:          {label}")
        print(f"Trojan Probability:  {probability:.4f} ({probability*100:.2f}%)")
        print(f"Confidence:          {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Threshold:           {threshold}")
        print("=" * 60)
        print()

        if prediction == 1:
            print("⚠ WARNING: Bitstream classified as INFECTED")
            print("   Hardware Trojan detected with high probability.")
            print("   Recommendations:")
            print("   - Do NOT flash this bitstream to hardware")
            print("   - Review build logs for anomalies")
            print("   - Verify source HDL and synthesis settings")
            print("   - Consider re-compiling with different seed")
        else:
            print("✓ Bitstream appears CLEAN")
            print("   No Hardware Trojan detected.")
            print("   Safe to proceed with deployment.")

        print()

    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze FPGA bitstream entropy map for Hardware Trojans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python infer_cnn.py --model arria10_cnn.pt --image entropy.png

  # Higher confidence threshold
  python infer_cnn.py --model arria10_cnn.pt --image entropy.png --threshold 0.9

  # Save results to JSON
  python infer_cnn.py --model arria10_cnn.pt --image entropy.png --json results.json

  # Batch analysis
  for img in data/images/*.png; do
      python infer_cnn.py --model arria10_cnn.pt --image "$img"
  done
        """
    )

    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model (.pt file)'
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to entropy map image (.png)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '--json',
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Analyze
    results = analyze_bitstream(
        model_path=args.model,
        image_path=args.image,
        threshold=args.threshold,
        device=args.device,
        verbose=not args.quiet
    )

    # Save JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Results saved to: {args.json}")

    # Exit code based on prediction
    # 0 = clean, 1 = infected
    return results['prediction']


if __name__ == '__main__':
    sys.exit(main())
