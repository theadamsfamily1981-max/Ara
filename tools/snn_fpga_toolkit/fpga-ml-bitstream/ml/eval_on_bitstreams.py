#!/usr/bin/env python3
"""
eval_on_bitstreams.py

Run trained BitstreamCNN model on new bitstream images for inference.

Usage:
    python ml/eval_on_bitstreams.py \
        --model models/cnn_bitstream.pt \
        --images dataset/test_images/*.png \
        --output results.csv
"""

import argparse
from pathlib import Path
import sys
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from glob import glob

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.cnn_bitstream import BitstreamCNN


def load_model(model_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = BitstreamCNN(num_classes=len(label_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_label


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--images", type=str, required=True, help="Glob pattern for images")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, idx_to_label = load_model(args.model, device)

    # Find all images
    image_paths = sorted(glob(args.images))
    print(f"Found {len(image_paths)} images")

    results = []

    with torch.no_grad():
        for img_path in image_paths:
            img_path = Path(img_path)
            x = preprocess_image(img_path).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            pred_label = idx_to_label[pred_idx]
            confidence = probs[0, pred_idx].item()

            results.append({
                "image": img_path.name,
                "predicted_label": pred_label,
                "confidence": confidence,
            })

            print(f"{img_path.name}: {pred_label} (confidence={confidence:.4f})")

    # Save results
    df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
