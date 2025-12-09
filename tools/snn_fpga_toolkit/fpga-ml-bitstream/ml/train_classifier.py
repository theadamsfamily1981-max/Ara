#!/usr/bin/env python3
"""
train_classifier.py

Train BitstreamCNN on a directory of PNG images and a labels CSV.

Assumptions:
    - images_root/
        - <id>.png
        - ...
    - labels.csv with columns: id,label
        where `id` matches the PNG basename without extension.

Usage:
    python ml/train_classifier.py \
        --images-root dataset/images \
        --labels-file dataset/labels.csv \
        --output models/cnn_bitstream.pt
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import Tuple
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.cnn_bitstream import BitstreamCNN


class BitstreamImageDataset(Dataset):
    def __init__(self, images_root: Path, labels_path: Path, transform=None):
        self.images_root = images_root
        self.labels = pd.read_csv(labels_path)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # converts to [0,1], (C,H,W)
        ])

        # Map labels to integers
        unique_labels = sorted(self.labels["label"].unique())
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.labels["label_idx"] = self.labels["label"].map(self.label_to_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = self.images_root / f"{row['id']}.png"
        img = Image.open(img_path).convert("L")  # grayscale
        x = self.transform(img)
        y = int(row["label_idx"])
        return x, y


def train(
    model: BitstreamCNN,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        print(
            f"Epoch {epoch}: "
            f"loss={running_loss/total:.4f}, acc={correct/total:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BitstreamImageDataset(args.images_root, args.labels_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = BitstreamCNN(num_classes=len(ds.label_to_idx))
    train(model, loader, device=device, epochs=args.epochs, lr=args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": ds.label_to_idx,
        },
        args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
