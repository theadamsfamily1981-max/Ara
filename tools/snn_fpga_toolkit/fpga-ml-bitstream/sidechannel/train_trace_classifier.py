#!/usr/bin/env python3
"""
train_trace_classifier.py

Train a 1D CNN to classify power traces as "clean" vs "trojan" (or more classes).

Assumptions:
    - processed_traces/ contains .npy files named like:
        clean_00000.npy, trojan_00001.npy, ...
    - Label = prefix before first underscore.

Usage:
    python sidechannel/train_trace_classifier.py \
        --traces-dir processed_traces \
        --output models/trace_cnn.pt
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TraceDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.npy"))
        # Derive labels from filename prefix
        labels = sorted({p.name.split("_")[0] for p in self.paths})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        trace = np.load(path).astype(np.float32)
        x = torch.from_numpy(trace)[None, :]  # (1, L)
        label_str = path.name.split("_")[0]
        y = self.label_to_idx[label_str]
        return x, y


class TraceCNN(nn.Module):
    """1D CNN for power trace classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    ds = TraceDataset(args.traces_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceCNN(num_classes=len(ds.label_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        print(
            f"Epoch {epoch}: loss={total_loss/total:.4f}, "
            f"acc={correct/total:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "label_to_idx": ds.label_to_idx},
        args.output,
    )
    print(f"Saved trace model to {args.output}")


if __name__ == "__main__":
    main()
