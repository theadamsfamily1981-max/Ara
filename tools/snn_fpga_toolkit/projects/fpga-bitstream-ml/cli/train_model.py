#!/usr/bin/env python3
"""
CNN Training Orchestration Wrapper

Simplified wrapper around models/arria10_cnn/train_cnn.py for consistent CLI.

Usage:
    python train_model.py --data data/images/arria10 --output models/arria10_cnn.pt
    python train_model.py --data data/images/arria10 --output models/arria10_cnn.pt --epochs 100
"""

import sys
from pathlib import Path

# Add model module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'arria10_cnn'))

from train_cnn import main

if __name__ == '__main__':
    sys.exit(main())
