# ara/meta/state_logger.py
"""
State Logger - Logging for Offline Training
============================================

Logs state samples, hypervectors, and latent points for offline training.

This data is used to:
1. Train the PCA/autoencoder for latent space
2. Train the regime classifier
3. Analyze session patterns
4. Build antifragility through learning from history

Storage Format:
- Binary numpy files for hypervectors (efficient)
- JSONL for features + metadata (readable)
- Session directories with timestamps

Usage:
    from ara.meta.state_logger import StateLogger

    logger = StateLogger("/var/ara/logs/state")

    # Log each tick
    logger.log_tick(
        features=sampler_output,
        hv=hypervector,
        z=latent_point,
        regime="FLOW",
    )

    # End session
    logger.end_session()

    # Train from logs
    hvs, labels = logger.load_training_data()
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TickLog:
    """A single tick's worth of logged data."""
    timestamp: str
    features: Dict[str, float]
    regime: Optional[str] = None
    mode: Optional[str] = None
    actions: Optional[List[str]] = None


class StateLogger:
    """
    Logs state data for offline training and analysis.

    Creates session directories with:
    - features.jsonl.gz: Compressed JSONL of tick features
    - hypervectors.npy: Binary numpy array of HVs
    - latent.npy: Binary numpy array of latent points
    - session.json: Session metadata
    """

    def __init__(
        self,
        base_dir: str = "/var/ara/logs/state",
        hv_dim: int = 1024,
        latent_dim: int = 10,
        buffer_size: int = 1000,
        compress: bool = True,
    ):
        """
        Initialize state logger.

        Args:
            base_dir: Base directory for log storage
            hv_dim: Hypervector dimensionality
            latent_dim: Latent space dimensionality
            buffer_size: Ticks to buffer before flush
            compress: Whether to gzip JSONL files
        """
        self.base_dir = Path(base_dir)
        self.hv_dim = hv_dim
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.compress = compress

        # Current session
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._session_start: Optional[datetime] = None

        # Buffers
        self._tick_buffer: List[TickLog] = []
        self._hv_buffer: List[np.ndarray] = []
        self._latent_buffer: List[np.ndarray] = []

        # Stats
        self._total_ticks: int = 0
        self._flush_count: int = 0

        logger.info(f"StateLogger initialized: base_dir={base_dir}")

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new logging session.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        self._session_start = datetime.utcnow()

        if session_id is None:
            session_id = self._session_start.strftime("%Y%m%d_%H%M%S")

        self._session_id = session_id
        self._session_dir = self.base_dir / session_id

        # Create directory
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Clear buffers
        self._tick_buffer = []
        self._hv_buffer = []
        self._latent_buffer = []
        self._total_ticks = 0
        self._flush_count = 0

        # Write session metadata
        metadata = {
            "session_id": session_id,
            "start_time": self._session_start.isoformat(),
            "hv_dim": self.hv_dim,
            "latent_dim": self.latent_dim,
        }
        with open(self._session_dir / "session.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Session started: {session_id}")
        return session_id

    def log_tick(
        self,
        features: Dict[str, float],
        hv: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
        mode: Optional[str] = None,
        actions: Optional[List[str]] = None,
    ) -> None:
        """
        Log a single tick's data.

        Args:
            features: Raw feature dict from StateSampler
            hv: Hypervector (optional)
            z: Latent point (optional)
            regime: Classified regime name
            mode: Operating mode name
            actions: List of action names taken
        """
        if self._session_dir is None:
            self.start_session()

        # Create tick log
        tick = TickLog(
            timestamp=datetime.utcnow().isoformat(),
            features=features,
            regime=regime,
            mode=mode,
            actions=actions,
        )
        self._tick_buffer.append(tick)

        # Buffer vectors
        if hv is not None:
            self._hv_buffer.append(np.asarray(hv))
        if z is not None:
            self._latent_buffer.append(np.asarray(z))

        self._total_ticks += 1

        # Flush if buffer full
        if len(self._tick_buffer) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        """Flush buffers to disk."""
        if not self._tick_buffer:
            return

        if self._session_dir is None:
            return

        # Write features JSONL
        features_file = self._session_dir / f"features_{self._flush_count:04d}.jsonl"
        if self.compress:
            features_file = Path(str(features_file) + ".gz")
            opener = gzip.open
        else:
            opener = open

        with opener(features_file, "wt") as f:
            for tick in self._tick_buffer:
                f.write(json.dumps(asdict(tick)) + "\n")

        # Write hypervectors
        if self._hv_buffer:
            hv_file = self._session_dir / f"hypervectors_{self._flush_count:04d}.npy"
            np.save(hv_file, np.array(self._hv_buffer))

        # Write latent points
        if self._latent_buffer:
            latent_file = self._session_dir / f"latent_{self._flush_count:04d}.npy"
            np.save(latent_file, np.array(self._latent_buffer))

        # Clear buffers
        self._tick_buffer = []
        self._hv_buffer = []
        self._latent_buffer = []
        self._flush_count += 1

        logger.debug(f"Flushed chunk {self._flush_count}")

    def end_session(self) -> Dict[str, Any]:
        """
        End the current session.

        Returns:
            Session summary
        """
        if self._session_dir is None:
            return {}

        # Final flush
        self._flush()

        # Update session metadata
        end_time = datetime.utcnow()
        duration = (end_time - self._session_start).total_seconds() if self._session_start else 0

        metadata = {
            "session_id": self._session_id,
            "start_time": self._session_start.isoformat() if self._session_start else None,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_ticks": self._total_ticks,
            "flush_count": self._flush_count,
            "hv_dim": self.hv_dim,
            "latent_dim": self.latent_dim,
        }

        with open(self._session_dir / "session.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Session ended: {self._session_id}, {self._total_ticks} ticks, {duration:.1f}s")

        # Reset
        session_id = self._session_id
        self._session_id = None
        self._session_dir = None
        self._session_start = None

        return metadata

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []

        if not self.base_dir.exists():
            return sessions

        for session_dir in sorted(self.base_dir.iterdir()):
            if session_dir.is_dir():
                meta_file = session_dir / "session.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        sessions.append(json.load(f))

        return sessions

    def load_session_features(
        self,
        session_id: str,
    ) -> Tuple[List[Dict[str, float]], List[Optional[str]]]:
        """
        Load features from a session.

        Args:
            session_id: Session to load

        Returns:
            (features_list, regime_list)
        """
        session_dir = self.base_dir / session_id
        features_list = []
        regime_list = []

        # Find all feature files
        feature_files = sorted(session_dir.glob("features_*.jsonl*"))

        for fpath in feature_files:
            if fpath.suffix == ".gz":
                opener = gzip.open
            else:
                opener = open

            with opener(fpath, "rt") as f:
                for line in f:
                    tick = json.loads(line)
                    features_list.append(tick["features"])
                    regime_list.append(tick.get("regime"))

        return features_list, regime_list

    def load_session_hypervectors(self, session_id: str) -> np.ndarray:
        """Load hypervectors from a session."""
        session_dir = self.base_dir / session_id
        hv_files = sorted(session_dir.glob("hypervectors_*.npy"))

        if not hv_files:
            return np.zeros((0, self.hv_dim))

        arrays = [np.load(f) for f in hv_files]
        return np.vstack(arrays)

    def load_session_latent(self, session_id: str) -> np.ndarray:
        """Load latent points from a session."""
        session_dir = self.base_dir / session_id
        latent_files = sorted(session_dir.glob("latent_*.npy"))

        if not latent_files:
            return np.zeros((0, self.latent_dim))

        arrays = [np.load(f) for f in latent_files]
        return np.vstack(arrays)

    def load_training_data(
        self,
        session_ids: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Optional[str]]]:
        """
        Load hypervectors and labels for training.

        Args:
            session_ids: Sessions to load (None = all)
            max_samples: Maximum samples to return

        Returns:
            (hypervectors, regime_labels)
        """
        if session_ids is None:
            sessions = self.list_sessions()
            session_ids = [s["session_id"] for s in sessions]

        all_hvs = []
        all_labels = []

        for sid in session_ids:
            hvs = self.load_session_hypervectors(sid)
            _, labels = self.load_session_features(sid)

            if len(hvs) > 0 and len(labels) == len(hvs):
                all_hvs.append(hvs)
                all_labels.extend(labels)

        if not all_hvs:
            return np.zeros((0, self.hv_dim)), []

        combined_hvs = np.vstack(all_hvs)

        # Limit samples if requested
        if max_samples and len(combined_hvs) > max_samples:
            indices = np.random.choice(len(combined_hvs), max_samples, replace=False)
            combined_hvs = combined_hvs[indices]
            all_labels = [all_labels[i] for i in indices]

        return combined_hvs, all_labels


# =============================================================================
# Singleton Access
# =============================================================================

_state_logger: Optional[StateLogger] = None


def get_state_logger(**kwargs) -> StateLogger:
    """Get the default state logger."""
    global _state_logger
    if _state_logger is None:
        _state_logger = StateLogger(**kwargs)
    return _state_logger


# =============================================================================
# Testing
# =============================================================================

def _test_logger():
    """Test state logger."""
    import tempfile

    print("=" * 60)
    print("State Logger Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StateLogger(base_dir=tmpdir, buffer_size=10)

        # Start session
        session_id = logger.start_session()
        print(f"Session: {session_id}")

        # Log some ticks
        np.random.seed(42)
        for i in range(25):
            features = {
                "system.cpu": np.random.rand(),
                "user.stress": np.random.rand(),
            }
            hv = np.random.choice([-1, 1], size=1024)
            z = np.random.randn(10)
            regime = np.random.choice(["FLOW", "IDLE", "BUILDING"])

            logger.log_tick(features=features, hv=hv, z=z, regime=regime)

        # End session
        summary = logger.end_session()
        print(f"Summary: {summary}")

        # List sessions
        sessions = logger.list_sessions()
        print(f"Sessions: {len(sessions)}")

        # Load data
        hvs, labels = logger.load_training_data()
        print(f"Loaded: {hvs.shape}, {len(labels)} labels")
        print(f"Label sample: {labels[:5]}")


if __name__ == "__main__":
    _test_logger()
