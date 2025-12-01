"""
Training service for T-FAN API
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict
from ..models.schemas import TrainingRequest, TrainingStatus
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training sessions."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = TrainingStatus(active=False)

    def start_training(self, request: TrainingRequest) -> Dict:
        """Start training session."""
        if self.is_training():
            raise ValueError("Training already active")

        config_file = Path(request.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {request.config_path}")

        # Build command
        cmd = [
            "python",
            "training/train.py",
            "--config", request.config_path,
            "--logdir", request.logdir
        ]

        if request.max_steps:
            cmd.extend(["--max-steps", str(request.max_steps)])

        try:
            # Start training process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.status = TrainingStatus(
                active=True,
                config=request.config_path,
                started_at=datetime.now().isoformat()
            )

            logger.info(f"Training started with config: {request.config_path}")

            return {
                "status": "started",
                "config": request.config_path,
                "pid": self.process.pid
            }

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            raise

    def stop_training(self) -> Dict:
        """Stop current training session."""
        if not self.is_training():
            raise ValueError("No active training")

        try:
            self.process.terminate()
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()

        self.status = TrainingStatus(active=False)
        logger.info("Training stopped")

        return {"status": "stopped"}

    def is_training(self) -> bool:
        """Check if training is active."""
        if self.process is None:
            return False

        # Check if process is still running
        if self.process.poll() is not None:
            # Process ended
            self.status = TrainingStatus(active=False)
            return False

        return True

    def get_status(self) -> TrainingStatus:
        """Get training status."""
        # Update active status
        self.status.active = self.is_training()
        return self.status

    def get_logs(self, lines: int = 100) -> Dict:
        """Get recent training logs."""
        if self.process is None:
            return {"logs": []}

        # Read from stdout
        logs = []
        try:
            if self.process.stdout:
                # Read available lines
                for _ in range(lines):
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    logs.append(line.strip())
        except Exception as e:
            logger.error(f"Error reading logs: {e}")

        return {"logs": logs}
