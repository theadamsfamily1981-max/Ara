"""
Metrics service for T-FAN API
"""

from pathlib import Path
import json
import logging
from ..models.schemas import MetricsResponse

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for managing training metrics."""

    def __init__(self, metrics_file: Path = None):
        if metrics_file is None:
            metrics_file = Path.home() / ".cache/tfan/metrics.json"
        self.metrics_file = metrics_file
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def get_current_metrics(self) -> MetricsResponse:
        """Get current metrics from file."""
        if not self.metrics_file.exists():
            return MetricsResponse()

        try:
            with open(self.metrics_file) as f:
                data = json.load(f)
            return MetricsResponse(**data)
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return MetricsResponse()

    def update_metrics(self, metrics: dict):
        """Update metrics file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
