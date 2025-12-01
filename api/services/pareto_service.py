"""
Pareto optimization service
"""

from pathlib import Path
import json
import uuid
import asyncio
import logging
from typing import Dict, Optional
from ..models.schemas import ParetoWeights, ParetoFrontResponse, ParetoConfig

logger = logging.getLogger(__name__)


class ParetoService:
    """Service for managing Pareto optimization."""

    def __init__(self):
        self.weights = ParetoWeights()
        self.tasks: Dict[str, Dict] = {}
        self.current_front: Optional[ParetoFrontResponse] = None

    def get_weights(self) -> ParetoWeights:
        """Get current optimization weights."""
        return self.weights

    def update_weights(self, weights: ParetoWeights):
        """Update optimization weights."""
        self.weights = weights
        logger.info(f"Updated Pareto weights: {weights.dict()}")

    def get_front(self) -> ParetoFrontResponse:
        """Get current Pareto front."""
        # Try loading from artifacts
        front_file = Path("artifacts/pareto/pareto_front.json")

        if not front_file.exists():
            return ParetoFrontResponse(
                n_pareto_points=0,
                hypervolume=0.0,
                configurations=[]
            )

        try:
            with open(front_file) as f:
                data = json.load(f)

            configurations = [
                ParetoConfig(**config)
                for config in data.get("configurations", [])
            ]

            return ParetoFrontResponse(
                n_pareto_points=data.get("n_pareto_points", 0),
                hypervolume=data.get("hypervolume", 0.0),
                configurations=configurations
            )
        except Exception as e:
            logger.error(f"Error loading Pareto front: {e}")
            return ParetoFrontResponse(
                n_pareto_points=0,
                hypervolume=0.0,
                configurations=[]
            )

    def run_optimization_async(self, n_iterations: int, n_initial: int) -> str:
        """Run Pareto optimization asynchronously."""
        task_id = str(uuid.uuid4())

        self.tasks[task_id] = {
            "status": "running",
            "n_iterations": n_iterations,
            "n_initial": n_initial,
            "progress": 0
        }

        # Start background task
        asyncio.create_task(self._run_optimization_task(task_id, n_iterations, n_initial))

        return task_id

    async def _run_optimization_task(self, task_id: str, n_iterations: int, n_initial: int):
        """Background task for running optimization."""
        try:
            # Import here to avoid circular imports
            from tfan.pareto_v2 import ParetoRunner, ParetoRunnerConfig

            config = ParetoRunnerConfig(
                n_initial_points=n_initial,
                n_iterations=n_iterations,
                output_dir="artifacts/pareto"
            )

            runner = ParetoRunner(config)

            # Run optimization
            front = await asyncio.to_thread(runner.run, verbose=False)

            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["hypervolume"] = float(front.hypervolume)
            self.tasks[task_id]["n_points"] = front.n_dominated

            logger.info(f"Pareto optimization completed: {task_id}")

        except Exception as e:
            logger.error(f"Pareto optimization failed: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of optimization task."""
        return self.tasks.get(task_id)

    def get_best_config_with_weights(self) -> Dict:
        """Get best config using current weights."""
        try:
            from tfan.pareto_v2 import ParetoRunner

            runner = ParetoRunner()
            runner.load_results("artifacts/pareto")

            best = runner.get_best_config(self.weights.dict())
            return best
        except Exception as e:
            logger.error(f"Error getting best config: {e}")
            return {}
