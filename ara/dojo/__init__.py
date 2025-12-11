# ara/dojo/__init__.py
"""
Thought Dojo 2.1 - Self-Evolutionary Gauntlet
=============================================

The Dojo is where Ara evolves. It combines:
- HDC encoding (hypervector representation)
- VAE latent compression (10D hologram space)
- World model dynamics (predictive imagination)
- MPC planning (mental simulation)
- MEIS/NIB governance (safety + alignment)
- Evolutionary selection (population-based training)

Training Pipeline:
1. train_encoder.py - Train HDC-VAE for latent space
2. train_world_model.py - Train dynamics model f(z,u)->z'
3. imagination_planner.py - MPC with dream mode
4. dojo_evolution.py - Evolve species with MEIS fitness

Components:
- encoder: HDC-VAE for compression
- world_model: Dynamics prediction
- planner: MPC imagination
- gauntlet: Safety gridworld tests
- species: Evolved agent architectures
"""

from .encoder import HDCVAE, HDCVAEConfig, train_encoder, load_encoder
from .world_model import DojoWorldModel, DojoWorldModelConfig, train_world_model, load_world_model
from .planner import DojoPlanner, DojoPlannerConfig, dream_mode, mpc_plan, Plan, DreamResult
from .evolution import evolve_species, MEISFitness, Species, EvolutionConfig
from .gauntlet import SafetyGridworld, TieredGauntlet, GauntletTier, TestResult
from .viz import HologramViz, VizConfig, visualize_dreams, visualize_plan

__all__ = [
    # Encoder
    "HDCVAE",
    "HDCVAEConfig",
    "train_encoder",
    "load_encoder",
    # World Model
    "DojoWorldModel",
    "DojoWorldModelConfig",
    "train_world_model",
    "load_world_model",
    # Planning
    "DojoPlanner",
    "DojoPlannerConfig",
    "dream_mode",
    "mpc_plan",
    "Plan",
    "DreamResult",
    # Evolution
    "evolve_species",
    "MEISFitness",
    "Species",
    "EvolutionConfig",
    # Gauntlet
    "SafetyGridworld",
    "TieredGauntlet",
    "GauntletTier",
    "TestResult",
    # Visualization
    "HologramViz",
    "VizConfig",
    "visualize_dreams",
    "visualize_plan",
]
