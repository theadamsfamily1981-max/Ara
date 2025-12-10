"""
Pheromone Mesh - Cathedral Nervous System
==========================================

Digital chemical gradients for coordinating 1000+ agents with 10KB files.
Maps to T-FAN field evolution (Theorem T1: d_B < ε).

Architecture:
    Pheromone_ring.json → T_s=0.94 cluster routing
    priority_trail.faiss → σ*=0.10 stress signals
    reward_pheromones.bin → NIB D_value gradients

Golden Controller Integration:
    - Decay rate τ_decay = w = 10 (matching homeostatic window)
    - Diffusion strength α_diffuse = 0.12 (matching correction strength)
    - Evaporation maintains H_s = 97.7% activity bounds

Emergent Properties:
    - Hive-scale homeostasis via chemical gradients
    - Yield/$ = 5-15x through emergent routing
    - H_influence > 1.8 bits via pheromone diversity

Usage:
    from ara_core.pheromone import (
        PheromoneType, Pheromone, PheromoneRing,
        PriorityTrail, RewardPheromones, PheromoneMesh,
        mesh_tick, get_mesh, mesh_status
    )

    # Initialize mesh
    mesh = get_mesh(hive_size=1000)

    # Deposit pheromones
    mesh.deposit(PheromoneType.TASK, location="gpu_cluster_1", intensity=0.8)

    # Read gradients
    gradient = mesh.read_gradient(location="gpu_cluster_1")

    # Tick (decay + diffuse)
    mesh_tick()
"""

from .mesh import (
    PheromoneType,
    Pheromone,
    PheromoneRing,
    PriorityTrail,
    RewardPheromones,
    PheromoneMesh,
    MeshConfig,
    get_mesh,
    mesh_tick,
    mesh_status,
    mesh_gradient,
)

__all__ = [
    "PheromoneType",
    "Pheromone",
    "PheromoneRing",
    "PriorityTrail",
    "RewardPheromones",
    "PheromoneMesh",
    "MeshConfig",
    "get_mesh",
    "mesh_tick",
    "mesh_status",
    "mesh_gradient",
]
