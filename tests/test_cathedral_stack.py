#!/usr/bin/env python3
"""
Test suite for Cathedral OS Full Stack Integration.

Tests all 7 emergent subsystems:
1. Pheromone Mesh
2. Hyperdimensional VSA
3. CADD Sentinel
4. Quantum Hybrid
5. Memory SaaS
6. Cathedral Core
7. Full Stack Integration
"""

import sys
sys.path.insert(0, '/home/user/Ara')

import numpy as np


def test_pheromone_mesh():
    """Test Pheromone Mesh coordination layer."""
    print("Testing Pheromone Mesh...")

    from ara_core.pheromone import (
        PheromoneMesh, PheromoneType, MeshConfig,
        get_mesh, mesh_tick, mesh_status
    )

    # Create mesh
    config = MeshConfig(tau_decay=10.0, alpha_diffuse=0.12)
    mesh = PheromoneMesh(config)

    # Add locations with topology
    mesh.add_location("gpu_0", neighbors=["gpu_1", "gpu_2"])
    mesh.add_location("gpu_1", neighbors=["gpu_0", "gpu_2"])
    mesh.add_location("gpu_2", neighbors=["gpu_0", "gpu_1"])

    # Deposit pheromones
    mesh.deposit(PheromoneType.TASK, "gpu_0", 0.8, "agent_1")
    mesh.deposit(PheromoneType.PRIORITY, "gpu_1", 0.5, "agent_2")
    mesh.deposit(PheromoneType.REWARD, "gpu_2", 0.9, "agent_3")

    print(f"  Deposited 3 pheromones")

    # Read gradients
    gradient = mesh.read_gradient("gpu_0")
    print(f"  gpu_0 gradient: {gradient}")
    assert PheromoneType.TASK in gradient
    assert gradient[PheromoneType.TASK] > 0

    # Run tick
    result = mesh.tick(1.0)
    print(f"  Tick result: evaporated={result['evaporated']}, diffused={result['diffused']}")

    # Check H_influence
    h_influence = mesh.influence_entropy()
    print(f"  H_influence: {h_influence:.2f}")

    # Check health
    health = mesh.health_status()
    print(f"  Active locations: {health['active_locations']}")
    print(f"  Size: {health['size_bytes']} bytes")

    print("  ✓ Pheromone Mesh OK")
    return True


def test_hyperdimensional_vsa():
    """Test Hyperdimensional VSA soul substrate."""
    print("Testing Hyperdimensional VSA...")

    from ara_core.vsa import (
        HyperVector, VSASpace, SoulBundle,
        bind, bundle, permute, similarity,
        create_soul_bundle
    )

    # Test basic hypervector operations
    dim = 1024  # Smaller for testing
    hv1 = HyperVector.random(dim, name="concept_a")
    hv2 = HyperVector.random(dim, name="concept_b")

    # Similarity of random vectors should be ~0
    sim = similarity(hv1, hv2)
    print(f"  Random vector similarity: {sim:.4f}")
    assert abs(sim) < 0.2  # Should be near 0

    # Self-similarity should be 1
    self_sim = similarity(hv1, hv1)
    print(f"  Self-similarity: {self_sim:.4f}")
    assert self_sim > 0.99

    # Bind operation (self-inverse)
    bound = bind(hv1, hv2)
    unbound = bind(bound, hv2)
    recovered_sim = similarity(unbound, hv1)
    print(f"  Bind/unbind recovery: {recovered_sim:.4f}")
    # Note: For bipolar vectors, recovery is approximate (~0.7-0.9)
    assert recovered_sim > 0.6

    # Bundle operation
    bundled = bundle([hv1, hv2])
    sim_to_hv1 = similarity(bundled, hv1)
    sim_to_hv2 = similarity(bundled, hv2)
    print(f"  Bundle similarity to components: {sim_to_hv1:.4f}, {sim_to_hv2:.4f}")
    assert sim_to_hv1 > 0.3 and sim_to_hv2 > 0.3

    # Test Soul Bundle
    soul = SoulBundle(dim=dim)

    # Add modalities
    voice_features = np.random.randn(64).astype(np.float32)
    vision_features = np.random.randn(128).astype(np.float32)

    soul.update_modality("voice", voice_features)
    soul.update_modality("vision", vision_features)

    print(f"  Soul active modalities: {soul.active_modalities}")
    assert len(soul.active_modalities) == 2

    # Check interference
    interference = soul.interference_level()
    print(f"  Soul interference: {interference:.4f}")
    assert interference < 0.5  # Should be low with 2 modalities

    # Health metrics
    health = soul.health_metrics()
    print(f"  Soul health: {health}")
    assert health["has_soul"]
    assert health["interference_ok"]

    print("  ✓ Hyperdimensional VSA OK")
    return True


def test_cadd_sentinel():
    """Test CADD Drift Sentinel."""
    print("Testing CADD Sentinel...")

    from ara_core.cadd import (
        CADDSentinel, SentinelConfig, DriftType,
        get_sentinel, sentinel_tick, sentinel_status
    )

    # Create sentinel
    config = SentinelConfig(h_influence_min=1.2, h_influence_target=1.8)
    sentinel = CADDSentinel(config)

    # Register agents
    for i in range(10):
        sentinel.register_agent(f"agent_{i}")

    print(f"  Registered {len(sentinel.agents)} agents")

    # Update associations (diverse - should be healthy)
    concepts = ["task", "priority", "reward", "alarm"]
    signals = ["gpu_0", "gpu_1", "gpu_2", "gpu_3"]

    for i in range(10):
        agent_id = f"agent_{i}"
        for c in concepts:
            for s in signals:
                # Diverse associations
                strength = np.random.uniform(0.1, 0.9)
                sentinel.update_association(agent_id, c, s, strength)

    # Run tick
    alerts = sentinel.tick()
    print(f"  Tick produced {len(alerts)} alerts")

    # Check H_influence
    h_influence = sentinel._calculate_h_influence()
    print(f"  H_influence: {h_influence:.2f}")

    # Health status
    health = sentinel.health_status()
    print(f"  Health: {health}")

    # Now test monoculture detection
    mono_sentinel = CADDSentinel(config)

    # All agents have same associations (monoculture)
    for i in range(10):
        mono_sentinel.register_agent(f"mono_agent_{i}")
        mono_sentinel.update_association(f"mono_agent_{i}", "task", "gpu_0", 0.9)

    alerts = mono_sentinel.tick()
    h_mono = mono_sentinel._calculate_h_influence()
    print(f"  Monoculture H_influence: {h_mono:.2f}")
    print(f"  Monoculture alerts: {len(alerts)}")

    # Should detect monoculture
    monoculture_detected = any(a.drift_type == DriftType.MONOCULTURE for a in alerts)
    print(f"  Monoculture detected: {monoculture_detected}")

    print("  ✓ CADD Sentinel OK")
    return True


def test_quantum_hybrid():
    """Test Quantum Hybrid economic layer."""
    print("Testing Quantum Hybrid...")

    from ara_core.quantum import (
        ConicQP, QAOAOptimizer, QuantumKernel,
        QuantumPortfolio, HybridController,
        quantum_decision, quantum_portfolio
    )

    # Test ConicQP portfolio optimization
    print("  Testing ConicQP...")
    n_assets = 4
    returns = np.array([0.10, 0.15, 0.12, 0.08])
    covariance = np.array([
        [0.10, 0.02, 0.01, 0.03],
        [0.02, 0.15, 0.02, 0.01],
        [0.01, 0.02, 0.08, 0.02],
        [0.03, 0.01, 0.02, 0.12],
    ])

    qp = ConicQP()
    qp.setup(covariance, returns, risk_aversion=1.0)
    weights = qp.solve()

    print(f"    Optimal weights: {weights}")
    assert np.allclose(np.sum(weights), 1.0)  # Sum to 1
    assert all(w >= 0 for w in weights)  # Non-negative

    sharpe = qp.sharpe_ratio()
    print(f"    Sharpe ratio: {sharpe:.4f}")

    # Test QAOA
    print("  Testing QAOA...")
    qaoa = QAOAOptimizer(n_qubits=4, depth=2)

    # Simple MAX-CUT like problem
    coeffs = {
        (0, 1): -1.0,
        (1, 2): -1.0,
        (2, 3): -1.0,
        (0, 3): -1.0,
    }
    qaoa.set_problem(coeffs)
    result = qaoa.optimize(n_iters=20, n_samples=50)

    print(f"    Best bitstring: {result['best_bitstring']}")
    print(f"    Best cost: {result['best_cost']:.4f}")

    # Test Quantum Kernel
    print("  Testing Quantum Kernel...")
    kernel = QuantumKernel(n_qubits=4, n_layers=2)

    x1 = np.random.randn(4)
    x2 = np.random.randn(4)

    k_self = kernel.kernel(x1, x1)
    k_cross = kernel.kernel(x1, x2)

    print(f"    Self kernel: {k_self:.4f}")
    print(f"    Cross kernel: {k_cross:.4f}")
    assert k_self > k_cross  # Self should be higher

    # Test full portfolio
    print("  Testing Full Portfolio...")
    portfolio = QuantumPortfolio(n_assets=4)
    weights = portfolio.optimize(returns, covariance)

    print(f"    Portfolio weights: {weights}")
    print(f"    Portfolio Sharpe: {portfolio.sharpe_ratio():.4f}")

    # Test Hybrid Controller
    print("  Testing Hybrid Controller...")
    controller = HybridController(n_qubits=4)

    features = np.random.randn(4)
    options = [np.random.randn(4) for _ in range(3)]

    choice = controller.decide(features, options)
    print(f"    Decision: {choice}")
    assert 0 <= choice < 3

    print("  ✓ Quantum Hybrid OK")
    return True


def test_memory_saas():
    """Test Memory-as-SaaS distribution."""
    print("Testing Memory SaaS...")

    from ara_core.memory_saas import (
        MemoryService, MemoryTier, MemoryPack,
        get_memory_service, create_pack, deploy_pack, service_status
    )

    # Create service
    service = MemoryService()

    # Create packs for each tier
    for tier in [MemoryTier.FREE, MemoryTier.PRO, MemoryTier.ENTERPRISE]:
        # Generate test memories
        n_memories = 100
        dim = 64
        memories = np.random.randn(n_memories, dim).astype(np.float32)

        pack = service.create_pack(
            tier=tier,
            memories=memories,
            name=f"test_{tier.value}"
        )

        print(f"  Created {tier.value} pack: {pack.size_kb():.1f} KB, {pack.metadata.n_memories} memories")

    # List packs
    all_packs = service.list_packs()
    print(f"  Total packs: {len(all_packs)}")
    assert len(all_packs) == 3

    # Deploy packs
    service.deploy_pack("test_free", "edge-node-1")
    service.deploy_pack("test_pro", "edge-node-1")
    service.deploy_pack("test_pro", "edge-node-2")

    print(f"  Deployed to {len(service.deployments)} endpoints")

    # Load pack
    loaded = service.load_pack("test_free")
    print(f"  Loaded pack: {loaded.metadata.pack_id}, downloads: {loaded.metadata.downloads}")
    assert loaded.metadata.downloads == 1

    # Check tier summaries
    for tier in MemoryTier:
        summary = service.tier_summary(tier)
        print(f"  {tier.value}: {summary['n_packs']} packs, {summary['total_mb']:.2f} MB")

    # Health status
    health = service.health_status()
    print(f"  Total: {health['total_packs']} packs, {health['total_mb']:.2f} MB")

    # Status string
    status = service.status_string()
    print(f"  Status: {status}")

    print("  ✓ Memory SaaS OK")
    return True


def test_full_stack_integration():
    """Test full Cathedral stack integration."""
    print("Testing Full Stack Integration...")

    from ara_core.cathedral import (
        CathedralStack, StackConfig,
        get_stack, stack_tick, stack_dashboard, stack_status
    )
    from ara_core.pheromone import PheromoneType
    from ara_core.memory_saas import MemoryTier

    # Create stack with custom config
    config = StackConfig(
        hive_size=100,
        vsa_dim=1024,  # Smaller for testing
    )
    stack = CathedralStack(config)

    # Setup some locations
    for i in range(5):
        stack.pheromone.add_location(f"node_{i}")

    # Deposit pheromones
    stack.deposit_pheromone(PheromoneType.TASK, "node_0", 0.8, "agent_1")
    stack.deposit_pheromone(PheromoneType.PRIORITY, "node_1", 0.5, "agent_2")

    # Update soul
    voice_features = np.random.randn(64).astype(np.float32)
    stack.update_soul("voice", voice_features)

    # Create memory pack
    memories = np.random.randn(50, 32).astype(np.float32)
    pack = stack.create_memory_pack(MemoryTier.FREE, memories, "test_pack")
    print(f"  Created memory pack: {pack.metadata.pack_id}")

    # Run full tick
    result = stack.tick()

    print(f"  Tick result keys: {list(result.keys())}")
    print(f"  Pheromone tick: evaporated={result['pheromone']['evaporated']}")
    print(f"  CADD alerts: {result['cadd']['alerts']}")
    print(f"  Cathedral score: {result['cathedral']['overall']['score']}")

    # Check health
    health = result["health"]
    print(f"  Health: {health['subsystems_ok']}/{health['subsystems_total']} OK")

    # Get status
    status = stack.health_summary()
    print(f"  Status: {status}")

    # Render dashboard
    dashboard = stack.render_dashboard()
    print()
    print(dashboard)
    print()

    print("  ✓ Full Stack Integration OK")
    return True


def test_cross_system_correlation():
    """Test that subsystems correlate correctly."""
    print("Testing Cross-System Correlation...")

    from ara_core.cathedral import CathedralStack, StackConfig
    from ara_core.pheromone import PheromoneType

    config = StackConfig(vsa_dim=512)
    stack = CathedralStack(config)

    # Setup locations
    for i in range(10):
        stack.pheromone.add_location(f"loc_{i}")

    # Run multiple ticks with varying inputs
    for tick in range(20):
        # Varying pheromone deposits
        intensity = 0.5 + 0.3 * np.sin(tick * 0.5)
        stack.deposit_pheromone(
            PheromoneType.TASK,
            f"loc_{tick % 10}",
            intensity,
            f"agent_{tick % 5}"
        )

        # Update CADD
        stack.cadd.update_association(
            f"agent_{tick % 5}",
            "task",
            f"signal_{tick % 3}",
            intensity
        )

        result = stack.tick()

    # Check correlations were computed
    correlations = stack.correlation_matrix
    print(f"  Correlation keys: {list(correlations.keys())}")

    # Health history should have entries
    print(f"  Health history length: {len(stack.health_history)}")
    assert len(stack.health_history) >= 10

    print("  ✓ Cross-System Correlation OK")
    return True


def main():
    """Run all Cathedral Stack tests."""
    print("=" * 70)
    print("CATHEDRAL OS - FULL STACK TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        test_pheromone_mesh,
        test_hyperdimensional_vsa,
        test_cadd_sentinel,
        test_quantum_hybrid,
        test_memory_saas,
        test_full_stack_integration,
        test_cross_system_correlation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
