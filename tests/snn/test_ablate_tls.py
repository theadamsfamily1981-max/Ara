"""
TLS mask quality ablation tests.

Compares:
1. TLS (Topological Landmark Selection) - α·persistence + (1-α)·diversity
2. Random mask - Uniform random connectivity
3. Degree-based mask - High-degree hubs
4. Local mask - Nearest neighbor connectivity

Metrics:
- Early-epoch loss slope
- Parameter efficiency
- Connectivity quality (diameter, clustering)
"""

import pytest
import torch
import numpy as np

from tfan.snn import (
    LIFLayerLowRank,
    LowRankMaskedSynapse,
    build_tls_mask_from_scores,
    build_uniform_random_mask,
    build_local_plus_random_mask,
    degree_from_csr,
    mask_density,
)


def build_degree_based_mask(N, k_per_row, seed=None):
    """
    Build mask favoring high-degree hub connections.

    Args:
        N: Matrix dimension
        k_per_row: Outgoing degree
        seed: Random seed

    Returns:
        CSR mask dict
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create scores favoring hubs (power-law-ish distribution)
    # High-index neurons are hubs
    hub_scores = torch.arange(N, dtype=torch.float32).pow(2)
    hub_scores = hub_scores / hub_scores.max()

    # Build scores matrix: prefer connections to hubs
    scores = hub_scores.unsqueeze(0).expand(N, N)

    # Add some noise
    scores = scores + torch.rand(N, N) * 0.1

    return build_tls_mask_from_scores(scores, k_per_row=k_per_row)


def compute_graph_diameter(indptr, indices, N):
    """
    Compute approximate graph diameter using BFS.

    Returns average shortest path length as proxy for diameter.
    """
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for i in range(N):
        j0, j1 = indptr[i].item(), indptr[i+1].item()
        neighbors = indices[j0:j1].tolist()
        adj[i] = neighbors

    # Sample random source nodes
    num_samples = min(20, N)
    source_nodes = np.random.choice(N, size=num_samples, replace=False)

    path_lengths = []

    for source in source_nodes:
        # BFS from source
        visited = {source}
        queue = [(source, 0)]
        depths = []

        while queue:
            node, depth = queue.pop(0)
            depths.append(depth)

            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        if depths:
            path_lengths.append(np.mean(depths))

    return np.mean(path_lengths) if path_lengths else float('inf')


def compute_clustering_coefficient(indptr, indices, N):
    """
    Compute local clustering coefficient.

    Measures how well-connected a node's neighbors are.
    """
    # Build adjacency list
    adj = [set() for _ in range(N)]
    for i in range(N):
        j0, j1 = indptr[i].item(), indptr[i+1].item()
        neighbors = indices[j0:j1].tolist()
        adj[i] = set(neighbors)

    clustering_coeffs = []

    for i in range(N):
        neighbors = list(adj[i])
        k = len(neighbors)

        if k < 2:
            continue

        # Count edges among neighbors
        edge_count = 0
        for idx, u in enumerate(neighbors):
            for v in neighbors[idx+1:]:
                if v in adj[u]:
                    edge_count += 1

        # Clustering coefficient for node i
        max_edges = k * (k - 1) / 2
        if max_edges > 0:
            clustering_coeffs.append(edge_count / max_edges)

    return np.mean(clustering_coeffs) if clustering_coeffs else 0.0


def test_tls_vs_random_density():
    """Test that TLS and random masks have same density."""
    N = 512
    k = 32

    # TLS mask
    scores = torch.rand(N, N)  # Random scores for TLS
    tls_mask = build_tls_mask_from_scores(scores, k_per_row=k)

    # Random mask
    random_mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Check densities are equal
    tls_density = mask_density(tls_mask['indptr'], N)
    random_density = mask_density(random_mask['indptr'], N)

    assert abs(tls_density - random_density) < 1e-6, \
        f"Densities differ: TLS={tls_density:.4f}, random={random_density:.4f}"


def test_tls_vs_random_connectivity():
    """Test connectivity properties of TLS vs random masks."""
    N = 256
    k = 32

    # TLS mask
    scores = torch.rand(N, N)
    tls_mask = build_tls_mask_from_scores(scores, k_per_row=k)

    # Random mask
    random_mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    # Compute graph properties
    tls_diameter = compute_graph_diameter(tls_mask['indptr'], tls_mask['indices'], N)
    random_diameter = compute_graph_diameter(random_mask['indptr'], random_mask['indices'], N)

    # Both should have finite diameter (connected components)
    assert tls_diameter < float('inf'), "TLS mask disconnected"
    assert random_diameter < float('inf'), "Random mask disconnected"

    # Print for comparison (both should be similar for random scores)
    print(f"\nTLS diameter: {tls_diameter:.2f}, Random diameter: {random_diameter:.2f}")


def test_degree_based_hubs():
    """Test that degree-based mask creates hub structure."""
    N = 256
    k = 32

    # Degree-based mask
    degree_mask = build_degree_based_mask(N=N, k_per_row=k, seed=42)

    # Check in-degree distribution (high-index nodes should have more incoming edges)
    in_degrees = torch.zeros(N, dtype=torch.int64)
    indices = degree_mask['indices']

    for col in indices:
        in_degrees[col] += 1

    # Top 20% nodes should have > average in-degree
    top_20_pct = int(0.8 * N)
    high_index_in_degree = in_degrees[top_20_pct:].float().mean()
    avg_in_degree = in_degrees.float().mean()

    assert high_index_in_degree > avg_in_degree, \
        f"Hub structure not formed: high={high_index_in_degree:.1f}, avg={avg_in_degree:.1f}"


def test_local_vs_random_clustering():
    """Test that local mask has higher clustering than random."""
    N = 256
    k_local = 24
    k_random = 8

    # Local + random mask
    local_mask = build_local_plus_random_mask(N=N, k_local=k_local, k_random=k_random)

    # Pure random mask
    random_mask = build_uniform_random_mask(N=N, k_per_row=k_local+k_random, seed=42)

    # Compute clustering coefficients
    local_clustering = compute_clustering_coefficient(
        local_mask['indptr'], local_mask['indices'], N
    )
    random_clustering = compute_clustering_coefficient(
        random_mask['indptr'], random_mask['indices'], N
    )

    # Local mask should have higher clustering
    assert local_clustering > random_clustering, \
        f"Local clustering ({local_clustering:.4f}) should be > random ({random_clustering:.4f})"


def test_mask_forward_pass_equivalence():
    """Test that different masks produce valid forward passes."""
    N = 256
    r = 16
    k = 32
    batch_size = 2

    # Build different masks
    scores = torch.rand(N, N)
    tls_mask = build_tls_mask_from_scores(scores, k_per_row=k)
    random_mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    degree_mask = build_degree_based_mask(N=N, k_per_row=k, seed=42)

    masks = {
        'TLS': tls_mask,
        'Random': random_mask,
        'Degree': degree_mask,
    }

    # Test forward pass with each mask
    for name, mask in masks.items():
        lif = LIFLayerLowRank(
            N=N, r=r,
            synapse_cls=LowRankMaskedSynapse,
            mask_csr=mask,
            dtype=torch.float32
        )

        v, s = lif.init_state(batch=batch_size)

        # Forward
        v_next, s_next = lif(v, s)

        # Check shapes
        assert v_next.shape == (batch_size, N), f"{name} mask: wrong v shape"
        assert s_next.shape == (batch_size, N), f"{name} mask: wrong s shape"

        # Check no NaN/Inf
        assert not torch.isnan(v_next).any(), f"{name} mask: NaN in v_next"
        assert not torch.isnan(s_next).any(), f"{name} mask: NaN in s_next"


def test_mask_gradient_flow():
    """Test that gradients flow correctly through different masks."""
    N = 256
    r = 16
    k = 32
    batch_size = 2

    scores = torch.rand(N, N)
    tls_mask = build_tls_mask_from_scores(scores, k_per_row=k)
    random_mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)

    masks = {'TLS': tls_mask, 'Random': random_mask}

    for name, mask in masks.items():
        lif = LIFLayerLowRank(
            N=N, r=r,
            synapse_cls=LowRankMaskedSynapse,
            mask_csr=mask,
            dtype=torch.float32
        )

        v, s = lif.init_state(batch=batch_size)

        # Forward
        v_next, s_next = lif(v, s)

        # Backward
        loss = s_next.sum()
        loss.backward()

        # Check gradients
        for param in lif.parameters():
            assert param.grad is not None, f"{name} mask: missing gradient"
            assert not torch.isnan(param.grad).any(), f"{name} mask: NaN gradient"


@pytest.mark.parametrize("mask_type,k", [
    ("tls", 32),
    ("random", 32),
    ("degree", 32),
    ("local", 32),
])
def test_various_mask_types(mask_type, k):
    """Test various mask types with same degree."""
    N = 256
    r = 16

    # Build mask
    if mask_type == "tls":
        scores = torch.rand(N, N)
        mask = build_tls_mask_from_scores(scores, k_per_row=k)
    elif mask_type == "random":
        mask = build_uniform_random_mask(N=N, k_per_row=k, seed=42)
    elif mask_type == "degree":
        mask = build_degree_based_mask(N=N, k_per_row=k, seed=42)
    elif mask_type == "local":
        mask = build_local_plus_random_mask(N=N, k_local=k//2, k_random=k//2)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    # Create LIF layer
    lif = LIFLayerLowRank(
        N=N, r=r,
        synapse_cls=LowRankMaskedSynapse,
        mask_csr=mask,
        dtype=torch.float32
    )

    # Forward pass
    v, s = lif.init_state(batch=2)
    v_next, s_next = lif(v, s)

    # Verify
    assert v_next.shape == (2, N)
    assert s_next.shape == (2, N)


def test_tls_alpha_sweep():
    """Test TLS masks with different α (persistence vs diversity balance)."""
    N = 256
    k = 32

    # Create hidden states for TLS scoring
    hidden_states = torch.randn(1, N, 128)

    # Compute centroid for persistence
    centroid = hidden_states.mean(dim=1, keepdim=True)
    dist_to_centroid = torch.norm(hidden_states - centroid, dim=-1).squeeze(0)
    persistence = dist_to_centroid / (dist_to_centroid.max() + 1e-8)

    # Compute diversity (max-min distance in k-NN)
    dists = torch.cdist(hidden_states.squeeze(0), hidden_states.squeeze(0), p=2)
    k_nn = min(5, N - 1)
    nearest_dists = torch.topk(dists, k=k_nn+1, largest=False)[0]
    diversity = nearest_dists[:, -1]
    diversity = diversity / (diversity.max() + 1e-8)

    # Test different α values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    masks = {}

    for alpha in alphas:
        # TLS scores: α·persistence + (1-α)·diversity
        scores = alpha * persistence + (1 - alpha) * diversity
        scores = scores.unsqueeze(0).expand(N, N)

        mask = build_tls_mask_from_scores(scores, k_per_row=k)
        masks[alpha] = mask

    # All masks should have same density
    densities = [mask_density(m['indptr'], N) for m in masks.values()]
    assert all(abs(d - densities[0]) < 1e-6 for d in densities), \
        "Different α values should produce same density"

    # Different α should produce different connectivity patterns
    # (Check that not all masks are identical)
    indices_list = [m['indices'].tolist() for m in masks.values()]
    all_same = all(idx == indices_list[0] for idx in indices_list[1:])
    assert not all_same, "Different α values should produce different masks"
