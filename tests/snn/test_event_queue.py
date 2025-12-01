"""
Unit tests for event-driven processing.
"""

import pytest
import torch

from tfan.snn import EventQueue, EventDrivenStepper


def test_event_queue_update():
    """Test event queue update with spikes."""
    queue = EventQueue()

    spikes = torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.float32)
    queue.update(spikes)

    assert queue.active is not None
    assert queue.active.shape == (2, 4)


def test_event_queue_active_indices():
    """Test extraction of active indices."""
    queue = EventQueue()

    spikes = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float32)
    queue.update(spikes)

    indices = queue.active_batch_indices()

    assert len(indices) == 2
    assert indices[0].tolist() == [1, 3]  # Batch 0: neurons 1 and 3
    assert indices[1].tolist() == [0, 2]  # Batch 1: neurons 0 and 2


def test_event_queue_empty():
    """Test event queue with no spikes."""
    queue = EventQueue()

    spikes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
    queue.update(spikes)

    indices = queue.active_batch_indices()

    assert len(indices) == 1
    assert len(indices[0]) == 0  # No active neurons


def test_event_queue_all_active():
    """Test event queue with all neurons spiking."""
    queue = EventQueue()

    spikes = torch.ones(2, 4, dtype=torch.float32)
    queue.update(spikes)

    indices = queue.active_batch_indices()

    assert len(indices) == 2
    assert indices[0].tolist() == [0, 1, 2, 3]
    assert indices[1].tolist() == [0, 1, 2, 3]


def test_event_queue_sparsity():
    """Test sparsity calculation."""
    queue = EventQueue()

    # 1 active out of 4 neurons -> 75% sparsity
    spikes = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    queue.update(spikes)

    sparsity = queue.sparsity()
    assert abs(sparsity - 0.75) < 1e-6


def test_event_queue_active_fraction():
    """Test active fraction calculation."""
    queue = EventQueue()

    # 2 active out of 4 neurons -> 50% active
    spikes = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    queue.update(spikes)

    active_frac = queue.active_fraction()
    assert abs(active_frac - 0.5) < 1e-6


def test_event_queue_global_indices():
    """Test global (batch, neuron) index extraction."""
    queue = EventQueue()

    spikes = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    queue.update(spikes)

    global_idx = queue.active_global_indices()

    assert global_idx.shape == (2, 2)  # 2 spikes, (batch, neuron) pairs
    assert global_idx[0].tolist() == [0, 0]  # Batch 0, neuron 0
    assert global_idx[1].tolist() == [1, 1]  # Batch 1, neuron 1


def test_event_queue_reset():
    """Test queue reset."""
    queue = EventQueue()

    spikes = torch.ones(2, 4)
    queue.update(spikes)
    assert queue.active is not None

    queue.reset()
    assert queue.active is None


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.75, 0.9, 1.0])
def test_event_queue_various_sparsities(sparsity):
    """Test queue with various sparsity levels."""
    N = 100
    active_fraction = 1.0 - sparsity
    num_active = int(N * active_fraction)

    # Create spikes with target sparsity
    spikes = torch.zeros(1, N)
    if num_active > 0:
        indices = torch.randperm(N)[:num_active]
        spikes[0, indices] = 1

    queue = EventQueue()
    queue.update(spikes)

    measured_sparsity = queue.sparsity()
    expected_sparsity = 1.0 - (num_active / N)

    assert abs(measured_sparsity - expected_sparsity) < 1e-6


def test_event_driven_stepper_initialization():
    """Test EventDrivenStepper initialization."""
    # Create a dummy model
    class DummyModel:
        def __call__(self, v, s):
            return v, s

    model = DummyModel()
    stepper = EventDrivenStepper(model, sparsity_threshold=0.75)

    assert stepper.queue is not None
    assert stepper.sparsity_threshold == 0.75


def test_event_driven_stepper_dense_mode():
    """Test stepper uses dense mode for low sparsity."""
    class DummyModel:
        def __call__(self, v, s):
            return v + 1, s + 1

    model = DummyModel()
    stepper = EventDrivenStepper(model, sparsity_threshold=0.75)

    v = torch.zeros(2, 4)
    s = torch.ones(2, 4)  # All active -> 0% sparsity

    v_next, s_next = stepper.step(v, s)

    # Should use dense mode and increment
    assert (v_next == 1).all()
    assert (s_next == 2).all()


def test_event_driven_stepper_sparse_mode():
    """Test stepper switches to sparse mode for high sparsity."""
    class DummyModel:
        def __call__(self, v, s):
            return v + 1, s + 1

    model = DummyModel()
    stepper = EventDrivenStepper(model, sparsity_threshold=0.50)

    v = torch.zeros(2, 4)
    s = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)  # 87.5% sparsity

    v_next, s_next = stepper.step(v, s)

    # Should use sparse mode (currently falls back to dense)
    assert (v_next == 1).all()


def test_event_queue_with_continuous_spikes():
    """Test queue with continuous spike values (not just 0/1)."""
    queue = EventQueue()

    # Continuous spike probabilities
    spikes = torch.tensor([[0.8, 0.1, 0.0, 0.5]], dtype=torch.float32)
    queue.update(spikes)

    # Should count all non-zero as active
    indices = queue.active_batch_indices()
    assert indices[0].tolist() == [0, 1, 3]

    # Sparsity: 1 zero out of 4 -> 25% sparsity
    sparsity = queue.sparsity()
    assert abs(sparsity - 0.25) < 1e-6
