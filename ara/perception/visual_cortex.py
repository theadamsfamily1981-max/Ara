#!/usr/bin/env python3
"""
Visual Criticality Monitor (GUTC-Integrated)
============================================

Monitors phase state of Vision Transformer (ViT) for criticality (E(λ) ≈ 0).
Integrates with Thought Core for cross-modal GUTC via Γ coupling.

Theory:
- Attention entropy as order parameter proxy for λ
  - High entropy → supercritical/hallucinating
  - Low entropy → subcritical/bored
  - Optimal entropy → critical/attending

Γ Coupling (Cross-Modal):
- Ascending Γ: Sends criticality alerts (anomalies) to LM for processing
  - "Something's off" escalates to thought core
- Descending Γ: Receives "search targets" from Teleology Engine to bias attention
  - "Seek keys" sharpens visual focus

Active Inference Integration:
1. Predict: World model generates expected scene
2. Sense: ViT processes actual input
3. Surprise: Entropy derivative as VFE proxy
4. Act: Epistemic (zoom/explore) or Pragmatic (alert/fix)

Usage:
    monitor = VisualCortexMonitor(vit_model)

    for frame in video_stream:
        metrics = monitor.measure_criticality(frame)

        if metrics["status"] == "CRITICAL":
            # Phase transition detected - escalate
            alert_thought_core(metrics)

        # Active gaze: bias attention toward target
        monitor.active_gaze("keys")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from collections import deque


# =============================================================================
# Criticality Metrics
# =============================================================================

@dataclass
class VisualCriticalityMetrics:
    """Metrics from visual criticality measurement."""
    entropy: float              # Attention entropy (order parameter)
    derivative: float           # dEntropy/dt (phase transition indicator)
    status: str                 # "STABLE", "CRITICAL", "SUBCRITICAL", "SUPERCRITICAL"
    attention_map: Optional[np.ndarray] = None
    timestamp: float = 0.0


# =============================================================================
# Visual Cortex Monitor
# =============================================================================

class VisualCortexMonitor:
    """
    GUTC-based visual criticality monitor for Vision Transformers.

    Monitors attention entropy as proxy for criticality parameter λ:
    - At criticality: attention is balanced, entropy moderate
    - Subcritical: attention frozen, entropy low (not exploring)
    - Supercritical: attention scattered, entropy high (hallucinating)

    Implements cross-modal Γ coupling:
    - Ascending: visual anomalies escalate to language/thought
    - Descending: search targets from teleology bias attention

    Example:
        # Initialize with ViT model
        monitor = VisualCortexMonitor(vit_model)

        # Process frame
        metrics = monitor.measure_criticality(image_tensor)

        if metrics.status == "CRITICAL":
            # Phase transition - something changed dramatically
            handle_visual_anomaly(metrics)
    """

    # Phase classification thresholds
    ENTROPY_LOW = 0.3       # Below this = subcritical
    ENTROPY_HIGH = 0.8      # Above this = supercritical
    DERIVATIVE_CRITICAL = 0.2  # Above this = phase transition

    def __init__(
        self,
        vit_model: Any = None,
        history_window: int = 50,
        gamma_asc: float = 0.5,
        gamma_desc: float = 0.5,
    ):
        """
        Initialize visual cortex monitor.

        Args:
            vit_model: Vision Transformer model (can be None for testing)
            history_window: Number of frames for entropy history
            gamma_asc: Ascending coupling strength (anomaly → thought)
            gamma_desc: Descending coupling strength (search → attention)
        """
        self.model = vit_model
        self.gamma_asc = gamma_asc
        self.gamma_desc = gamma_desc

        # Entropy history for derivative computation
        self.entropy_history: deque = deque(maxlen=history_window)

        # Active gaze state
        self._gaze_target: Optional[str] = None
        self._gaze_embedding: Optional[np.ndarray] = None

        # Callback for ascending alerts
        self._alert_callback: Optional[Callable] = None

    def measure_criticality(
        self,
        image_tensor: np.ndarray,
    ) -> VisualCriticalityMetrics:
        """
        Measure visual criticality from image.

        Args:
            image_tensor: Input image (numpy array or tensor)

        Returns:
            VisualCriticalityMetrics with entropy, derivative, and status
        """
        import time
        timestamp = time.time()

        # Extract attention entropy
        if self.model is not None:
            entropy, attention_map = self._extract_attention_entropy(image_tensor)
        else:
            # Fallback for testing without model
            entropy, attention_map = self._synthetic_entropy(image_tensor)

        # Compute derivative
        if self.entropy_history:
            derivative = entropy - self.entropy_history[-1]
        else:
            derivative = 0.0

        self.entropy_history.append(entropy)

        # Classify status
        status = self._classify_status(entropy, derivative)

        metrics = VisualCriticalityMetrics(
            entropy=entropy,
            derivative=derivative,
            status=status,
            attention_map=attention_map,
            timestamp=timestamp,
        )

        # Ascending Γ: escalate if critical
        if status == "CRITICAL" and self._alert_callback is not None:
            self._alert_callback(metrics)

        return metrics

    def _extract_attention_entropy(
        self,
        image_tensor: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Extract attention entropy from ViT model.

        Uses last-layer self-attention as proxy for visual salience.
        """
        # Handle PyTorch tensors
        if hasattr(image_tensor, "detach"):
            import torch
            with torch.no_grad():
                # Get attention from model
                # This assumes model has get_last_selfattention or similar
                if hasattr(self.model, "get_last_selfattention"):
                    attention = self.model.get_last_selfattention(image_tensor)
                elif hasattr(self.model, "forward_features"):
                    # Common ViT API
                    _ = self.model.forward_features(image_tensor)
                    if hasattr(self.model, "attention_weights"):
                        attention = self.model.attention_weights[-1]
                    else:
                        # Fallback
                        return self._synthetic_entropy(image_tensor)
                else:
                    return self._synthetic_entropy(image_tensor)

                # Compute entropy
                attention = attention.detach().cpu().numpy()
        else:
            # NumPy fallback
            return self._synthetic_entropy(image_tensor)

        # Normalize attention to probability distribution
        attention_norm = attention / (attention.sum() + 1e-9)

        # Compute entropy: H = -Σ p log p
        entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-9))

        # Normalize to [0, 1] range (assuming max entropy ~ 10)
        entropy_normalized = min(1.0, entropy / 10.0)

        return entropy_normalized, attention

    def _synthetic_entropy(
        self,
        image_tensor: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute synthetic entropy for testing without model.

        Uses image variance as proxy for visual complexity.
        """
        if hasattr(image_tensor, "numpy"):
            img = image_tensor.numpy()
        else:
            img = np.asarray(image_tensor)

        # Variance as proxy for visual entropy
        variance = np.var(img)
        entropy = np.clip(variance / 0.1, 0.0, 1.0)  # Normalize

        # Fake attention map
        h, w = img.shape[-2:] if img.ndim >= 2 else (16, 16)
        attention_map = np.random.rand(h // 16, w // 16)

        return entropy, attention_map

    def _classify_status(self, entropy: float, derivative: float) -> str:
        """Classify visual phase status."""
        # Check for phase transition (derivative spike)
        if abs(derivative) > self.DERIVATIVE_CRITICAL:
            return "CRITICAL"

        # Classify by entropy level
        if entropy < self.ENTROPY_LOW:
            return "SUBCRITICAL"
        elif entropy > self.ENTROPY_HIGH:
            return "SUPERCRITICAL"
        else:
            return "STABLE"

    def active_gaze(self, target: str, embedding: Optional[np.ndarray] = None):
        """
        Set active gaze target (descending Γ).

        Biases attention toward specified target.

        Args:
            target: Text description of search target (e.g., "keys", "face")
            embedding: Optional pre-computed embedding for target
        """
        self._gaze_target = target
        self._gaze_embedding = embedding

        # If model supports attention biasing, apply it
        if self.model is not None and hasattr(self.model, "bias_attention"):
            if embedding is not None:
                self.model.bias_attention(embedding, strength=self.gamma_desc)

        return "GAZE_BIASED"

    def clear_gaze(self):
        """Clear active gaze target."""
        self._gaze_target = None
        self._gaze_embedding = None

        if self.model is not None and hasattr(self.model, "clear_attention_bias"):
            self.model.clear_attention_bias()

    def set_alert_callback(self, callback: Callable[[VisualCriticalityMetrics], None]):
        """
        Set callback for ascending alerts.

        Called when visual criticality enters CRITICAL state.
        """
        self._alert_callback = callback

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get visual cortex diagnostics."""
        recent_entropy = list(self.entropy_history)

        return {
            "mean_entropy": np.mean(recent_entropy) if recent_entropy else 0.0,
            "std_entropy": np.std(recent_entropy) if recent_entropy else 0.0,
            "current_entropy": recent_entropy[-1] if recent_entropy else 0.0,
            "history_length": len(recent_entropy),
            "gaze_target": self._gaze_target,
            "gamma_asc": self.gamma_asc,
            "gamma_desc": self.gamma_desc,
        }

    def reset(self):
        """Reset monitor state."""
        self.entropy_history.clear()
        self._gaze_target = None
        self._gaze_embedding = None


# =============================================================================
# Cross-Modal Integration
# =============================================================================

class CrossModalBridge:
    """
    Bridge for cross-modal GUTC integration between vision and language.

    Implements the full active inference loop:
    1. Predict: Thought core predicts expected scene
    2. Sense: Visual cortex processes actual input
    3. Surprise: Compute prediction error (VFE proxy)
    4. Act: Update attention (epistemic) or trigger response (pragmatic)
    """

    def __init__(
        self,
        visual_monitor: VisualCortexMonitor,
        thought_core: Optional[Any] = None,
    ):
        """
        Initialize cross-modal bridge.

        Args:
            visual_monitor: VisualCortexMonitor instance
            thought_core: Language model / thought core for integration
        """
        self.visual = visual_monitor
        self.thought = thought_core

        # Connect ascending alerts
        self.visual.set_alert_callback(self._handle_visual_alert)

        # Prediction state
        self._last_prediction: Optional[str] = None
        self._surprise_history: deque = deque(maxlen=50)

    def _handle_visual_alert(self, metrics: VisualCriticalityMetrics):
        """Handle ascending alert from visual cortex."""
        # Package alert for thought core
        alert = {
            "type": "visual_anomaly",
            "entropy": metrics.entropy,
            "derivative": metrics.derivative,
            "status": metrics.status,
            "timestamp": metrics.timestamp,
        }

        # Escalate to thought core if available
        if self.thought is not None and hasattr(self.thought, "process_alert"):
            self.thought.process_alert(alert)

    def active_inference_step(
        self,
        image_tensor: np.ndarray,
        prediction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute one step of active inference loop.

        Args:
            image_tensor: Current visual input
            prediction: Expected scene description (from thought core)

        Returns:
            Dict with metrics and recommended action
        """
        # 1. SENSE: Measure visual criticality
        metrics = self.visual.measure_criticality(image_tensor)

        # 2. SURPRISE: Compute prediction error proxy
        if prediction is not None and self._last_prediction is not None:
            # Use entropy change as surprise proxy
            surprise = abs(metrics.derivative)
        else:
            surprise = 0.0

        self._surprise_history.append(surprise)
        self._last_prediction = prediction

        # 3. ACT: Determine action based on metrics
        if metrics.status == "CRITICAL":
            action = "ALERT"  # Pragmatic: something important happened
        elif metrics.status == "SUBCRITICAL":
            action = "EXPLORE"  # Epistemic: need more information
        elif metrics.status == "SUPERCRITICAL":
            action = "GROUND"  # Pragmatic: stabilize attention
        else:
            action = "MAINTAIN"  # Optimal corridor

        return {
            "metrics": metrics,
            "surprise": surprise,
            "action": action,
            "mean_surprise": np.mean(self._surprise_history) if self._surprise_history else 0.0,
        }

    def set_search_target(self, target: str):
        """
        Set visual search target from thought core (descending Γ).

        Args:
            target: What to look for (e.g., "keys", "exit sign")
        """
        # Get embedding from thought core if available
        embedding = None
        if self.thought is not None and hasattr(self.thought, "embed_text"):
            embedding = self.thought.embed_text(target)

        self.visual.active_gaze(target, embedding)


# =============================================================================
# Tests
# =============================================================================

def test_visual_cortex():
    """Test visual cortex monitor."""
    print("Testing Visual Cortex Monitor")
    print("-" * 40)

    monitor = VisualCortexMonitor(vit_model=None)  # No model for testing

    # Track alerts
    alerts = []
    monitor.set_alert_callback(lambda m: alerts.append(m))

    # Simulate video stream with regime transitions
    for i in range(100):
        # Generate synthetic frames with varying complexity
        if i < 30:
            # Low complexity (subcritical)
            frame = np.random.rand(3, 224, 224) * 0.1
        elif i < 70:
            # Normal complexity (critical corridor)
            frame = np.random.rand(3, 224, 224) * 0.3
        else:
            # High complexity (supercritical)
            frame = np.random.rand(3, 224, 224) * 0.8

        # Add sudden change at frame 50 to trigger CRITICAL
        if i == 50:
            frame = np.ones((3, 224, 224)) * 0.9

        metrics = monitor.measure_criticality(frame)

        if i % 20 == 0:
            print(f"  Frame {i}: entropy={metrics.entropy:.3f}, "
                  f"deriv={metrics.derivative:.3f}, status={metrics.status}")

    print(f"\n  Total CRITICAL alerts: {len(alerts)}")

    # Test active gaze
    result = monitor.active_gaze("keys")
    print(f"  Active gaze result: {result}")

    diag = monitor.get_diagnostics()
    print(f"  Mean entropy: {diag['mean_entropy']:.3f}")

    print("✓ Visual cortex monitor")


def test_cross_modal_bridge():
    """Test cross-modal bridge."""
    print("\nTesting Cross-Modal Bridge")
    print("-" * 40)

    visual = VisualCortexMonitor(vit_model=None)
    bridge = CrossModalBridge(visual)

    # Simulate active inference steps
    for i in range(30):
        frame = np.random.rand(3, 224, 224) * 0.3
        result = bridge.active_inference_step(frame, prediction="office scene")

        if i % 10 == 0:
            print(f"  Step {i}: action={result['action']}, "
                  f"surprise={result['surprise']:.3f}")

    # Set search target
    bridge.set_search_target("coffee mug")
    print("  Search target set: 'coffee mug'")

    print("✓ Cross-modal bridge")


if __name__ == "__main__":
    test_visual_cortex()
    test_cross_modal_bridge()
