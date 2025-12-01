"""
Ara CXL Control Plane - Triton Inference Server Model

Real-time control loop with ultra-low latency for FPGA offload.
Target: p95 < 200us for control cycle.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

# Add project to path
_workspace = Path("/workspace")
if _workspace.exists():
    sys.path.insert(0, str(_workspace))

# Import Ara CXL control
try:
    from ara.cxl_control import ControlPlane, ControlPlaneMode
    ARA_AVAILABLE = True
except ImportError:
    ARA_AVAILABLE = False


class TritonPythonModel:
    """Triton Python backend model for Ara CXL Control."""

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])

        # Get parameters
        params = self.model_config.get("parameters", {})
        self.target_latency_us = float(params.get("target_latency_us", {}).get("string_value", "200.0"))
        self.enable_cache = params.get("enable_pgu_cache", {}).get("string_value", "true") == "true"
        self.lif_neurons = int(params.get("lif_neurons", {}).get("string_value", "64"))

        # Initialize control plane
        if ARA_AVAILABLE:
            self.control_plane = ControlPlane(
                mode=ControlPlaneMode.EMULATED_HLS,
                target_latency_us=self.target_latency_us,
            )
        else:
            self.control_plane = None

        print(f"[ara_cxl_control] Initialized, target={self.target_latency_us}us")

    def execute(self, requests):
        """Execute inference for a batch of requests."""
        responses = []

        for request in requests:
            try:
                start_ns = time.time_ns()

                # Get inputs
                valence_t = pb_utils.get_input_tensor_by_name(request, "valence")
                arousal_t = pb_utils.get_input_tensor_by_name(request, "arousal")
                dominance_t = pb_utils.get_input_tensor_by_name(request, "dominance")
                current_t = pb_utils.get_input_tensor_by_name(request, "input_current")

                valence = valence_t.as_numpy()[0] if valence_t else 0.0
                arousal = arousal_t.as_numpy()[0] if arousal_t else 0.5
                dominance = dominance_t.as_numpy()[0] if dominance_t else 0.5
                input_current = current_t.as_numpy()[0] if current_t else 0.0

                # Execute control cycle
                if ARA_AVAILABLE and self.control_plane is not None:
                    result = self.control_plane.fast_control_step(
                        valence=valence,
                        arousal=arousal,
                        dominance=dominance,
                        input_current=input_current,
                    )
                    temp_mult = result["temperature_mult"]
                    mem_mult = result["memory_mult"]
                    attn_gain = result["attention_gain"]
                    cache_hit = result["cache_hit"]
                else:
                    # Fallback HLS-compatible computation
                    temp_mult = 0.8 + 0.5 * arousal
                    mem_mult = 0.95 + 0.25 * (valence + 1) / 2
                    attn_gain = 0.8 + 0.4 * dominance
                    cache_hit = False

                end_ns = time.time_ns()
                latency_us = (end_ns - start_ns) / 1000.0

                # Create output tensors
                temp_out = pb_utils.Tensor("temperature_mult", np.array([temp_mult], dtype=np.float32))
                mem_out = pb_utils.Tensor("memory_mult", np.array([mem_mult], dtype=np.float32))
                attn_out = pb_utils.Tensor("attention_gain", np.array([attn_gain], dtype=np.float32))
                cache_out = pb_utils.Tensor("cache_hit", np.array([cache_hit], dtype=np.bool_))
                latency_out = pb_utils.Tensor("latency_us", np.array([latency_us], dtype=np.float32))

                response = pb_utils.InferenceResponse(
                    output_tensors=[temp_out, mem_out, attn_out, cache_out, latency_out]
                )
                responses.append(response)

            except Exception as e:
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def finalize(self):
        """Clean up resources."""
        print("[ara_cxl_control] Model finalized")
