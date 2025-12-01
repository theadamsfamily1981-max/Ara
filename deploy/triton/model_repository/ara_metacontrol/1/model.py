"""
Ara L3 Metacontrol - Triton Inference Server Model

Computes L3 metacontrol signals from PAD state.
Supports D-Bus integration for external cockpit communication.
"""

import json
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

# Add project to path
_workspace = Path("/workspace")
if _workspace.exists():
    sys.path.insert(0, str(_workspace))

# Import Ara metacontrol
try:
    from ara.metacontrol import L3Controller, WorkspaceMode, compute_pad_gating
    ARA_AVAILABLE = True
except ImportError:
    ARA_AVAILABLE = False


class TritonPythonModel:
    """Triton Python backend model for Ara Metacontrol."""

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])

        # Get parameters
        params = self.model_config.get("parameters", {})
        self.enable_dbus = params.get("enable_dbus", {}).get("string_value", "false") == "true"
        self.default_mode = params.get("default_mode", {}).get("string_value", "focus")

        # Initialize L3 controller
        if ARA_AVAILABLE:
            self.controller = L3Controller()
        else:
            self.controller = None

        print(f"[ara_metacontrol] Initialized, D-Bus={self.enable_dbus}")

    def execute(self, requests):
        """Execute inference for a batch of requests."""
        responses = []

        for request in requests:
            try:
                # Get inputs
                valence_t = pb_utils.get_input_tensor_by_name(request, "valence")
                arousal_t = pb_utils.get_input_tensor_by_name(request, "arousal")
                dominance_t = pb_utils.get_input_tensor_by_name(request, "dominance")
                mode_t = pb_utils.get_input_tensor_by_name(request, "workspace_mode")

                valence = valence_t.as_numpy()[0] if valence_t else 0.0
                arousal = arousal_t.as_numpy()[0] if arousal_t else 0.5
                dominance = dominance_t.as_numpy()[0] if dominance_t else 0.5
                mode_str = mode_t.as_numpy()[0].decode() if mode_t else self.default_mode

                # Compute L3 control
                if ARA_AVAILABLE and self.controller is not None:
                    try:
                        mode = WorkspaceMode(mode_str)
                    except ValueError:
                        mode = WorkspaceMode.FOCUS

                    self.controller.set_workspace_mode(mode)
                    gating = self.controller.compute_gating(valence, arousal, dominance)

                    temp_mult = gating.get("temperature_mult", 1.0)
                    mem_mult = gating.get("memory_mult", 1.0)
                    attn_gain = gating.get("attention_gain", 1.0)
                    decision = gating.get("gating_decision", "proceed")
                else:
                    # Fallback computation
                    temp_mult = 0.8 + 0.5 * arousal
                    mem_mult = 0.95 + 0.25 * (valence + 1) / 2
                    attn_gain = 0.8 + 0.4 * dominance
                    decision = "proceed" if valence > -0.5 else "suppress"

                # Create output tensors
                temp_out = pb_utils.Tensor("temperature_mult", np.array([temp_mult], dtype=np.float32))
                mem_out = pb_utils.Tensor("memory_mult", np.array([mem_mult], dtype=np.float32))
                attn_out = pb_utils.Tensor("attention_gain", np.array([attn_gain], dtype=np.float32))
                decision_out = pb_utils.Tensor("gating_decision", np.array([decision], dtype=object))
                mode_out = pb_utils.Tensor("mode", np.array([mode_str], dtype=object))

                response = pb_utils.InferenceResponse(
                    output_tensors=[temp_out, mem_out, attn_out, decision_out, mode_out]
                )
                responses.append(response)

            except Exception as e:
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def finalize(self):
        """Clean up resources."""
        print("[ara_metacontrol] Model finalized")
