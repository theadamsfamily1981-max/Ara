"""
Ara Interoception Core - Triton Inference Server Model

Processes L1 body state and L2 perception to generate true interoceptive signals.
SNN membrane dynamics drive L3 metacontrol decisions.
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

# Import Ara interoception core
try:
    from ara.interoception import (
        InteroceptionCore,
        SNNToL3Bridge,
        L1BodyState,
        L2PerceptionState,
    )
    ARA_AVAILABLE = True
except ImportError:
    ARA_AVAILABLE = False


class TritonPythonModel:
    """Triton Python backend model for Ara Interoception."""

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])

        # Get parameters
        params = self.model_config.get("parameters", {})
        self.num_neurons = int(params.get("num_snn_neurons", {}).get("string_value", "128"))
        self.tau = float(params.get("tau", {}).get("string_value", "10.0"))
        self.stability_threshold = float(params.get("stability_threshold", {}).get("string_value", "0.3"))

        # Initialize interoception core
        if ARA_AVAILABLE:
            self.core = InteroceptionCore(
                num_neurons=self.num_neurons,
                tau=self.tau,
            )
            self.bridge = SNNToL3Bridge(
                self.core,
                stability_threshold=self.stability_threshold,
            )
        else:
            self.core = None
            self.bridge = None

        print(f"[ara_interoception] Initialized with N={self.num_neurons}, tau={self.tau}")

    def execute(self, requests):
        """Execute inference for a batch of requests."""
        responses = []

        for request in requests:
            try:
                # Get inputs
                heart_rate = pb_utils.get_input_tensor_by_name(request, "heart_rate")
                breath_rate = pb_utils.get_input_tensor_by_name(request, "breath_rate")
                muscle_tension = pb_utils.get_input_tensor_by_name(request, "muscle_tension")

                # Optional inputs
                audio_valence = pb_utils.get_input_tensor_by_name(request, "audio_valence")
                audio_arousal = pb_utils.get_input_tensor_by_name(request, "audio_arousal")
                text_sentiment = pb_utils.get_input_tensor_by_name(request, "text_sentiment")

                # Convert to numpy
                hr = heart_rate.as_numpy()[0] if heart_rate else 72.0
                br = breath_rate.as_numpy()[0] if breath_rate else 16.0
                mt = muscle_tension.as_numpy()[0] if muscle_tension else 0.3
                av = audio_valence.as_numpy()[0] if audio_valence else 0.0
                aa = audio_arousal.as_numpy()[0] if audio_arousal else 0.5
                ts = text_sentiment.as_numpy()[0] if text_sentiment else 0.0

                # Process through interoception core
                if ARA_AVAILABLE and self.bridge is not None:
                    l1 = L1BodyState(heart_rate=hr, breath_rate=br, muscle_tension=mt)
                    l2 = L2PerceptionState(audio_valence=av, audio_arousal=aa, text_sentiment=ts)

                    signal = self.bridge.run_control_step(l1, l2, num_steps=5)

                    valence = signal.valence
                    arousal = signal.arousal
                    dominance = signal.dominance
                    topology_gap = signal.topology_gap
                    temp_mult = signal.temperature_mult
                    mem_mult = signal.memory_mult
                    needs_supp = signal.needs_suppression
                    stab_warn = signal.stability_warning
                else:
                    # Fallback: simple computation
                    valence = ts * 0.5 + av * 0.5
                    arousal = min(1.0, hr / 100.0 * 0.5 + aa * 0.5)
                    dominance = 0.5 + mt * 0.3
                    topology_gap = 0.1
                    temp_mult = 0.8 + arousal * 0.5
                    mem_mult = 0.95 + valence * 0.25
                    needs_supp = arousal > 0.8
                    stab_warn = topology_gap > 0.3

                # Create output tensors
                valence_out = pb_utils.Tensor("valence", np.array([valence], dtype=np.float32))
                arousal_out = pb_utils.Tensor("arousal", np.array([arousal], dtype=np.float32))
                dominance_out = pb_utils.Tensor("dominance", np.array([dominance], dtype=np.float32))
                topo_out = pb_utils.Tensor("topology_gap", np.array([topology_gap], dtype=np.float32))
                temp_out = pb_utils.Tensor("temperature_mult", np.array([temp_mult], dtype=np.float32))
                mem_out = pb_utils.Tensor("memory_mult", np.array([mem_mult], dtype=np.float32))
                supp_out = pb_utils.Tensor("needs_suppression", np.array([needs_supp], dtype=np.bool_))
                warn_out = pb_utils.Tensor("stability_warning", np.array([stab_warn], dtype=np.bool_))

                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        valence_out, arousal_out, dominance_out, topo_out,
                        temp_out, mem_out, supp_out, warn_out
                    ]
                )
                responses.append(response)

            except Exception as e:
                error = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def finalize(self):
        """Clean up resources."""
        print("[ara_interoception] Model finalized")
