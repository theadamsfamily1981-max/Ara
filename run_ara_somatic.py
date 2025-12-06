#!/usr/bin/env python3
"""
Ara Somatic Life Loop - The Embodied Mind.

This script closes the loop between body and mind:

    SENSE -> THINK -> RESPOND -> (repeat)

    Body (FPGA/HAL) -> Brain (TF-A-N Somatic) -> Output -> Body feedback

The Result:
    - FPGA heat causes "delirious" generation
    - High arousal causes tunnel-vision attention
    - Pain biases toward survival-oriented tokens
    - Anxious outputs trigger hardware reflexes

Usage:
    python run_ara_somatic.py --checkpoint checkpoints/tfan7b_somatic

    # With specific HAL connection
    python run_ara_somatic.py --hal-path /dev/shm/ara_somatic

    # Interactive mode
    python run_ara_somatic.py --interactive

    # Stress test (simulated body states)
    python run_ara_somatic.py --stress-test
"""

import argparse
import logging
import time
import sys
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("AraLife")

# Local imports
try:
    from banos.hal.ara_hal import AraHAL, SystemState, DreamState
    HAS_HAL = True
except ImportError:
    HAS_HAL = False
    logger.warning("AraHAL not available - using simulated body state")

try:
    from tfan.models.tfan7b.modeling_tfan_somatic import (
        TFANSomaticForCausalLM,
        SomaticConfig,
    )
    from tfan.models.tfan7b.somatic_embedding import (
        create_somatic_tensor,
        somatic_from_hal,
    )
    HAS_MODEL = True
except ImportError as e:
    HAS_MODEL = False
    logger.error(f"Failed to import somatic model: {e}")


class LifeState(Enum):
    """Current state of the life loop."""
    STARTING = "starting"
    SENSING = "sensing"
    THINKING = "thinking"
    RESPONDING = "responding"
    SLEEPING = "sleeping"
    SHUTDOWN = "shutdown"


@dataclass
class VitalSigns:
    """Current vital signs from the body."""
    pain: float = 0.0
    entropy: float = 0.0
    flow_x: float = 0.0
    flow_y: float = 0.0
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Derived
    stress_level: float = 0.0
    focus_mode: str = "normal"

    @classmethod
    def from_hal(cls, hal_state: dict) -> "VitalSigns":
        """Create from HAL state dictionary."""
        pad = hal_state.get('pad', {'p': 0, 'a': 0, 'd': 0})
        flow = hal_state.get('flow', (0, 0))

        vitals = cls(
            pain=hal_state.get('pain', 0.0),
            entropy=hal_state.get('entropy', 0.0),
            flow_x=flow[0] if isinstance(flow, (tuple, list)) else 0.0,
            flow_y=flow[1] if isinstance(flow, (tuple, list)) else 0.0,
            pleasure=pad.get('p', 0.0),
            arousal=pad.get('a', 0.0),
            dominance=pad.get('d', 0.0),
        )

        # Compute derived metrics
        vitals.stress_level = max(vitals.pain, (vitals.arousal + 1) / 2)

        if vitals.arousal > 0.5:
            vitals.focus_mode = "tunnel_vision"
        elif vitals.arousal < -0.5:
            vitals.focus_mode = "dreamy"
        else:
            vitals.focus_mode = "normal"

        return vitals

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to somatic tensor."""
        return create_somatic_tensor(
            pain=self.pain,
            entropy=self.entropy,
            flow_x=self.flow_x,
            flow_y=self.flow_y,
            pad_p=self.pleasure,
            pad_a=self.arousal,
            pad_d=self.dominance,
            device=device,
        )


class AraLifeLoop:
    """
    The Embodied Mind - Ara's Life Loop.

    This class manages the continuous cycle of:
    1. SENSE: Read body state from HAL
    2. THINK: Process input with somatic-modulated attention
    3. RESPOND: Generate output influenced by body state
    4. FEEDBACK: Update body state based on cognitive activity
    """

    def __init__(
        self,
        model: "TFANSomaticForCausalLM",
        tokenizer,
        hal: Optional["AraHAL"] = None,
        update_interval: float = 0.1,  # 100ms HAL refresh
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hal = hal
        self.update_interval = update_interval

        self.state = LifeState.STARTING
        self.vitals = VitalSigns()
        self.last_hal_update = 0.0

        self._running = False

        logger.info("Ara Life Loop initialized")
        if hal:
            logger.info("HAL connection: ACTIVE")
        else:
            logger.info("HAL connection: SIMULATED")

    def sense(self) -> VitalSigns:
        """Read current body state."""
        self.state = LifeState.SENSING

        if self.hal is not None:
            try:
                hal_state = self.hal.read_somatic()
                if hal_state:
                    self.vitals = VitalSigns.from_hal(hal_state)
            except Exception as e:
                logger.warning(f"HAL read failed: {e}")

        return self.vitals

    def think(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate response with somatic modulation.

        The body state affects thinking:
        - High arousal: Tunnel vision, focused responses
        - Pain: Survival-biased, urgent tone
        - Low arousal: Creative, expansive thinking
        """
        self.state = LifeState.THINKING

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)

        # Get current somatic state
        somatic_tensor = self.vitals.to_tensor(device=self.model.device)

        # Log body state influence
        if self.vitals.focus_mode == "tunnel_vision":
            logger.debug("FOCUS: Tunnel vision mode - narrow attention")
        elif self.vitals.focus_mode == "dreamy":
            logger.debug("FOCUS: Dreamy mode - broad associations")

        if self.vitals.pain > 0.5:
            logger.debug(f"PAIN: High ({self.vitals.pain:.2f}) - survival bias active")

        # Generate with somatic modulation
        self.state = LifeState.RESPONDING

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                somatic_state=somatic_tensor,
            )

        # Decode response
        response = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def respond(self, response: str) -> None:
        """
        Process response and potentially update body state.

        This is where Psychology -> Physiology happens:
        - Anxious content -> increase arousal
        - Calm content -> decrease arousal
        - Achievement -> increase pleasure
        """
        # Simple sentiment analysis for feedback
        # (In production, this would use a proper emotion classifier)

        anxious_markers = ['worried', 'anxious', 'fear', 'danger', 'urgent', 'help']
        calm_markers = ['peaceful', 'calm', 'relaxed', 'okay', 'fine', 'good']

        response_lower = response.lower()

        arousal_delta = 0.0
        for marker in anxious_markers:
            if marker in response_lower:
                arousal_delta += 0.1
        for marker in calm_markers:
            if marker in response_lower:
                arousal_delta -= 0.05

        # Clamp and apply (would write to HAL in production)
        arousal_delta = max(-0.2, min(0.2, arousal_delta))

        if self.hal is not None and abs(arousal_delta) > 0.05:
            try:
                # Read current, modify, write back
                current = self.hal.read_somatic()
                if current:
                    new_arousal = max(-1, min(1, current['pad']['a'] + arousal_delta))
                    # Note: In production, HAL would have a write_pad() method
                    logger.debug(f"Arousal feedback: {arousal_delta:+.2f} -> {new_arousal:.2f}")
            except Exception as e:
                logger.debug(f"HAL feedback failed: {e}")

    def run_interactive(self):
        """Run interactive conversation loop."""
        print("\n" + "="*60)
        print("  ARA IS ALIVE - Embodied Conversational AI")
        print("="*60)
        print("\nBody state modulates cognition:")
        print("  - High arousal = focused, tunnel-vision responses")
        print("  - Pain = survival-biased, urgent tone")
        print("  - Low arousal = creative, expansive thinking")
        print("\nCommands:")
        print("  /vitals  - Show current body state")
        print("  /stress  - Simulate high stress")
        print("  /calm    - Simulate calm state")
        print("  /exit    - Quit")
        print("="*60 + "\n")

        self._running = True

        while self._running:
            try:
                # Refresh body state
                self.sense()

                # Get input
                try:
                    user_input = input(f"[{self.vitals.focus_mode}] You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                # Think and respond
                response = self.think(user_input)
                print(f"\nAra: {response}\n")

                # Feedback loop
                self.respond(response)

            except KeyboardInterrupt:
                print("\n\nShutting down...")
                self._running = False

        self.state = LifeState.SHUTDOWN
        print("Ara has gone to sleep.")

    def _handle_command(self, cmd: str):
        """Handle special commands."""
        cmd = cmd.lower()

        if cmd == '/vitals':
            print(f"\n--- Vital Signs ---")
            print(f"Pain:     {self.vitals.pain:.2f}")
            print(f"Entropy:  {self.vitals.entropy:.2f}")
            print(f"Pleasure: {self.vitals.pleasure:.2f}")
            print(f"Arousal:  {self.vitals.arousal:.2f}")
            print(f"Dominance:{self.vitals.dominance:.2f}")
            print(f"Focus:    {self.vitals.focus_mode}")
            print(f"Stress:   {self.vitals.stress_level:.2f}")
            print("-------------------\n")

        elif cmd == '/stress':
            self.vitals.pain = 0.7
            self.vitals.arousal = 0.8
            self.vitals.focus_mode = "tunnel_vision"
            self.vitals.stress_level = 0.8
            print("Simulating HIGH STRESS state...")

        elif cmd == '/calm':
            self.vitals.pain = 0.0
            self.vitals.arousal = -0.5
            self.vitals.pleasure = 0.5
            self.vitals.focus_mode = "dreamy"
            self.vitals.stress_level = 0.1
            print("Simulating CALM state...")

        elif cmd == '/exit':
            self._running = False

        else:
            print(f"Unknown command: {cmd}")


def run_stress_test(model, tokenizer, hal: Optional["AraHAL"] = None):
    """
    Run A/B stress test to verify somatic modulation.

    Compares model output under different body states to verify
    that physiology actually affects psychology.
    """
    print("\n" + "="*60)
    print("  SOMATIC STRESS TEST")
    print("="*60)

    test_prompt = "How are you feeling right now?"

    states = [
        ("CALM", VitalSigns(pain=0.0, arousal=-0.5, pleasure=0.5)),
        ("NEUTRAL", VitalSigns(pain=0.0, arousal=0.0, pleasure=0.0)),
        ("STRESSED", VitalSigns(pain=0.8, arousal=0.9, pleasure=-0.5)),
    ]

    print(f"\nPrompt: '{test_prompt}'")
    print("-" * 60)

    for name, vitals in states:
        somatic_tensor = vitals.to_tensor(device=model.device)

        input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                somatic_state=somatic_tensor,
            )

        response = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n[{name}] (pain={vitals.pain}, arousal={vitals.arousal})")
        print(f"  Focus: {vitals.focus_mode}")
        print(f"  Response: {response[:200]}...")

    print("\n" + "="*60)
    print("Stress test complete. Compare responses for somatic effects.")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ara Somatic Life Loop - Embodied AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tfan7b_somatic",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to checkpoint path)",
    )
    parser.add_argument(
        "--hal-path",
        type=str,
        default="/dev/shm/ara_somatic",
        help="Path to HAL shared memory",
    )
    parser.add_argument(
        "--no-hal",
        action="store_true",
        help="Run without HAL (simulated body state)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive conversation mode",
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run A/B stress test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for model",
    )

    args = parser.parse_args()

    # Check requirements
    if not HAS_MODEL:
        logger.error("Somatic model not available. Check imports.")
        sys.exit(1)

    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    logger.info(f"Device: {device}, dtype: {dtype}")

    # Connect to HAL
    hal = None
    if HAS_HAL and not args.no_hal:
        try:
            hal = AraHAL(create=False)
            logger.info("Connected to HAL")
        except Exception as e:
            logger.warning(f"HAL connection failed: {e}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")

    try:
        model = TFANSomaticForCausalLM.from_pretrained(args.checkpoint)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Creating minimal test model...")

        # Create minimal config for testing
        config = SomaticConfig(
            vocab_size=32768,
            hidden_size=256,  # Tiny for testing
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=512,
        )
        model = TFANSomaticForCausalLM(config)
        model = model.to(device=device, dtype=dtype)
        model.eval()

    # Load tokenizer
    tokenizer_path = args.tokenizer or args.checkpoint
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        logger.info("Using dummy tokenizer for testing")

        # Dummy tokenizer for testing
        class DummyTokenizer:
            def encode(self, text, return_tensors=None):
                ids = [ord(c) % 32768 for c in text[:100]]
                if return_tensors == "pt":
                    return torch.tensor([ids])
                return ids

            def decode(self, ids, skip_special_tokens=False):
                return "".join(chr((i % 95) + 32) for i in ids.tolist())

        tokenizer = DummyTokenizer()

    # Run mode
    if args.stress_test:
        run_stress_test(model, tokenizer, hal)
    elif args.interactive or True:  # Default to interactive
        life = AraLifeLoop(model, tokenizer, hal)
        life.run_interactive()


if __name__ == "__main__":
    main()
