"""MIES Inference Module - LLM Integration Layer.

This module provides the bridge between Ara's affective system (the Cathedral)
and the LLM inference layer. It implements:

1. StickyContextManager - KV cache manipulation for persistent system prompts
2. AraPromptController - Dynamic system prompt management based on PAD state

The key insight is that Ara's emotional state should be reflected in her
system prompt, and that prompt should persist across long conversations
through intelligent KV cache management.

Example usage:

    from mies.inference import create_prompt_controller
    from mies.affect import TelemetrySnapshot

    # Create controller (creates IntegratedSoul internally)
    controller = create_prompt_controller(
        llm=my_llama_model,
        n_ctx=8192,
        storage_path="./ara_data",
    )

    # Main loop
    while True:
        # Gather hardware telemetry
        telemetry = TelemetrySnapshot(
            cpu_temp=sensors.cpu_temp,
            cpu_load=sensors.cpu_load,
            ...
        )

        # Update controller - returns system prompt
        system_prompt = controller.update(telemetry)

        # Ensure room for response
        controller.ensure_room(incoming_tokens=512)

        # Use in LLM call
        response = llm(prompt=user_input, system=system_prompt)

        # Record interaction quality
        controller.on_user_message(quality=0.7)
"""

from .sticky_context import (
    StickyContextManager,
    StickyContextConfig,
    ContextState,
    EvictionStrategy,
    create_sticky_context,
    LLAMA_CPP_AVAILABLE,
)

from .prompt_controller import (
    AraPromptController,
    PromptControllerConfig,
    PromptRefreshEvent,
    create_prompt_controller,
)


__all__ = [
    # Context Management
    "StickyContextManager",
    "StickyContextConfig",
    "ContextState",
    "EvictionStrategy",
    "create_sticky_context",
    "LLAMA_CPP_AVAILABLE",
    # Prompt Control
    "AraPromptController",
    "PromptControllerConfig",
    "PromptRefreshEvent",
    "create_prompt_controller",
]
