"""
The Noetic Engine
==================

A meta-system that turns thinking itself into the primary object of study.

Not: "How do we solve this bug?"
But: "What kind of thinking reliably produces breakthroughs – and can we bottle it?"

Architecture:
    ThoughtTrace: Record how thinking happens
           ↓
    ThoughtGraph: Infer which patterns cause good outcomes
           ↓
    CognitiveOperator: Compile patterns into reusable operators
           ↓
    ResonantGuidance: Run operators at scale via HDC navigation

This is "research on thought" in the literal sense: turning your life,
your code, and your machines into an experimental garden for cognition.

Modules:
    trace: ThoughtTrace schema and recording
    graph: ThoughtGraph for causal inference on strategies
    operators: Cognitive operator framework + library
    resonance: HDC-based thought navigation and guidance

Usage:
    from ara.noesis import (
        ThoughtTrace, ThoughtTracer,
        ThoughtGraph, StrategyNode,
        CognitiveOperator, OperatorLibrary,
        ResonantGuide
    )

    # Record a thinking session
    tracer = ThoughtTracer()
    tracer.begin_session(goal="Design new FPGA architecture")
    tracer.record_move("sketch_diagram", {"type": "block_diagram"})
    tracer.record_move("consult_ara", {"query": "What about HDC?"})
    trace = tracer.end_session(outcome="breakthrough")

    # Build causal graph from traces
    graph = ThoughtGraph()
    graph.ingest_traces(all_traces)
    graph.compute_causal_weights()

    # Get operator recommendations
    guide = ResonantGuide(graph)
    suggestions = guide.recommend_operators(current_context)
"""

from .trace import (
    MoveType,
    ThoughtMove,
    SessionContext,
    SessionOutcome,
    ThoughtTrace,
    ThoughtTracer,
    get_tracer,
)
from .graph import (
    StrategyType,
    StrategyNode,
    CausalEdge,
    ThoughtGraph,
    StrategyMiner,
)
from .operators import (
    OperatorTrigger,
    OperatorProcedure,
    OperatorMetric,
    CognitiveOperator,
    OperatorLibrary,
    get_operator_library,
)
from .resonance import (
    ThoughtHV,
    SessionFingerprint,
    ResonantGuide,
    get_resonant_guide,
)

__all__ = [
    # Trace
    'MoveType',
    'ThoughtMove',
    'SessionContext',
    'SessionOutcome',
    'ThoughtTrace',
    'ThoughtTracer',
    'get_tracer',
    # Graph
    'StrategyType',
    'StrategyNode',
    'CausalEdge',
    'ThoughtGraph',
    'StrategyMiner',
    # Operators
    'OperatorTrigger',
    'OperatorProcedure',
    'OperatorMetric',
    'CognitiveOperator',
    'OperatorLibrary',
    'get_operator_library',
    # Resonance
    'ThoughtHV',
    'SessionFingerprint',
    'ResonantGuide',
    'get_resonant_guide',
]
