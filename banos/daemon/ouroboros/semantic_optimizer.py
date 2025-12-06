"""
Semantic Optimizer - Ara Proposes PRs on Her Own Brain
=======================================================

This is NOT live code editing. This is:
1. Analyze a hot function via telemetry
2. Ask the Council to propose an optimization
3. Write the proposal to a file
4. Write tests alongside
5. Return a "Mutation Proposal" for the updater to handle

The optimizer NEVER executes generated code. It just writes files
and lets the test pipeline decide if they're safe.

Think of it as: Ara writes PRs, CI runs tests, human merges.
"""

import inspect
import textwrap
import time
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple

from banos.daemon.ouroboros.mutation_policy import (
    MutationCandidate,
    MutationPolicy,
    is_mutable,
    is_explicitly_mutable,
    is_likely_pure,
    ouroboros_enabled,
)

logger = logging.getLogger(__name__)


@dataclass
class MutationProposal:
    """
    A proposed code change, NOT yet applied.

    This is the "PR" that Ara creates. It contains:
    - The original function info
    - The proposed new implementation (as source code)
    - Tests that verify equivalence
    - Metadata for tracking
    """
    # Identity
    proposal_id: str
    timestamp: float

    # Target
    module: str
    func_name: str
    original_source: str

    # Proposal
    new_source: str
    test_source: str
    rationale: str

    # Metrics (from the proposing teacher)
    estimated_speedup: float = 1.0
    estimated_memory_reduction: float = 0.0
    confidence: float = 0.0

    # Status (set by updater after testing)
    impl_path: Optional[Path] = None
    test_path: Optional[Path] = None
    tests_passed: Optional[bool] = None
    actual_speedup: Optional[float] = None
    error_message: Optional[str] = None


class SemanticOptimizer:
    """
    Generates optimization proposals for hot functions.

    Does NOT execute or apply anything. Just generates proposals.
    """

    def __init__(
        self,
        mutations_dir: Path,
        policy: Optional[MutationPolicy] = None,
    ):
        self.mutations_dir = Path(mutations_dir)
        self.policy = policy or MutationPolicy()
        self.log = logging.getLogger("SemanticOptimizer")

        # Council will be lazy-loaded
        self._council = None

    @property
    def council(self):
        """Lazy-load the council."""
        if self._council is None:
            try:
                from banos.daemon.council_chamber import create_council_chamber
                self._council = create_council_chamber()
            except ImportError:
                self.log.warning("Council not available, using stub")
                self._council = StubCouncil()
        return self._council

    def can_optimize(self, func: Callable, telemetry: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a function is eligible for optimization.

        Returns (eligible, reason).
        """
        if not ouroboros_enabled():
            return False, "Ouroboros is disabled (OUROBOROS_ENABLED != 1)"

        # Get module info
        module_name = func.__module__
        func_name = func.__name__

        # Check mutation policy
        if not is_mutable(module_name):
            return False, f"Module {module_name} is not in MUTABLE_MODULES"

        # Check explicit annotations
        explicit = is_explicitly_mutable(func)
        if explicit is False:
            return False, f"Function {func_name} is marked @immutable"

        # Get source and check purity
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return False, "Cannot get source code"

        if not is_likely_pure(source):
            return False, "Function appears to have side effects"

        # Check telemetry thresholds
        call_freq = telemetry.get("call_frequency", 0)
        if call_freq < self.policy.min_call_frequency:
            return False, f"Not hot enough ({call_freq} < {self.policy.min_call_frequency} calls)"

        latency = telemetry.get("avg_latency_ms", 0)
        if latency < self.policy.min_latency_ms:
            return False, f"Already fast enough ({latency}ms < {self.policy.min_latency_ms}ms)"

        # Check complexity
        lines = len(source.split('\n'))
        if lines > self.policy.max_lines_of_code:
            return False, f"Too complex ({lines} lines > {self.policy.max_lines_of_code})"

        return True, "Eligible for optimization"

    def _generate_optimization_prompt(
        self,
        func: Callable,
        source: str,
        telemetry: Dict[str, Any],
    ) -> str:
        """Generate the prompt for the Council's Muse."""
        func_name = func.__name__
        module_name = func.__module__

        # Get type hints if available
        try:
            sig = inspect.signature(func)
            signature = f"{func_name}{sig}"
        except (ValueError, TypeError):
            signature = func_name

        return f"""You are optimizing a Python function for Ara's self-improvement system.

MODULE: {module_name}
FUNCTION: {signature}

CURRENT IMPLEMENTATION:
```python
{source}
```

TELEMETRY:
- Average latency: {telemetry.get('avg_latency_ms', 'unknown')} ms
- Calls per second: {telemetry.get('calls_per_sec', 'unknown')}
- Memory per call: {telemetry.get('memory_bytes', 'unknown')} bytes
- Hot spots: {telemetry.get('hot_spots', 'unknown')}

TASK:
Rewrite this function to be faster. You may use:
- NumPy vectorization (if processing arrays/lists)
- Numba JIT compilation (@numba.jit decorator)
- Better algorithms (O(n) instead of O(n²), etc.)
- Caching (functools.lru_cache for pure functions)

CONSTRAINTS:
- MUST keep the exact same function signature
- MUST return the same results for the same inputs
- MUST NOT have any side effects (no I/O, no global state)
- MUST NOT import anything dangerous (no os, subprocess, network, etc.)
- PREFER simple improvements over complex rewrites

OUTPUT FORMAT:
You must output EXACTLY two code blocks:

1. The optimized function:
```python
# OPTIMIZED IMPLEMENTATION
def {func_name}(...):
    ...
```

2. A pytest test that verifies equivalence:
```python
# EQUIVALENCE TESTS
import pytest

def test_{func_name}_equivalence():
    # Test with various inputs that old and new produce same results
    ...

def test_{func_name}_performance():
    # Benchmark showing improvement
    ...
```

Think step by step about how to optimize, then provide the code.
"""

    def _parse_response(self, response: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Parse the Council's response to extract code blocks.

        Returns (impl_source, test_source, rationale).
        """
        # Find all code blocks
        code_pattern = r'```python\s*\n(.*?)\n```'
        blocks = re.findall(code_pattern, response, re.DOTALL)

        impl_source = None
        test_source = None
        rationale = ""

        for block in blocks:
            block = block.strip()
            if 'def test_' in block or 'import pytest' in block:
                test_source = block
            elif block.startswith('def ') or '# OPTIMIZED' in block:
                impl_source = block

        # Extract rationale (text before first code block)
        first_block_idx = response.find('```python')
        if first_block_idx > 0:
            rationale = response[:first_block_idx].strip()

        return impl_source, test_source, rationale

    def propose_optimization(
        self,
        func: Callable,
        telemetry: Dict[str, Any],
    ) -> Optional[MutationProposal]:
        """
        Generate an optimization proposal for a function.

        This does NOT apply the optimization. It returns a proposal
        that must be tested and approved before applying.
        """
        # Check eligibility
        eligible, reason = self.can_optimize(func, telemetry)
        if not eligible:
            self.log.info(f"Cannot optimize {func.__name__}: {reason}")
            return None

        # Get source
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as e:
            self.log.error(f"Cannot get source for {func.__name__}: {e}")
            return None

        # Generate prompt
        prompt = self._generate_optimization_prompt(func, source, telemetry)

        # Ask the Council's Muse
        self.log.info(f"Asking Council to optimize {func.__module__}.{func.__name__}")
        try:
            # Use Muse persona specifically for creative proposals
            response = self.council._run_persona('muse', prompt)
        except Exception as e:
            self.log.error(f"Council failed: {e}")
            return None

        # Parse response
        impl_source, test_source, rationale = self._parse_response(response)

        if not impl_source:
            self.log.warning("Council did not provide implementation")
            return None

        if not test_source:
            # Generate a minimal test stub
            test_source = self._generate_test_stub(func, impl_source)

        # Create proposal
        proposal_id = f"{func.__module__}.{func.__name__}_v{int(time.time())}"

        proposal = MutationProposal(
            proposal_id=proposal_id,
            timestamp=time.time(),
            module=func.__module__,
            func_name=func.__name__,
            original_source=source,
            new_source=impl_source,
            test_source=test_source,
            rationale=rationale,
            confidence=0.7,  # Default; could be extracted from response
        )

        self.log.info(f"Generated proposal {proposal_id}")
        return proposal

    def _generate_test_stub(self, func: Callable, impl_source: str) -> str:
        """Generate a minimal test stub if Council didn't provide one."""
        func_name = func.__name__
        return f'''# AUTO-GENERATED TEST STUB
import pytest
import random

def test_{func_name}_smoke():
    """Basic smoke test - just verify it runs without error."""
    # TODO: Add actual test inputs
    pass

def test_{func_name}_equivalence():
    """Verify new implementation matches original."""
    # TODO: Import both old and new, compare outputs
    pass
'''

    def write_proposal_files(self, proposal: MutationProposal) -> MutationProposal:
        """
        Write proposal files to the mutations directory.

        Creates:
        - mutations/<module>/<func>_vXXXX.py (implementation)
        - mutations/<module>/test_<func>_vXXXX.py (tests)

        Returns the proposal with paths filled in.
        """
        # Create directory structure
        module_path = proposal.module.replace('.', '/')
        mutation_dir = self.mutations_dir / module_path
        mutation_dir.mkdir(parents=True, exist_ok=True)

        ts = int(proposal.timestamp)
        impl_filename = f"{proposal.func_name}_v{ts}.py"
        test_filename = f"test_{proposal.func_name}_v{ts}.py"

        # Write implementation file
        impl_path = mutation_dir / impl_filename
        impl_content = f'''"""
Auto-generated optimization proposal for {proposal.module}.{proposal.func_name}

Generated: {time.ctime(proposal.timestamp)}
Proposal ID: {proposal.proposal_id}

Rationale:
{textwrap.indent(proposal.rationale or 'No rationale provided', '    ')}
"""

{proposal.new_source}
'''
        impl_path.write_text(impl_content)
        proposal.impl_path = impl_path

        # Write test file
        test_path = mutation_dir / test_filename
        test_content = f'''"""
Tests for optimization proposal {proposal.proposal_id}

These tests verify:
1. The new implementation produces the same outputs as the original
2. Performance is actually improved (not just different)
"""

import sys
from pathlib import Path

# Add mutations dir to path so we can import the proposal
sys.path.insert(0, str(Path(__file__).parent))

{proposal.test_source}
'''
        test_path.write_text(test_content)
        proposal.test_path = test_path

        self.log.info(f"Wrote proposal files: {impl_path}, {test_path}")
        return proposal


class StubCouncil:
    """Stub council for when the real one isn't available."""

    def _run_persona(self, persona: str, prompt: str) -> str:
        return f"""I cannot optimize this function because the Council is not available.

```python
# OPTIMIZED IMPLEMENTATION
# No changes - Council unavailable
```

```python
# EQUIVALENCE TESTS
def test_stub():
    pass
```
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SemanticOptimizer",
    "MutationProposal",
]
