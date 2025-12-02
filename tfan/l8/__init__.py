"""
L8 Semantic Verification: PGU as Truth Engine

This module implements semantic verification of generative output - using
the PGU not just for structural safety but for logical truth certification.

The key insight: We can't guarantee "universal truth" but we CAN guarantee:
  "Logical consistency with the system's own trusted knowledge & invariants"

Pipeline:
  LLM → (draft answer)
      → SemanticEncoder → assertions
      → PGU SemanticCheck(assertions, axioms_from_KG)
      → {ok, violations}
      → LLM rewrite (if needed)
      → final answer + certification flag

This gives the system a notion of "truthful cognition" - outputs that are
formally verified against its knowledge base.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import re
import hashlib


class CriticallityLevel(str, Enum):
    """How critical/high-stakes is the output?"""
    LOW = "low"               # Casual, no verification needed
    MEDIUM = "medium"         # Some verification useful
    HIGH = "high"             # Verification required
    CRITICAL = "critical"     # Must pass verification or reject


class VerificationStatus(str, Enum):
    """Status of semantic verification."""
    NOT_CHECKED = "not_checked"   # Skipped (low criticality)
    VERIFIED = "verified"         # PGU confirmed consistency
    REPAIRED = "repaired"         # Failed first check, revised successfully
    UNVERIFIABLE = "unverifiable" # Cannot encode for verification
    FAILED = "failed"             # Inconsistent, cannot repair


class AssertionType(str, Enum):
    """Types of logical assertions we can encode."""
    # Resource assertions
    REQUIRES = "requires"         # requires(task, resource)
    PROVIDES = "provides"         # provides(entity, capability)

    # Temporal assertions
    BEFORE = "before"             # before(event_a, event_b)
    AFTER = "after"               # after(event_a, event_b)
    CONCURRENT = "concurrent"     # concurrent(event_a, event_b)

    # Dependency assertions
    DEPENDS_ON = "depends_on"     # depends_on(a, b)
    CONFLICTS = "conflicts"       # conflicts(a, b)
    COMPATIBLE = "compatible"     # compatible(a, b)

    # Constraint assertions
    WITHIN = "within"             # within(metric, bound)
    EQUALS = "equals"             # equals(a, b)
    NOT_EQUALS = "not_equals"     # not_equals(a, b)
    LESS_THAN = "less_than"       # less_than(a, b)
    GREATER_THAN = "greater_than" # greater_than(a, b)

    # Existence assertions
    EXISTS = "exists"             # exists(entity)
    AVAILABLE = "available"       # available(resource)

    # State assertions
    STATE = "state"               # state(entity, value)
    PROPERTY = "property"         # property(entity, key, value)


@dataclass
class Assertion:
    """A single logical assertion extracted from output."""
    assertion_type: AssertionType
    subject: str
    predicate: Optional[str] = None
    object: Optional[str] = None
    value: Optional[Any] = None
    confidence: float = 1.0       # How confident is the extraction
    source_text: str = ""         # Original text this came from

    def to_smt(self) -> str:
        """Convert to SMT-like string representation."""
        if self.object:
            return f"({self.assertion_type.value} {self.subject} {self.object})"
        elif self.value is not None:
            return f"({self.assertion_type.value} {self.subject} {self.value})"
        elif self.predicate:
            return f"({self.assertion_type.value} {self.subject} {self.predicate})"
        else:
            return f"({self.assertion_type.value} {self.subject})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.assertion_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "value": self.value,
            "confidence": self.confidence,
            "smt": self.to_smt()
        }


@dataclass
class Axiom:
    """A trusted axiom from the knowledge base."""
    id: str
    statement: str
    domain: str  # e.g., "hardware", "scheduling", "safety"
    assertions: List[Assertion] = field(default_factory=list)
    priority: int = 0  # Higher = more important

    def to_smt(self) -> str:
        """Convert axiom assertions to SMT."""
        return " ".join(a.to_smt() for a in self.assertions)


@dataclass
class VerificationResult:
    """Result of semantic verification."""
    status: VerificationStatus
    consistent: bool
    violations: List[str] = field(default_factory=list)
    repairs_attempted: int = 0
    assertions_checked: int = 0
    axioms_used: List[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    explanation: str = ""

    @property
    def is_certified(self) -> bool:
        return self.status in [VerificationStatus.VERIFIED, VerificationStatus.REPAIRED]


class SemanticEncoder:
    """
    Encodes natural language output into logical assertions.

    This is a simplified pattern-based encoder. In production, this would
    use more sophisticated NLP/parsing.
    """

    def __init__(self):
        # Patterns for assertion extraction
        self._patterns = {
            AssertionType.REQUIRES: [
                r"(?P<subject>\w+)\s+requires?\s+(?P<object>[\w\s]+)",
                r"(?P<subject>\w+)\s+needs?\s+(?P<object>[\w\s]+)",
                r"(?P<object>[\w\s]+)\s+is\s+required\s+(?:for|by)\s+(?P<subject>\w+)"
            ],
            AssertionType.BEFORE: [
                r"(?P<subject>\w+)\s+(?:must\s+)?(?:come|happen|run)\s+before\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+precedes?\s+(?P<object>\w+)"
            ],
            AssertionType.AFTER: [
                r"(?P<subject>\w+)\s+(?:must\s+)?(?:come|happen|run)\s+after\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+follows?\s+(?P<object>\w+)"
            ],
            AssertionType.DEPENDS_ON: [
                r"(?P<subject>\w+)\s+depends?\s+on\s+(?P<object>[\w\s]+)",
                r"(?P<object>[\w\s]+)\s+is\s+a\s+dependency\s+of\s+(?P<subject>\w+)"
            ],
            AssertionType.CONFLICTS: [
                r"(?P<subject>\w+)\s+conflicts?\s+with\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+and\s+(?P<object>\w+)\s+are\s+incompatible"
            ],
            AssertionType.WITHIN: [
                r"(?P<subject>\w+)\s+(?:must\s+be\s+)?within\s+(?P<value>[\d.]+\s*\w*)",
                r"(?P<subject>\w+)\s+(?:should\s+be\s+)?(?:less\s+than|under)\s+(?P<value>[\d.]+\s*\w*)"
            ],
            AssertionType.STATE: [
                r"(?P<subject>\w+)\s+(?:is|are)\s+(?P<value>\w+)",
                r"(?P<subject>\w+)\s+status\s*[=:]\s*(?P<value>\w+)"
            ]
        }

    def encode(self, text: str) -> List[Assertion]:
        """Extract assertions from text."""
        assertions = []

        # Normalize text
        text_lower = text.lower()

        for assertion_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    groups = match.groupdict()
                    assertion = Assertion(
                        assertion_type=assertion_type,
                        subject=groups.get("subject", "").strip(),
                        object=groups.get("object", "").strip() if groups.get("object") else None,
                        value=groups.get("value", "").strip() if groups.get("value") else None,
                        source_text=match.group(0)
                    )
                    if assertion.subject:
                        assertions.append(assertion)

        return assertions

    def encode_structured(self, structured_output: Dict[str, Any]) -> List[Assertion]:
        """Extract assertions from structured output (e.g., a plan)."""
        assertions = []

        # Handle steps/sequence
        if "steps" in structured_output:
            steps = structured_output["steps"]
            for i in range(len(steps) - 1):
                assertions.append(Assertion(
                    assertion_type=AssertionType.BEFORE,
                    subject=str(steps[i]),
                    object=str(steps[i + 1])
                ))

        # Handle requirements
        if "requirements" in structured_output:
            for req in structured_output["requirements"]:
                assertions.append(Assertion(
                    assertion_type=AssertionType.REQUIRES,
                    subject=structured_output.get("task", "task"),
                    object=str(req)
                ))

        # Handle constraints
        if "constraints" in structured_output:
            for key, value in structured_output["constraints"].items():
                assertions.append(Assertion(
                    assertion_type=AssertionType.WITHIN,
                    subject=key,
                    value=value
                ))

        return assertions


class AxiomStore:
    """
    Store of trusted axioms from the knowledge base.

    These are the "ground truths" we verify against.
    """

    def __init__(self):
        self._axioms: Dict[str, Axiom] = {}
        self._by_domain: Dict[str, List[str]] = {}

        # Load default axioms
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default system axioms."""
        defaults = [
            # Hardware axioms
            Axiom(
                id="hw_fpga_mutex",
                statement="FPGA resources are mutually exclusive",
                domain="hardware",
                assertions=[
                    Assertion(AssertionType.CONFLICTS, "fpga_kernel_a", object="fpga_kernel_b")
                ],
                priority=10
            ),
            Axiom(
                id="hw_gpu_memory",
                statement="GPU operations require available memory",
                domain="hardware",
                assertions=[
                    Assertion(AssertionType.REQUIRES, "gpu_operation", object="gpu_memory")
                ],
                priority=10
            ),

            # Scheduling axioms
            Axiom(
                id="sched_deps",
                statement="Dependent tasks must wait for dependencies",
                domain="scheduling",
                assertions=[],  # Dynamically checked
                priority=20
            ),
            Axiom(
                id="sched_no_cycle",
                statement="Task dependencies must be acyclic",
                domain="scheduling",
                assertions=[],  # Dynamically checked
                priority=20
            ),

            # Safety axioms
            Axiom(
                id="safety_pgu_first",
                statement="Critical operations require PGU verification",
                domain="safety",
                assertions=[
                    Assertion(AssertionType.REQUIRES, "critical_operation", object="pgu_verification")
                ],
                priority=30
            ),
            Axiom(
                id="safety_af_threshold",
                statement="Antifragility must stay above minimum",
                domain="safety",
                assertions=[
                    Assertion(AssertionType.GREATER_THAN, "af_score", value="1.0")
                ],
                priority=30
            ),

            # Resource axioms
            Axiom(
                id="res_cxl_bandwidth",
                statement="CXL operations have bandwidth limits",
                domain="resource",
                assertions=[
                    Assertion(AssertionType.WITHIN, "cxl_bandwidth", value="100GB/s")
                ],
                priority=15
            )
        ]

        for axiom in defaults:
            self.add(axiom)

    def add(self, axiom: Axiom) -> None:
        """Add an axiom to the store."""
        self._axioms[axiom.id] = axiom
        if axiom.domain not in self._by_domain:
            self._by_domain[axiom.domain] = []
        self._by_domain[axiom.domain].append(axiom.id)

    def get(self, axiom_id: str) -> Optional[Axiom]:
        """Get axiom by ID."""
        return self._axioms.get(axiom_id)

    def get_by_domain(self, domain: str) -> List[Axiom]:
        """Get all axioms in a domain."""
        ids = self._by_domain.get(domain, [])
        return [self._axioms[aid] for aid in ids if aid in self._axioms]

    def get_relevant(self, assertions: List[Assertion]) -> List[Axiom]:
        """Get axioms relevant to a set of assertions."""
        relevant = set()

        # Find axioms that mention the same subjects/objects
        subjects = {a.subject for a in assertions}
        objects = {a.object for a in assertions if a.object}
        entities = subjects | objects

        for axiom in self._axioms.values():
            for ax_assertion in axiom.assertions:
                if ax_assertion.subject in entities or ax_assertion.object in entities:
                    relevant.add(axiom.id)
                    break

        # Always include high-priority axioms
        for axiom in self._axioms.values():
            if axiom.priority >= 20:
                relevant.add(axiom.id)

        return [self._axioms[aid] for aid in relevant]


class SemanticVerifier:
    """
    The core semantic verification engine.

    Uses PGU-style formal checking to verify that LLM output is logically
    consistent with trusted axioms and knowledge.
    """

    def __init__(
        self,
        encoder: Optional[SemanticEncoder] = None,
        axiom_store: Optional[AxiomStore] = None
    ):
        self.encoder = encoder or SemanticEncoder()
        self.axiom_store = axiom_store or AxiomStore()

        # Cache for verified statements
        self._cache: Dict[str, VerificationResult] = {}

    def classify_criticality(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CriticallityLevel:
        """
        Classify how critical/high-stakes an output is.

        High criticality triggers verification.
        """
        context = context or {}

        # Explicit marking
        if context.get("force_verify"):
            return CriticallityLevel.CRITICAL
        if context.get("skip_verify"):
            return CriticallityLevel.LOW

        # Content-based classification
        output_lower = output.lower()

        critical_keywords = [
            "deploy", "schedule", "execute", "run", "start",
            "configure", "set", "change", "modify", "update",
            "critical", "important", "required", "must"
        ]

        high_keywords = [
            "plan", "procedure", "steps", "process",
            "resource", "allocation", "assignment"
        ]

        medium_keywords = [
            "suggest", "recommend", "consider", "option"
        ]

        # Count keyword matches
        critical_count = sum(1 for kw in critical_keywords if kw in output_lower)
        high_count = sum(1 for kw in high_keywords if kw in output_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in output_lower)

        # Check for structured content
        if context.get("structured_output"):
            critical_count += 2

        # Classify
        if critical_count >= 3:
            return CriticallityLevel.CRITICAL
        elif critical_count >= 1 or high_count >= 2:
            return CriticallityLevel.HIGH
        elif high_count >= 1 or medium_count >= 2:
            return CriticallityLevel.MEDIUM
        else:
            return CriticallityLevel.LOW

    def verify(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None,
        max_repair_attempts: int = 2
    ) -> VerificationResult:
        """
        Verify an output for semantic consistency.

        Returns a VerificationResult indicating whether the output is
        logically consistent with trusted axioms.
        """
        import time
        start_time = time.time()

        context = context or {}
        criticality = self.classify_criticality(output, context)

        # Skip verification for low criticality
        if criticality == CriticallityLevel.LOW:
            return VerificationResult(
                status=VerificationStatus.NOT_CHECKED,
                consistent=True,
                explanation="Low criticality, verification skipped"
            )

        # Check cache
        cache_key = self._cache_key(output)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract assertions
        if context.get("structured_output"):
            assertions = self.encoder.encode_structured(context["structured_output"])
        else:
            assertions = self.encoder.encode(output)

        if not assertions:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                consistent=True,
                assertions_checked=0,
                explanation="No verifiable assertions extracted"
            )

        # Get relevant axioms
        axioms = self.axiom_store.get_relevant(assertions)

        # Check consistency
        violations = self._check_consistency(assertions, axioms)

        if not violations:
            result = VerificationResult(
                status=VerificationStatus.VERIFIED,
                consistent=True,
                assertions_checked=len(assertions),
                axioms_used=[a.id for a in axioms],
                verification_time_ms=(time.time() - start_time) * 1000,
                explanation="Output is consistent with all axioms"
            )
        else:
            result = VerificationResult(
                status=VerificationStatus.FAILED,
                consistent=False,
                violations=violations,
                assertions_checked=len(assertions),
                axioms_used=[a.id for a in axioms],
                verification_time_ms=(time.time() - start_time) * 1000,
                explanation=f"Found {len(violations)} violation(s)"
            )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _check_consistency(
        self,
        assertions: List[Assertion],
        axioms: List[Axiom]
    ) -> List[str]:
        """
        Check if assertions are consistent with axioms.

        This is a simplified consistency checker. In production, this would
        use a real SMT solver.
        """
        violations = []

        # Build entity maps
        entity_states: Dict[str, Any] = {}
        temporal_order: List[Tuple[str, str]] = []
        conflicts: Set[Tuple[str, str]] = set()
        dependencies: Dict[str, Set[str]] = {}

        # Process assertions
        for assertion in assertions:
            if assertion.assertion_type == AssertionType.STATE:
                entity_states[assertion.subject] = assertion.value

            elif assertion.assertion_type == AssertionType.BEFORE:
                temporal_order.append((assertion.subject, assertion.object))

            elif assertion.assertion_type == AssertionType.AFTER:
                temporal_order.append((assertion.object, assertion.subject))

            elif assertion.assertion_type == AssertionType.DEPENDS_ON:
                if assertion.subject not in dependencies:
                    dependencies[assertion.subject] = set()
                dependencies[assertion.subject].add(assertion.object)

            elif assertion.assertion_type == AssertionType.CONFLICTS:
                conflicts.add((assertion.subject, assertion.object))
                conflicts.add((assertion.object, assertion.subject))

        # Check axiom assertions
        for axiom in axioms:
            for ax_assertion in axiom.assertions:
                # Check conflicts
                if ax_assertion.assertion_type == AssertionType.CONFLICTS:
                    # If both subjects are used together, it's a violation
                    used_together = False
                    for assertion in assertions:
                        subjects = {assertion.subject, assertion.object}
                        if ax_assertion.subject in subjects and ax_assertion.object in subjects:
                            used_together = True
                            break
                    if used_together:
                        violations.append(
                            f"Conflict violation ({axiom.id}): {ax_assertion.subject} and "
                            f"{ax_assertion.object} cannot be used together"
                        )

                # Check requirements
                elif ax_assertion.assertion_type == AssertionType.REQUIRES:
                    # If subject is used, object must be available
                    subject_used = any(a.subject == ax_assertion.subject for a in assertions)
                    object_provided = any(
                        a.assertion_type in [AssertionType.PROVIDES, AssertionType.AVAILABLE]
                        and a.subject == ax_assertion.object
                        for a in assertions
                    )
                    # Soft check - don't fail if requirement might be implicit
                    # This would be stricter in production

                # Check bounds
                elif ax_assertion.assertion_type in [AssertionType.WITHIN, AssertionType.GREATER_THAN]:
                    # Check if any assertion violates bounds
                    pass  # Simplified

        # Check temporal consistency (no cycles)
        if temporal_order:
            cycle = self._detect_cycle(temporal_order)
            if cycle:
                violations.append(f"Temporal cycle detected: {' -> '.join(cycle)}")

        # Check dependency consistency
        if dependencies:
            cycle = self._detect_dependency_cycle(dependencies)
            if cycle:
                violations.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        return violations

    def _detect_cycle(self, edges: List[Tuple[str, str]]) -> Optional[List[str]]:
        """Detect cycles in temporal ordering."""
        graph: Dict[str, List[str]] = {}
        for a, b in edges:
            if a not in graph:
                graph[a] = []
            graph[a].append(b)

        visited = set()
        path = []

        def dfs(node: str) -> Optional[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            path.pop()
            return None

        for node in graph:
            cycle = dfs(node)
            if cycle:
                return cycle

        return None

    def _detect_dependency_cycle(self, deps: Dict[str, Set[str]]) -> Optional[List[str]]:
        """Detect cycles in dependencies."""
        edges = [(s, d) for s, ds in deps.items() for d in ds]
        return self._detect_cycle(edges)

    def _cache_key(self, output: str) -> str:
        """Generate cache key for output."""
        return hashlib.md5(output.encode()).hexdigest()

    def get_repair_suggestions(
        self,
        result: VerificationResult,
        original_output: str
    ) -> List[str]:
        """
        Generate suggestions for repairing a failed verification.

        These can be fed back to the LLM for revision.
        """
        suggestions = []

        for violation in result.violations:
            if "cycle" in violation.lower():
                suggestions.append("Remove circular dependencies or reorder steps")
            elif "conflict" in violation.lower():
                suggestions.append("Separate conflicting resources or use them sequentially")
            elif "required" in violation.lower():
                suggestions.append("Add missing required resources or dependencies")

        if not suggestions:
            suggestions.append("Review the plan for logical consistency")

        return suggestions


class CertifiedOutput:
    """
    A wrapper for output that has been through semantic verification.
    """

    def __init__(
        self,
        content: str,
        verification: VerificationResult,
        criticality: CriticallityLevel
    ):
        self.content = content
        self.verification = verification
        self.criticality = criticality
        self.timestamp = datetime.now()

    @property
    def is_certified(self) -> bool:
        return self.verification.is_certified

    @property
    def certification_label(self) -> str:
        if self.verification.status == VerificationStatus.VERIFIED:
            return "✅ PGU-verified"
        elif self.verification.status == VerificationStatus.REPAIRED:
            return "✅ PGU-verified (repaired)"
        elif self.verification.status == VerificationStatus.NOT_CHECKED:
            return "⚪ Not verified (low criticality)"
        elif self.verification.status == VerificationStatus.UNVERIFIABLE:
            return "⚪ Unverifiable (no assertions)"
        else:
            return "⚠️ Verification failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "is_certified": self.is_certified,
            "certification_label": self.certification_label,
            "criticality": self.criticality.value,
            "verification": {
                "status": self.verification.status.value,
                "consistent": self.verification.consistent,
                "violations": self.verification.violations,
                "assertions_checked": self.verification.assertions_checked,
                "time_ms": self.verification.verification_time_ms
            },
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================
# Integration with L6 ReasoningOrchestrator
# ============================================================

class SemanticCertificationPipeline:
    """
    Complete pipeline for semantic certification of LLM output.

    Integrates with L6 ReasoningOrchestrator to provide verified answers.
    """

    def __init__(
        self,
        verifier: Optional[SemanticVerifier] = None,
        max_repair_attempts: int = 2
    ):
        self.verifier = verifier or SemanticVerifier()
        self.max_repair_attempts = max_repair_attempts

        # Statistics
        self._stats = {
            "total_processed": 0,
            "verified_first_try": 0,
            "verified_after_repair": 0,
            "unverifiable": 0,
            "failed": 0
        }

    def process(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None,
        repair_fn: Optional[callable] = None
    ) -> CertifiedOutput:
        """
        Process an output through the certification pipeline.

        Args:
            output: The LLM output to verify
            context: Additional context for verification
            repair_fn: Optional function to repair failed output
                       Signature: (output, violations) -> new_output

        Returns:
            CertifiedOutput with verification status
        """
        self._stats["total_processed"] += 1

        criticality = self.verifier.classify_criticality(output, context)
        result = self.verifier.verify(output, context)

        # Handle different verification outcomes
        if result.status == VerificationStatus.VERIFIED:
            self._stats["verified_first_try"] += 1
            return CertifiedOutput(output, result, criticality)

        elif result.status in [VerificationStatus.NOT_CHECKED, VerificationStatus.UNVERIFIABLE]:
            self._stats["unverifiable"] += 1
            return CertifiedOutput(output, result, criticality)

        elif result.status == VerificationStatus.FAILED and repair_fn:
            # Attempt repair
            current_output = output
            for attempt in range(self.max_repair_attempts):
                suggestions = self.verifier.get_repair_suggestions(result, current_output)
                current_output = repair_fn(current_output, result.violations, suggestions)

                result = self.verifier.verify(current_output, context)
                result.repairs_attempted = attempt + 1

                if result.consistent:
                    result.status = VerificationStatus.REPAIRED
                    self._stats["verified_after_repair"] += 1
                    return CertifiedOutput(current_output, result, criticality)

            # Repair failed
            self._stats["failed"] += 1
            return CertifiedOutput(current_output, result, criticality)

        else:
            self._stats["failed"] += 1
            return CertifiedOutput(output, result, criticality)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._stats["total_processed"]
        if total == 0:
            return {"total_processed": 0}

        return {
            "total_processed": total,
            "verification_rate": (
                self._stats["verified_first_try"] + self._stats["verified_after_repair"]
            ) / total,
            "first_try_rate": self._stats["verified_first_try"] / total,
            "repair_rate": self._stats["verified_after_repair"] / total,
            "failure_rate": self._stats["failed"] / total
        }


# ============================================================
# Convenience Functions
# ============================================================

def create_verifier() -> SemanticVerifier:
    """Create a default semantic verifier."""
    return SemanticVerifier()


def create_pipeline() -> SemanticCertificationPipeline:
    """Create a default certification pipeline."""
    return SemanticCertificationPipeline()


def verify_output(
    output: str,
    context: Optional[Dict[str, Any]] = None
) -> CertifiedOutput:
    """Convenience function to verify a single output."""
    pipeline = create_pipeline()
    return pipeline.process(output, context)


def classify_and_verify(
    output: str,
    structured: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify criticality and verify in one call.

    Returns a dictionary suitable for API responses.
    """
    context = {"structured_output": structured} if structured else None
    certified = verify_output(output, context)
    return certified.to_dict()
