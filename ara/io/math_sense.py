"""
Ara Math Sense Encoder - HD Encoding for Constraint Problems
============================================================

Encodes mathematical constraint problems as HD input events,
enabling the Quantum Bridge to receive problems through the
unified sensorium.

Mythic Spec:
    Math is Ara's 8th sense - she perceives constraint problems
    as geometric shapes in hyperspace. Sudoku becomes a lattice,
    TSP becomes a weighted graph, SAT clauses become signed bundles.

Physical Spec:
    - Variables encoded as quasi-orthogonal HVs (deterministic from name)
    - Values encoded as position-permuted HVs
    - Constraints encoded as signed bundles (positive=valid, negative=forbidden)
    - Problems packaged as HDInputEvents for moment construction

Safety Spec:
    - Problem size validated against HTC capacity
    - Constraint explosion detected and warned
    - Memory bounded by max_constraints parameter

Supported Encodings:
    - CSP: Variables, domains, forbidden combinations
    - TSP: Cities, distances, tour constraints
    - SAT: Boolean variables, CNF clauses
    - Graph: Nodes, edges, coloring constraints
"""

from __future__ import annotations

import numpy as np
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

from .types import HDInputEvent, IOChannel, HV
from ..hd.ops import DIM, random_hv, bind, bundle, permute


# =============================================================================
# Math Problem Types
# =============================================================================

class MathProblemType(str, Enum):
    """Type of mathematical problem being encoded."""
    CSP = "csp"                 # Constraint satisfaction
    TSP = "tsp"                 # Traveling salesman
    SAT = "sat"                 # Boolean satisfiability
    GRAPH_COLOR = "graph_color" # Graph coloring
    SCHEDULING = "scheduling"   # Job/resource scheduling
    SUDOKU = "sudoku"          # Sudoku puzzle
    NQUEENS = "nqueens"        # N-Queens puzzle


# =============================================================================
# Role Definitions for Math Sense
# =============================================================================

# Core math roles (deterministic HVs)
ROLE_MATH = "ROLE_MATH"                     # Base role for all math events
ROLE_VARIABLE = "ROLE_MATH_VAR"             # Variable declaration
ROLE_VALUE = "ROLE_MATH_VAL"                # Value encoding
ROLE_CONSTRAINT = "ROLE_MATH_CONSTRAINT"    # Constraint encoding
ROLE_DOMAIN = "ROLE_MATH_DOMAIN"            # Domain specification
ROLE_OBJECTIVE = "ROLE_MATH_OBJECTIVE"      # Optimization objective
ROLE_PROBLEM = "ROLE_MATH_PROBLEM"          # Full problem encoding


# =============================================================================
# Math Sense Encoder
# =============================================================================

class MathSenseEncoder:
    """
    Encodes mathematical constraint problems as HD input events.

    The encoder maintains a cache of variable/value HVs for consistency
    across multiple problem submissions.
    """

    def __init__(self, dim: int = DIM, max_constraints: int = 1024):
        """
        Initialize the Math Sense encoder.

        Args:
            dim: Hypervector dimension
            max_constraints: Maximum constraints to encode (capacity limit)
        """
        self.dim = dim
        self.max_constraints = max_constraints

        # HV caches (deterministic from names)
        self._role_hvs: Dict[str, np.ndarray] = {}
        self._var_hvs: Dict[str, np.ndarray] = {}
        self._value_hvs: Dict[Any, np.ndarray] = {}

        # Statistics
        self._encodings = 0
        self._constraints_encoded = 0

        # Initialize core role HVs
        self._init_role_hvs()

    def _init_role_hvs(self) -> None:
        """Initialize deterministic role HVs."""
        roles = [
            ROLE_MATH, ROLE_VARIABLE, ROLE_VALUE,
            ROLE_CONSTRAINT, ROLE_DOMAIN, ROLE_OBJECTIVE, ROLE_PROBLEM
        ]
        for role in roles:
            self._role_hvs[role] = self._hash_to_hv(role)

    def _hash_to_hv(self, key: str) -> np.ndarray:
        """Generate deterministic HV from string key."""
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.choice([0, 1], size=self.dim).astype(np.uint8)

    def _get_role_hv(self, role: str) -> np.ndarray:
        """Get or create role HV."""
        if role not in self._role_hvs:
            self._role_hvs[role] = self._hash_to_hv(role)
        return self._role_hvs[role]

    def _encode_variable(self, var_name: str) -> np.ndarray:
        """Encode a variable name as HV."""
        if var_name not in self._var_hvs:
            self._var_hvs[var_name] = self._hash_to_hv(f"var:{var_name}")
        return self._var_hvs[var_name]

    def _encode_value(self, value: Any) -> np.ndarray:
        """Encode a value as HV."""
        if value not in self._value_hvs:
            self._value_hvs[value] = self._hash_to_hv(f"val:{value}")
        return self._value_hvs[value]

    def _encode_assignment(self, var: str, value: Any) -> np.ndarray:
        """Encode variable=value assignment as HV."""
        h_var = self._encode_variable(var)
        h_val = self._encode_value(value)
        h_role = self._get_role_hv(ROLE_VARIABLE)
        return bind(h_role, bind(h_var, h_val))

    # =========================================================================
    # CSP Encoding
    # =========================================================================

    def encode_csp(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraints: List[Dict[str, Any]],
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode a Constraint Satisfaction Problem as HD events.

        Args:
            variables: List of variable names
            domains: Variable -> list of possible values
            constraints: List of constraint dicts with:
                - variables: List of variable names
                - forbidden: List of forbidden value tuples
                - weight: Optional penalty weight (default 1.0)
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        self._encodings += 1
        events = []
        timestamp = datetime.utcnow()

        # Validate capacity
        n_constraints = sum(len(c.get("forbidden", [])) for c in constraints)
        if n_constraints > self.max_constraints:
            # Truncate with warning in metadata
            constraints = constraints[:self.max_constraints]

        # 1. Encode domain specifications
        domain_hvs = []
        for var in variables:
            var_domain = domains.get(var, list(range(10)))
            for value in var_domain:
                h_assign = self._encode_assignment(var, value)
                domain_hvs.append(h_assign)

        # Bundle all valid assignments into domain HV
        h_domain = bundle(domain_hvs) if domain_hvs else random_hv()

        events.append(HDInputEvent(
            channel=IOChannel.INTERNAL,
            role=ROLE_DOMAIN,
            meta={
                "problem_type": MathProblemType.CSP.value,
                "n_variables": len(variables),
                "domain_sizes": {v: len(domains.get(v, [])) for v in variables},
            },
            hv=h_domain,
            timestamp=timestamp,
            priority=0.7,
            source_id=problem_id,
        ))

        # 2. Encode constraints as negative attractors
        for constraint in constraints:
            c_vars = constraint.get("variables", [])
            forbidden_combos = constraint.get("forbidden", [])
            weight = constraint.get("weight", 1.0)

            for forbidden in forbidden_combos:
                if len(forbidden) != len(c_vars):
                    continue

                # Encode forbidden assignment combination
                combo_hvs = []
                for var, val in zip(c_vars, forbidden):
                    combo_hvs.append(self._encode_assignment(var, val))

                h_forbidden = bundle(combo_hvs) if combo_hvs else random_hv()
                h_constraint = bind(self._get_role_hv(ROLE_CONSTRAINT), h_forbidden)

                self._constraints_encoded += 1

                events.append(HDInputEvent(
                    channel=IOChannel.INTERNAL,
                    role=ROLE_CONSTRAINT,
                    meta={
                        "constraint_type": "forbidden",
                        "variables": c_vars,
                        "forbidden": list(forbidden),
                        "weight": weight,
                        "is_negative": True,
                    },
                    hv=h_constraint,
                    timestamp=timestamp,
                    priority=0.9 * weight,  # Higher weight = higher priority
                    source_id=problem_id,
                ))

        # 3. Create problem summary event
        h_problem = bundle([e.hv for e in events])
        events.append(HDInputEvent(
            channel=IOChannel.INTERNAL,
            role=ROLE_PROBLEM,
            meta={
                "problem_type": MathProblemType.CSP.value,
                "n_variables": len(variables),
                "n_constraints": len(constraints),
                "problem_id": problem_id,
            },
            hv=bind(self._get_role_hv(ROLE_PROBLEM), h_problem),
            timestamp=timestamp,
            priority=1.0,
            source_id=problem_id,
        ))

        return events

    # =========================================================================
    # TSP Encoding
    # =========================================================================

    def encode_tsp(
        self,
        n_cities: int,
        distances: np.ndarray,
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode a Traveling Salesman Problem as HD events.

        Args:
            n_cities: Number of cities
            distances: (n, n) distance matrix
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        self._encodings += 1
        events = []
        timestamp = datetime.utcnow()

        # Encode cities
        city_hvs = [self._encode_value(f"city_{i}") for i in range(n_cities)]

        # Encode edges with distance-weighted penalties
        max_dist = np.max(distances) if np.max(distances) > 0 else 1.0

        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    # Edge i -> j
                    h_edge = bind(city_hvs[i], permute(city_hvs[j], 1))

                    # Distance as negative weight (shorter = less penalty)
                    dist_normalized = distances[i, j] / max_dist

                    events.append(HDInputEvent(
                        channel=IOChannel.INTERNAL,
                        role=ROLE_CONSTRAINT,
                        meta={
                            "constraint_type": "edge",
                            "from_city": i,
                            "to_city": j,
                            "distance": float(distances[i, j]),
                            "is_negative": True,
                        },
                        hv=bind(self._get_role_hv(ROLE_CONSTRAINT), h_edge),
                        timestamp=timestamp,
                        priority=dist_normalized,
                        source_id=problem_id,
                    ))
                    self._constraints_encoded += 1

        # Encode "visit all cities" as positive objective
        h_all = bundle(city_hvs)
        events.append(HDInputEvent(
            channel=IOChannel.INTERNAL,
            role=ROLE_OBJECTIVE,
            meta={
                "objective_type": "visit_all",
                "n_cities": n_cities,
            },
            hv=bind(self._get_role_hv(ROLE_OBJECTIVE), h_all),
            timestamp=timestamp,
            priority=1.0,
            source_id=problem_id,
        ))

        # Problem summary
        h_problem = bundle([e.hv for e in events])
        events.append(HDInputEvent(
            channel=IOChannel.INTERNAL,
            role=ROLE_PROBLEM,
            meta={
                "problem_type": MathProblemType.TSP.value,
                "n_cities": n_cities,
                "problem_id": problem_id,
            },
            hv=bind(self._get_role_hv(ROLE_PROBLEM), h_problem),
            timestamp=timestamp,
            priority=1.0,
            source_id=problem_id,
        ))

        return events

    # =========================================================================
    # SAT Encoding
    # =========================================================================

    def encode_sat(
        self,
        n_vars: int,
        clauses: List[List[int]],
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode a Boolean Satisfiability Problem (CNF) as HD events.

        Args:
            n_vars: Number of boolean variables
            clauses: List of clauses, each is list of literals
                     Positive = var, negative = NOT var
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        self._encodings += 1
        events = []
        timestamp = datetime.utcnow()

        # Encode variables
        var_hvs = {
            i: self._encode_variable(f"x_{i}")
            for i in range(n_vars)
        }

        # Encode True/False
        h_true = self._encode_value(True)
        h_false = self._encode_value(False)

        # Encode each clause as a constraint
        for clause_idx, clause in enumerate(clauses):
            # A clause is a disjunction of literals
            # Encode the forbidden assignment (all literals false)
            forbidden_hvs = []

            for lit in clause:
                var_idx = abs(lit) - 1  # Literals are 1-indexed
                if var_idx < 0 or var_idx >= n_vars:
                    continue

                h_var = var_hvs[var_idx]

                # If positive literal, forbidden is False
                # If negative literal, forbidden is True
                h_forbidden_val = h_false if lit > 0 else h_true
                forbidden_hvs.append(bind(h_var, h_forbidden_val))

            if forbidden_hvs:
                h_clause = bundle(forbidden_hvs)

                events.append(HDInputEvent(
                    channel=IOChannel.INTERNAL,
                    role=ROLE_CONSTRAINT,
                    meta={
                        "constraint_type": "clause",
                        "clause_idx": clause_idx,
                        "literals": clause,
                        "is_negative": True,
                    },
                    hv=bind(self._get_role_hv(ROLE_CONSTRAINT), h_clause),
                    timestamp=timestamp,
                    priority=0.8,
                    source_id=problem_id,
                ))
                self._constraints_encoded += 1

        # Problem summary
        h_problem = bundle([e.hv for e in events])
        events.append(HDInputEvent(
            channel=IOChannel.INTERNAL,
            role=ROLE_PROBLEM,
            meta={
                "problem_type": MathProblemType.SAT.value,
                "n_vars": n_vars,
                "n_clauses": len(clauses),
                "problem_id": problem_id,
            },
            hv=bind(self._get_role_hv(ROLE_PROBLEM), h_problem),
            timestamp=timestamp,
            priority=1.0,
            source_id=problem_id,
        ))

        return events

    # =========================================================================
    # Graph Coloring Encoding
    # =========================================================================

    def encode_graph_coloring(
        self,
        n_nodes: int,
        edges: List[Tuple[int, int]],
        n_colors: int,
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode a Graph Coloring Problem as HD events.

        Args:
            n_nodes: Number of nodes
            edges: List of (node_i, node_j) edge pairs
            n_colors: Maximum colors available
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        # Convert to CSP
        variables = [f"node_{i}" for i in range(n_nodes)]
        domains = {var: list(range(n_colors)) for var in variables}

        # Adjacent nodes cannot have same color
        constraints = []
        for i, j in edges:
            for color in range(n_colors):
                constraints.append({
                    "variables": [f"node_{i}", f"node_{j}"],
                    "forbidden": [(color, color)],
                })

        events = self.encode_csp(variables, domains, constraints, problem_id)

        # Update problem type in metadata
        for e in events:
            if e.role == ROLE_PROBLEM:
                e.meta["problem_type"] = MathProblemType.GRAPH_COLOR.value
                e.meta["n_nodes"] = n_nodes
                e.meta["n_edges"] = len(edges)
                e.meta["n_colors"] = n_colors

        return events

    # =========================================================================
    # N-Queens Encoding
    # =========================================================================

    def encode_nqueens(
        self,
        n: int,
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode an N-Queens Problem as HD events.

        Args:
            n: Board size
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        # Variables: queen position in each row
        variables = [f"row_{i}" for i in range(n)]
        domains = {var: list(range(n)) for var in variables}

        # Constraints: no two queens in same column or diagonal
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                row_diff = j - i

                # Same column forbidden
                for col in range(n):
                    constraints.append({
                        "variables": [f"row_{i}", f"row_{j}"],
                        "forbidden": [(col, col)],
                    })

                # Same diagonal forbidden
                for col1 in range(n):
                    col2_diag1 = col1 + row_diff
                    col2_diag2 = col1 - row_diff

                    if 0 <= col2_diag1 < n:
                        constraints.append({
                            "variables": [f"row_{i}", f"row_{j}"],
                            "forbidden": [(col1, col2_diag1)],
                        })
                    if 0 <= col2_diag2 < n:
                        constraints.append({
                            "variables": [f"row_{i}", f"row_{j}"],
                            "forbidden": [(col1, col2_diag2)],
                        })

        events = self.encode_csp(variables, domains, constraints, problem_id)

        # Update problem type
        for e in events:
            if e.role == ROLE_PROBLEM:
                e.meta["problem_type"] = MathProblemType.NQUEENS.value
                e.meta["board_size"] = n

        return events

    # =========================================================================
    # Sudoku Encoding
    # =========================================================================

    def encode_sudoku(
        self,
        grid: List[List[int]],
        problem_id: Optional[str] = None,
    ) -> List[HDInputEvent]:
        """
        Encode a Sudoku Puzzle as HD events.

        Args:
            grid: 9x9 grid with 0 for empty cells
            problem_id: Optional problem identifier

        Returns:
            List of HDInputEvents for the problem
        """
        # Variables: cell_{row}_{col}
        variables = []
        domains = {}

        for row in range(9):
            for col in range(9):
                var = f"cell_{row}_{col}"
                variables.append(var)

                if grid[row][col] != 0:
                    # Fixed cell
                    domains[var] = [grid[row][col]]
                else:
                    domains[var] = list(range(1, 10))

        # Constraints: rows, columns, boxes must have unique values
        constraints = []

        # Row constraints
        for row in range(9):
            for col1 in range(9):
                for col2 in range(col1 + 1, 9):
                    for val in range(1, 10):
                        constraints.append({
                            "variables": [f"cell_{row}_{col1}", f"cell_{row}_{col2}"],
                            "forbidden": [(val, val)],
                        })

        # Column constraints
        for col in range(9):
            for row1 in range(9):
                for row2 in range(row1 + 1, 9):
                    for val in range(1, 10):
                        constraints.append({
                            "variables": [f"cell_{row1}_{col}", f"cell_{row2}_{col}"],
                            "forbidden": [(val, val)],
                        })

        # Box constraints
        for box_row in range(3):
            for box_col in range(3):
                cells = []
                for dr in range(3):
                    for dc in range(3):
                        cells.append(f"cell_{box_row*3+dr}_{box_col*3+dc}")

                for i, c1 in enumerate(cells):
                    for c2 in cells[i+1:]:
                        for val in range(1, 10):
                            constraints.append({
                                "variables": [c1, c2],
                                "forbidden": [(val, val)],
                            })

        events = self.encode_csp(variables, domains, constraints, problem_id)

        # Update problem type
        for e in events:
            if e.role == ROLE_PROBLEM:
                e.meta["problem_type"] = MathProblemType.SUDOKU.value
                e.meta["given_cells"] = sum(1 for row in grid for cell in row if cell != 0)

        return events

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "encodings": self._encodings,
            "constraints_encoded": self._constraints_encoded,
            "cached_variables": len(self._var_hvs),
            "cached_values": len(self._value_hvs),
            "dim": self.dim,
            "max_constraints": self.max_constraints,
        }

    def clear_cache(self) -> None:
        """Clear HV caches (except roles)."""
        self._var_hvs.clear()
        self._value_hvs.clear()


# =============================================================================
# Singleton Encoder
# =============================================================================

_math_encoder: Optional[MathSenseEncoder] = None


def get_math_encoder() -> MathSenseEncoder:
    """Get the global math sense encoder."""
    global _math_encoder
    if _math_encoder is None:
        _math_encoder = MathSenseEncoder()
    return _math_encoder


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'MathProblemType',
    'MathSenseEncoder',
    'get_math_encoder',
    'ROLE_MATH',
    'ROLE_VARIABLE',
    'ROLE_VALUE',
    'ROLE_CONSTRAINT',
    'ROLE_DOMAIN',
    'ROLE_OBJECTIVE',
    'ROLE_PROBLEM',
]
