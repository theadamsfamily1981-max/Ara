#!/usr/bin/env python
"""
Formula Normalizer for PGU TurboCache

Implements alpha-renaming to canonicalize formulas for cache lookup.
Two formulas that differ only in variable names/order will map to
the same canonical form.

Example:
    (x > y) ∧ (y < 10)  →  (x1 > x2) ∧ (x2 < 10)
    (a > b) ∧ (b < 10)  →  (x1 > x2) ∧ (x2 < 10)  # same canonical form

This enables cache hits even when variable names change across queries.

Usage:
    canon, mapping = alpha_rename("(x > y) and (y < 10)")
    # canon: "(x1 > x2) and (x2 < 10)"
    # mapping: {"x": "x1", "y": "x2"}
"""

from typing import Dict, Tuple, Set, List
import re
from collections import OrderedDict


def alpha_rename(formula: str) -> Tuple[str, Dict[str, str]]:
    """
    Canonicalize variable symbols to x1, x2, ... by order of appearance.

    Args:
        formula: Formula string with variable names

    Returns:
        canon: Canonicalized formula
        mapping: Dict from original var names to canonical names

    Examples:
        >>> alpha_rename("(a > b) and (b < 10)")
        ('(x1 > x2) and (x2 < 10)', {'a': 'x1', 'b': 'x2'})

        >>> alpha_rename("(foo > bar) and (bar < 10)")
        ('(x1 > x2) and (x2 < 10)', {'foo': 'x1', 'bar': 'x2'})
    """
    syms = OrderedDict()
    nxt = 1

    def repl(m):
        nonlocal nxt
        v = m.group(0)

        # Skip reserved keywords
        if v in RESERVED_KEYWORDS:
            return v

        if v not in syms:
            syms[v] = f"x{nxt}"
            nxt += 1

        return syms[v]

    # Match variable identifiers (start with letter or underscore)
    canon = re.sub(r'\b[A-Za-z_]\w*\b', repl, formula)

    return canon, dict(syms)


def normalize_formula(
    formula: str,
    assumptions: List[str] = None,
    sort_conjuncts: bool = True
) -> Tuple[str, Dict]:
    """
    Fully normalize formula for cache lookup.

    Normalization steps:
    1. Alpha-rename variables
    2. Sort conjuncts (if requested)
    3. Canonicalize whitespace
    4. Sort assumptions

    Args:
        formula: Formula string
        assumptions: List of assumption formulas
        sort_conjuncts: Whether to sort conjuncts in AND clauses

    Returns:
        normalized: Normalized formula string
        metadata: Dict with normalization info
    """
    # Step 1: Alpha-rename
    canon, var_mapping = alpha_rename(formula)

    # Step 2: Canonicalize whitespace
    canon = canonicalize_whitespace(canon)

    # Step 3: Sort conjuncts (makes "a ∧ b" == "b ∧ a")
    if sort_conjuncts:
        canon = sort_and_clauses(canon)

    # Step 4: Normalize assumptions
    norm_assumptions = []
    if assumptions:
        for assume in assumptions:
            assume_canon, _ = alpha_rename(assume)
            assume_canon = canonicalize_whitespace(assume_canon)
            norm_assumptions.append(assume_canon)

        # Sort assumptions for order-independence
        norm_assumptions.sort()

    metadata = {
        'var_mapping': var_mapping,
        'num_vars': len(var_mapping),
        'assumptions': norm_assumptions
    }

    return canon, metadata


def canonicalize_whitespace(s: str) -> str:
    """
    Normalize whitespace to single spaces.

    Args:
        s: String with arbitrary whitespace

    Returns:
        Canonicalized string
    """
    # Replace all whitespace sequences with single space
    s = re.sub(r'\s+', ' ', s)

    # Remove spaces around operators
    s = re.sub(r'\s*([()><=!&|+\-*/])\s*', r'\1', s)

    return s.strip()


def sort_and_clauses(formula: str) -> str:
    """
    Sort conjuncts in AND clauses for order-independence.

    This is a simple implementation that handles basic cases.
    For complex nested formulas, a full AST would be better.

    Args:
        formula: Formula string

    Returns:
        Formula with sorted AND clauses
    """
    # Simple heuristic: split on ' and ', sort, rejoin
    # This works for flat conjunctions like "(a>b) and (c<d) and (e==f)"

    # Check if formula contains ' and '
    if ' and ' in formula.lower():
        # Split on and
        parts = re.split(r'\s+and\s+', formula, flags=re.IGNORECASE)

        # Sort parts
        parts.sort()

        # Rejoin
        return ' and '.join(parts)

    return formula


def extract_variables(formula: str) -> Set[str]:
    """
    Extract all variable names from formula.

    Args:
        formula: Formula string

    Returns:
        Set of variable names
    """
    # Find all identifiers
    matches = re.findall(r'\b[A-Za-z_]\w*\b', formula)

    # Filter out reserved keywords
    variables = {m for m in matches if m not in RESERVED_KEYWORDS}

    return variables


def reverse_rename(canon_result: str, var_mapping: Dict[str, str]) -> str:
    """
    Reverse alpha-renaming to get result in original variable names.

    Args:
        canon_result: Result in canonical variable names
        var_mapping: Mapping from original to canonical

    Returns:
        Result in original variable names
    """
    # Invert mapping
    inv_mapping = {v: k for k, v in var_mapping.items()}

    def repl(m):
        canon_var = m.group(0)
        return inv_mapping.get(canon_var, canon_var)

    # Replace canonical vars with original vars
    result = re.sub(r'\bx\d+\b', repl, canon_result)

    return result


# Reserved keywords that should not be renamed
RESERVED_KEYWORDS = {
    # Boolean operators
    'and', 'or', 'not', 'xor', 'implies', 'iff',
    'true', 'false',

    # Comparison operators (some solvers use these as keywords)
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',

    # Quantifiers
    'forall', 'exists',

    # Built-in functions
    'abs', 'min', 'max', 'mod', 'div',

    # Python keywords (in case formulas use Python-like syntax)
    'if', 'else', 'elif', 'while', 'for', 'in',
    'is', 'None', 'True', 'False',

    # Common mathematical functions
    'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'ln',
    'floor', 'ceil', 'round',

    # SMT-LIB keywords
    'assert', 'check-sat', 'declare-const', 'declare-fun',
    'define-fun', 'get-model', 'get-value',

    # Case-insensitive variants
    'AND', 'OR', 'NOT', 'XOR', 'IMPLIES', 'IFF',
    'TRUE', 'FALSE', 'FORALL', 'EXISTS'
}


def similarity_score(formula1: str, formula2: str) -> float:
    """
    Compute similarity between two formulas after normalization.

    Returns value in [0, 1] where 1.0 means identical after normalization.

    Args:
        formula1: First formula
        formula2: Second formula

    Returns:
        Similarity score
    """
    canon1, _ = alpha_rename(formula1)
    canon2, _ = alpha_rename(formula2)

    canon1 = canonicalize_whitespace(canon1)
    canon2 = canonicalize_whitespace(canon2)

    if canon1 == canon2:
        return 1.0

    # Compute character-level similarity
    # (simple Jaccard similarity on character n-grams)
    n = 3
    ngrams1 = {canon1[i:i+n] for i in range(len(canon1)-n+1)}
    ngrams2 = {canon2[i:i+n] for i in range(len(canon2)-n+1)}

    if not ngrams1 and not ngrams2:
        return 1.0 if canon1 == canon2 else 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0
