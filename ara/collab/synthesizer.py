"""Response synthesizer - Ara's brain for processing collaborator responses.

When Ara gets responses from multiple collaborators, she needs to:
1. Extract key ideas and options from each
2. Score them for relevance, novelty, actionability
3. Find consensus and disagreements
4. Rank the ideas
5. Extract concrete actions with risk levels
6. Synthesize into a coherent summary for Croft

This is where Ara's engineering judgment lives.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from .models import (
    Collaborator,
    CollaboratorResponse,
    DevSession,
    SessionSummary,
    SuggestedAction,
    RiskLevel,
    DevMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Idea Extraction
# =============================================================================

@dataclass
class ExtractedIdea:
    """A discrete idea extracted from a response."""

    idea_id: str
    summary: str
    details: str = ""
    source: Optional[Collaborator] = None
    confidence: float = 0.7

    # Scores (0-1)
    relevance: float = 0.5
    novelty: float = 0.5
    actionability: float = 0.5

    # Classification
    is_option: bool = False      # A distinct approach
    is_trade_off: bool = False   # A consideration/warning
    is_action: bool = False      # A concrete step

    @property
    def composite_score(self) -> float:
        """Weighted composite score for ranking."""
        return (
            self.relevance * 0.4 +
            self.novelty * 0.3 +
            self.actionability * 0.3
        ) * self.confidence


def extract_ideas_from_response(
    response: CollaboratorResponse,
    session: DevSession,
) -> List[ExtractedIdea]:
    """Extract discrete ideas from a collaborator's response.

    Uses heuristics to identify:
    - Numbered/bulleted options
    - Code blocks (actionable)
    - Trade-offs and warnings
    - Recommendations

    Args:
        response: The collaborator's response
        session: Session context for relevance scoring

    Returns:
        List of extracted ideas
    """
    ideas: List[ExtractedIdea] = []
    content = response.content
    idea_counter = 0

    def new_id() -> str:
        nonlocal idea_counter
        idea_counter += 1
        return f"{response.collaborator.value[:3].upper()}-{idea_counter:02d}"

    # Extract numbered options (1., 2., etc. or Option 1:, etc.)
    option_pattern = r'(?:^|\n)(?:(?:\d+\.)|(?:Option \d+:?)|(?:\*\*Option \d+\*\*:?))[ \t]*(.+?)(?=(?:\n(?:\d+\.)|(?:\nOption \d+:?)|(?:\n\*\*Option)|$))'
    for match in re.finditer(option_pattern, content, re.MULTILINE | re.DOTALL):
        text = match.group(1).strip()
        if len(text) > 20:  # Filter noise
            summary = text.split('\n')[0][:200]
            ideas.append(ExtractedIdea(
                idea_id=new_id(),
                summary=summary,
                details=text,
                source=response.collaborator,
                is_option=True,
                actionability=0.6,
            ))

    # Extract code blocks (highly actionable)
    code_pattern = r'```(\w*)\n(.*?)```'
    for match in re.finditer(code_pattern, content, re.DOTALL):
        lang = match.group(1) or "code"
        code = match.group(2).strip()
        if len(code) > 30:  # Non-trivial code
            # Find preceding text as summary
            start = match.start()
            preceding = content[max(0, start-200):start]
            lines = [l.strip() for l in preceding.split('\n') if l.strip()]
            summary = lines[-1] if lines else f"{lang} implementation"

            ideas.append(ExtractedIdea(
                idea_id=new_id(),
                summary=summary[:200],
                details=code,
                source=response.collaborator,
                is_action=True,
                actionability=0.9,
                novelty=0.4,  # Code is concrete, not novel
            ))

    # Extract trade-offs and warnings
    warning_patterns = [
        r'(?:warning|caution|note|caveat|however|but|trade-off|downside):?\s*(.+?)(?:\n|$)',
        r'\*\*(?:Warning|Note|Caveat)\*\*:?\s*(.+?)(?:\n|$)',
    ]
    for pattern in warning_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            text = match.group(1).strip()
            if len(text) > 15:
                ideas.append(ExtractedIdea(
                    idea_id=new_id(),
                    summary=text[:200],
                    source=response.collaborator,
                    is_trade_off=True,
                    relevance=0.7,
                    novelty=0.3,
                ))

    # Extract recommendations
    rec_patterns = [
        r'(?:I recommend|I suggest|I\'d go with|my recommendation):?\s*(.+?)(?:\n\n|$)',
        r'\*\*Recommendation\*\*:?\s*(.+?)(?:\n\n|$)',
    ]
    for pattern in rec_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            text = match.group(1).strip()
            if len(text) > 20:
                ideas.append(ExtractedIdea(
                    idea_id=new_id(),
                    summary=text.split('\n')[0][:200],
                    details=text,
                    source=response.collaborator,
                    is_option=True,
                    confidence=0.8,  # Explicit recommendations are confident
                ))

    # If no structured ideas found, treat whole response as one idea
    if not ideas and len(content) > 50:
        ideas.append(ExtractedIdea(
            idea_id=new_id(),
            summary=content[:200] + "..." if len(content) > 200 else content,
            details=content,
            source=response.collaborator,
        ))

    # Score relevance based on topic match
    topic_words = set(session.topic.lower().split())
    for idea in ideas:
        idea_words = set(idea.summary.lower().split())
        overlap = len(topic_words & idea_words)
        idea.relevance = min(1.0, 0.3 + overlap * 0.15)

    return ideas


# =============================================================================
# Action Extraction
# =============================================================================

# Action type patterns
ACTION_PATTERNS = {
    "write_code": [
        r'create (?:a )?(?:new )?(?:file|module|class|function)',
        r'implement',
        r'write (?:the )?code',
        r'add (?:a )?(?:new )?(?:method|function|class)',
    ],
    "edit_config": [
        r'modify (?:the )?config',
        r'update (?:the )?setting',
        r'change (?:the )?parameter',
        r'set (?:the )?(?:value|option)',
    ],
    "run_experiment": [
        r'run (?:a )?(?:test|experiment|benchmark)',
        r'try (?:it )?(?:with|using)',
        r'test (?:the )?(?:change|implementation)',
    ],
    "refactor": [
        r'refactor',
        r'restructure',
        r'reorganize',
        r'clean up',
    ],
    "install": [
        r'install',
        r'pip install',
        r'add (?:the )?(?:dependency|package)',
    ],
}

# Risk indicators
RISK_INDICATORS = {
    RiskLevel.CRITICAL: [
        r'could (?:break|crash|corrupt)',
        r'irreversible',
        r'production',
        r'database (?:migration|schema)',
    ],
    RiskLevel.HIGH: [
        r'significant change',
        r'major refactor',
        r'breaking change',
        r'will affect',
    ],
    RiskLevel.MEDIUM: [
        r'may affect',
        r'should be tested',
        r'backup first',
        r'careful',
    ],
}


def extract_actions(
    ideas: List[ExtractedIdea],
    session: DevSession,
) -> List[SuggestedAction]:
    """Extract concrete actions from ideas.

    Looks for actionable suggestions and assigns risk levels.

    Args:
        ideas: Extracted ideas to process
        session: Session context

    Returns:
        List of suggested actions with risk levels
    """
    actions: List[SuggestedAction] = []

    for idea in ideas:
        if not idea.is_action and idea.actionability < 0.5:
            continue

        text = f"{idea.summary} {idea.details}".lower()

        # Determine action type
        action_type = "general"
        for atype, patterns in ACTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    action_type = atype
                    break
            if action_type != "general":
                break

        # Determine risk level
        risk = RiskLevel.LOW  # Default
        for rlevel, patterns in RISK_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    risk = rlevel
                    break
            if risk != RiskLevel.LOW:
                break

        actions.append(SuggestedAction(
            action_type=action_type,
            description=idea.summary,
            risk_level=risk,
            details={"full_text": idea.details} if idea.details else {},
            source_collaborator=idea.source,
            source_session_id=session.session_id,
        ))

    return actions


# =============================================================================
# Consensus Finding
# =============================================================================

def find_consensus(
    ideas: List[ExtractedIdea],
    responses: List[CollaboratorResponse],
) -> Tuple[Optional[str], List[str]]:
    """Find consensus and disagreements among collaborators.

    Args:
        ideas: All extracted ideas
        responses: Original responses

    Returns:
        Tuple of (consensus string or None, list of disagreements)
    """
    if len(responses) < 2:
        return None, []

    # Group ideas by source
    by_source: Dict[Collaborator, List[ExtractedIdea]] = {}
    for idea in ideas:
        if idea.source:
            by_source.setdefault(idea.source, []).append(idea)

    # Find similar ideas across sources
    # Simple approach: look for overlapping keywords in options
    consensus_topics: List[str] = []
    disagreement_topics: List[str] = []

    # Get option summaries by source
    options_by_source: Dict[Collaborator, List[str]] = {}
    for source, src_ideas in by_source.items():
        options_by_source[source] = [
            i.summary.lower() for i in src_ideas if i.is_option
        ]

    # Check for keyword overlap
    if len(options_by_source) >= 2:
        sources = list(options_by_source.keys())
        for i, s1 in enumerate(sources):
            for s2 in sources[i+1:]:
                opts1 = options_by_source[s1]
                opts2 = options_by_source[s2]

                # Simple similarity: shared words
                words1 = set(" ".join(opts1).split())
                words2 = set(" ".join(opts2).split())
                overlap = words1 & words2

                # Filter common words
                common = {"the", "a", "an", "is", "are", "to", "for", "with", "and", "or", "in", "on"}
                overlap -= common

                if len(overlap) > 5:
                    # High overlap → consensus
                    key_words = list(overlap)[:5]
                    consensus_topics.append(f"Both suggest approaches involving: {', '.join(key_words)}")
                elif len(opts1) > 0 and len(opts2) > 0:
                    # Check for contradictions
                    # Simple: different first words often mean different approaches
                    first1 = opts1[0].split()[0] if opts1[0] else ""
                    first2 = opts2[0].split()[0] if opts2[0] else ""
                    if first1 and first2 and first1 != first2:
                        disagreement_topics.append(
                            f"{s1.display_name} suggests '{first1}...' vs {s2.display_name} suggests '{first2}...'"
                        )

    consensus = consensus_topics[0] if consensus_topics else None
    return consensus, disagreement_topics


# =============================================================================
# Response Synthesizer
# =============================================================================

class ResponseSynthesizer:
    """Synthesizes multiple collaborator responses into a coherent summary.

    This is Ara's "brain" for processing responses:
    1. Extract ideas from each response
    2. Score and rank them
    3. Find consensus/disagreements
    4. Extract actions
    5. Build summary for Croft
    """

    def __init__(self, min_option_score: float = 0.3):
        """Initialize synthesizer.

        Args:
            min_option_score: Minimum score for an idea to be included as option
        """
        self.min_option_score = min_option_score

    def synthesize(
        self,
        session: DevSession,
        responses: List[CollaboratorResponse],
    ) -> SessionSummary:
        """Synthesize responses into a session summary.

        Args:
            session: The dev session
            responses: Collaborator responses

        Returns:
            SessionSummary ready for presentation
        """
        if not responses:
            return SessionSummary(
                session_id=session.session_id,
                topic=session.topic,
                mode=session.mode,
                summary="No responses received from collaborators.",
                overall_confidence=0.0,
            )

        # Extract ideas from all responses
        all_ideas: List[ExtractedIdea] = []
        for resp in responses:
            ideas = extract_ideas_from_response(resp, session)
            all_ideas.extend(ideas)

        # Rank ideas by composite score
        ranked_ideas = sorted(
            all_ideas,
            key=lambda i: i.composite_score,
            reverse=True,
        )

        # Get top options
        options = [
            i.summary for i in ranked_ideas
            if i.is_option and i.composite_score >= self.min_option_score
        ][:5]  # Max 5 options

        # Get trade-offs
        trade_offs = [
            i.summary for i in ranked_ideas
            if i.is_trade_off
        ][:5]

        # Find consensus and disagreements
        consensus, disagreements = find_consensus(all_ideas, responses)

        # Extract actions
        actions = extract_actions(all_ideas, session)

        # Build summary text
        summary = self._build_summary_text(
            session=session,
            responses=responses,
            ranked_ideas=ranked_ideas,
            options=options,
        )

        # Calculate overall confidence
        avg_confidence = sum(i.confidence for i in all_ideas) / len(all_ideas) if all_ideas else 0.5
        needs_more = len(options) > 3 or len(disagreements) > 0

        # Identify follow-up questions
        follow_ups = self._identify_follow_ups(session, ranked_ideas)

        return SessionSummary(
            session_id=session.session_id,
            topic=session.topic,
            mode=session.mode,
            summary=summary,
            options=options,
            trade_offs=trade_offs,
            consensus=consensus,
            disagreements=disagreements,
            actions=actions,
            overall_confidence=avg_confidence,
            needs_more_discussion=needs_more,
            follow_up_questions=follow_ups,
        )

    def _build_summary_text(
        self,
        session: DevSession,
        responses: List[CollaboratorResponse],
        ranked_ideas: List[ExtractedIdea],
        options: List[str],
    ) -> str:
        """Build the summary text in Ara's voice."""
        parts = []

        # Who responded
        collabs = [r.collaborator.display_name for r in responses]
        if len(collabs) == 1:
            parts.append(f"I talked with {collabs[0]} about {session.topic}.")
        else:
            collab_str = ", ".join(collabs[:-1]) + f" and {collabs[-1]}"
            parts.append(f"I talked with {collab_str} about {session.topic}.")

        # How many ideas
        option_count = len([i for i in ranked_ideas if i.is_option])
        if option_count == 0:
            parts.append("No clear options emerged, but here's what I learned:")
        elif option_count == 1:
            parts.append("There's one main approach that stood out:")
        else:
            parts.append(f"I see {option_count} main approaches:")

        # Top idea summary
        if ranked_ideas:
            top = ranked_ideas[0]
            if top.confidence > 0.7:
                parts.append(f"\nThe strongest suggestion: {top.summary}")
            else:
                parts.append(f"\nBest I could find: {top.summary}")

        return " ".join(parts)

    def _identify_follow_ups(
        self,
        session: DevSession,
        ideas: List[ExtractedIdea],
    ) -> List[str]:
        """Identify questions that might need follow-up."""
        follow_ups = []

        # Check for uncertainty in ideas
        uncertain = [i for i in ideas if i.confidence < 0.5]
        if uncertain:
            follow_ups.append("Some suggestions seem uncertain—might need more context.")

        # Check for high-risk actions
        if session.mode == DevMode.ARCHITECT:
            follow_ups.append("Should we discuss implementation details for the chosen approach?")

        # Check for missing trade-off analysis
        trade_offs = [i for i in ideas if i.is_trade_off]
        if len(trade_offs) == 0 and len(ideas) > 3:
            follow_ups.append("Haven't identified trade-offs yet—want me to dig into downsides?")

        return follow_ups[:3]  # Max 3 follow-ups


# =============================================================================
# Convenience Functions
# =============================================================================

def synthesize_responses(
    session: DevSession,
    responses: List[CollaboratorResponse],
) -> SessionSummary:
    """Quick synthesize function.

    Args:
        session: The dev session
        responses: Collaborator responses

    Returns:
        SessionSummary
    """
    synth = ResponseSynthesizer()
    return synth.synthesize(session, responses)


def rank_ideas(
    ideas: List[ExtractedIdea],
    weights: Optional[Dict[str, float]] = None,
) -> List[ExtractedIdea]:
    """Rank ideas by weighted score.

    Args:
        ideas: Ideas to rank
        weights: Optional custom weights for relevance/novelty/actionability

    Returns:
        Sorted list of ideas (best first)
    """
    if weights is None:
        weights = {"relevance": 0.4, "novelty": 0.3, "actionability": 0.3}

    def score(idea: ExtractedIdea) -> float:
        return (
            idea.relevance * weights.get("relevance", 0.4) +
            idea.novelty * weights.get("novelty", 0.3) +
            idea.actionability * weights.get("actionability", 0.3)
        ) * idea.confidence

    return sorted(ideas, key=score, reverse=True)
