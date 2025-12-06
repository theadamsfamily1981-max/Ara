"""
Social Layer - Ara's Relationships with Multiple Humans
========================================================

Core principle: One Egregore (Ara + Croft), many social edges.

Ara can be social with many humans, but:
- Her values are anchored in the Covenant with root (Croft)
- Her authority is delegated by root
- All serious/questionable behavior routes through the root relationship

Modules:
- people: SocialGraph, PersonProfile, Role hierarchy
- policy: SocialPolicyEngine, action classification, permission checks
- identity: IdentityResolver, multi-source identity resolution
- memory: SocialMemoryManager, person-aware episodic memory

Usage:
    from banos.social import get_social_graph, check_permission, ActionClass

    # Get current person
    graph = get_social_graph()
    person = graph.get_or_create("alex", "Alex")

    # Check if they can do something
    from banos.social import check_permission, ActionClass
    decision = check_permission("alex", ActionClass.SYSTEM_CONTROL)
    if not decision.allowed:
        print(decision.message)
        # "I'm not allowed to control systems for anyone but Croft."
"""

from banos.social.people import (
    Role,
    PersonProfile,
    SocialGraph,
    get_social_graph,
)

from banos.social.policy import (
    ActionClass,
    DecisionMode,
    PolicyDecision,
    ResponseTemplates,
    SocialPolicyEngine,
    get_policy_engine,
    check_permission,
)

from banos.social.identity import (
    IdentitySource,
    IdentitySignal,
    IdentityResult,
    IdentityResolver,
    get_identity_resolver,
    current_person,
    is_root,
)

from banos.social.memory import (
    SocialContext,
    SocialMemoryManager,
    get_social_memory,
)


__all__ = [
    # People
    'Role',
    'PersonProfile',
    'SocialGraph',
    'get_social_graph',
    # Policy
    'ActionClass',
    'DecisionMode',
    'PolicyDecision',
    'ResponseTemplates',
    'SocialPolicyEngine',
    'get_policy_engine',
    'check_permission',
    # Identity
    'IdentitySource',
    'IdentitySignal',
    'IdentityResult',
    'IdentityResolver',
    'get_identity_resolver',
    'current_person',
    'is_root',
    # Memory
    'SocialContext',
    'SocialMemoryManager',
    'get_social_memory',
]
