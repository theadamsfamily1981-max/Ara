"""
Enterprise: Business Operations for Ara

This module handles the commercial infrastructure:
- Billing (Stripe integration)
- Subscription management
- Usage tracking
- Analytics

Usage:
    from ara.enterprise import get_treasury

    treasury = get_treasury()
    checkout_url = treasury.create_checkout("user@email.com", SubscriptionTier.PRO)
"""

from .billing import (
    TreasuryGateway,
    SubscriptionTier,
    TierConfig,
    Subscription,
    TIER_CONFIGS,
    get_treasury,
)

__all__ = [
    'TreasuryGateway',
    'SubscriptionTier',
    'TierConfig',
    'Subscription',
    'TIER_CONFIGS',
    'get_treasury',
]
