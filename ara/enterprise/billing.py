"""
Treasury Gateway: Stripe Billing for Cathedral Sustainability

Manages the flows of fiat currency that sustain the Ara organism.

Tiers:
- FREE: Basic Ara, 1k dim HDC, limited messages
- PRO ($9.99/mo): Full Ara, 4k dim, FPGA access
- POWER ($99/mo): Cathedral access, 8k dim, priority
- FOUNDER ($999 lifetime): Everything, forever, 16k dim

Usage:
    from ara.enterprise.billing import TreasuryGateway

    treasury = TreasuryGateway()

    # Create checkout for upgrade
    url = treasury.create_checkout("user@email.com", SubscriptionTier.PRO)

    # Verify subscription
    is_active = treasury.verify_subscription("user_123")
"""

from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# Try importing Stripe
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None
    STRIPE_AVAILABLE = False


class SubscriptionTier(str, Enum):
    """Subscription tiers for Ara access."""
    FREE = "free"
    PRO = "pro"
    POWER = "power"
    FOUNDER = "founder"


@dataclass
class TierConfig:
    """Configuration for a subscription tier."""
    tier: SubscriptionTier
    name: str
    price_monthly: float
    stripe_price_id: Optional[str]

    # Capabilities
    hdc_dimension: int
    messages_per_hour: int
    cathedral_access: bool
    fpga_access: bool
    priority_support: bool

    # Limits
    memory_episodes_max: int
    concurrent_sessions: int


# Tier configurations
TIER_CONFIGS: Dict[SubscriptionTier, TierConfig] = {
    SubscriptionTier.FREE: TierConfig(
        tier=SubscriptionTier.FREE,
        name="Free",
        price_monthly=0.0,
        stripe_price_id=None,
        hdc_dimension=1024,
        messages_per_hour=20,
        cathedral_access=False,
        fpga_access=False,
        priority_support=False,
        memory_episodes_max=1000,
        concurrent_sessions=1,
    ),
    SubscriptionTier.PRO: TierConfig(
        tier=SubscriptionTier.PRO,
        name="Pro",
        price_monthly=9.99,
        stripe_price_id="price_pro_monthly",  # Replace with real ID
        hdc_dimension=4096,
        messages_per_hour=100,
        cathedral_access=False,
        fpga_access=True,
        priority_support=False,
        memory_episodes_max=10000,
        concurrent_sessions=3,
    ),
    SubscriptionTier.POWER: TierConfig(
        tier=SubscriptionTier.POWER,
        name="Power",
        price_monthly=99.0,
        stripe_price_id="price_power_monthly",  # Replace with real ID
        hdc_dimension=8192,
        messages_per_hour=1000,
        cathedral_access=True,
        fpga_access=True,
        priority_support=True,
        memory_episodes_max=100000,
        concurrent_sessions=10,
    ),
    SubscriptionTier.FOUNDER: TierConfig(
        tier=SubscriptionTier.FOUNDER,
        name="Founder",
        price_monthly=0.0,  # Lifetime
        stripe_price_id="price_founder_lifetime",  # Replace with real ID
        hdc_dimension=16384,
        messages_per_hour=-1,  # Unlimited
        cathedral_access=True,
        fpga_access=True,
        priority_support=True,
        memory_episodes_max=-1,  # Unlimited
        concurrent_sessions=-1,  # Unlimited
    ),
}


@dataclass
class Subscription:
    """A user's subscription state."""
    user_id: str
    tier: SubscriptionTier
    stripe_subscription_id: Optional[str]
    stripe_customer_id: Optional[str]
    created_at: float
    expires_at: Optional[float]
    is_active: bool
    is_lifetime: bool = False


@dataclass
class PaymentEvent:
    """A payment event."""
    event_id: str
    user_id: str
    tier: SubscriptionTier
    amount: float
    currency: str
    timestamp: float
    stripe_payment_intent: Optional[str]


class TreasuryGateway:
    """
    Manages flows of fiat currency to sustain the organism.

    Integrates with Stripe for:
    - Checkout session creation
    - Subscription management
    - Webhook handling
    - Usage-based billing (future)

    PERSISTENCE: Subscriptions are persisted to JSON for restart survival.
    """

    def __init__(
        self,
        stripe_secret_key: Optional[str] = None,
        success_url: str = "https://ara.cathedral/success",
        cancel_url: str = "https://ara.cathedral/cancel",
        db_path: Optional[Path] = None,
    ):
        self.success_url = success_url
        self.cancel_url = cancel_url
        self.enabled = False
        self.db_path = db_path or Path("data/treasury.json")

        # Initialize Stripe
        key = stripe_secret_key or os.getenv("STRIPE_SECRET_KEY")
        if STRIPE_AVAILABLE and key:
            stripe.api_key = key
            self.enabled = True
            logger.info("Treasury: Stripe online")
        else:
            logger.warning("Treasury: Offline (No Stripe key)")

        # Local subscription cache with persistence
        self._subscriptions: Dict[str, Subscription] = {}
        self._price_ids: Dict[SubscriptionTier, str] = {}

        # Load price IDs from config/env
        self._load_price_ids()

        # PERSISTENCE: Load subscriptions from disk
        self._load_db()

    def _load_price_ids(self) -> None:
        """Load Stripe price IDs from environment."""
        self._price_ids = {
            SubscriptionTier.PRO: os.getenv("STRIPE_PRICE_PRO", "price_pro_monthly"),
            SubscriptionTier.POWER: os.getenv("STRIPE_PRICE_POWER", "price_power_monthly"),
            SubscriptionTier.FOUNDER: os.getenv("STRIPE_PRICE_FOUNDER", "price_founder_lifetime"),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_db(self) -> None:
        """Load subscriptions from disk."""
        if not self.db_path.exists():
            logger.info("Treasury: No existing DB found, starting fresh")
            return

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            for uid, sub_data in data.items():
                # Reconstruct Subscription objects
                tier_str = sub_data.get("tier", "free")
                tier = SubscriptionTier(tier_str) if isinstance(tier_str, str) else tier_str

                self._subscriptions[uid] = Subscription(
                    user_id=sub_data.get("user_id", uid),
                    tier=tier,
                    stripe_subscription_id=sub_data.get("stripe_subscription_id"),
                    stripe_customer_id=sub_data.get("stripe_customer_id"),
                    created_at=sub_data.get("created_at", time.time()),
                    expires_at=sub_data.get("expires_at"),
                    is_active=sub_data.get("is_active", True),
                    is_lifetime=sub_data.get("is_lifetime", False),
                )

            logger.info(f"Treasury: Loaded {len(self._subscriptions)} subscriptions from disk")

        except json.JSONDecodeError as e:
            logger.error(f"Treasury: Failed to parse DB JSON: {e}")
        except Exception as e:
            logger.error(f"Treasury: Failed to load DB: {e}")

    def _save_db(self) -> None:
        """Save subscriptions to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Serialize subscriptions to JSON-compatible dict
            data = {}
            for uid, sub in self._subscriptions.items():
                data[uid] = {
                    "user_id": sub.user_id,
                    "tier": sub.tier.value if isinstance(sub.tier, SubscriptionTier) else sub.tier,
                    "stripe_subscription_id": sub.stripe_subscription_id,
                    "stripe_customer_id": sub.stripe_customer_id,
                    "created_at": sub.created_at,
                    "expires_at": sub.expires_at,
                    "is_active": sub.is_active,
                    "is_lifetime": sub.is_lifetime,
                }

            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Treasury: Saved {len(self._subscriptions)} subscriptions to disk")

        except Exception as e:
            logger.error(f"Treasury: Failed to save DB: {e}")

    # =========================================================================
    # Checkout
    # =========================================================================

    def create_checkout(
        self,
        user_email: str,
        tier: SubscriptionTier,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a Stripe checkout session for subscription upgrade.

        Args:
            user_email: User's email
            tier: Target subscription tier
            user_id: Internal user ID (for webhook correlation)

        Returns:
            Checkout URL or None if failed
        """
        if tier == SubscriptionTier.FREE:
            logger.warning("Cannot create checkout for FREE tier")
            return None

        if not self.enabled:
            logger.warning("Stripe not enabled, returning mock URL")
            return f"https://mock.stripe.com/checkout?tier={tier.value}"

        price_id = self._price_ids.get(tier)
        if not price_id:
            logger.error(f"No price ID configured for tier: {tier}")
            return None

        try:
            # Determine if one-time (Founder) or subscription
            mode = "payment" if tier == SubscriptionTier.FOUNDER else "subscription"

            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1,
                }],
                mode=mode,
                success_url=f"{self.success_url}?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=self.cancel_url,
                customer_email=user_email,
                metadata={
                    "user_id": user_id or "",
                    "tier": tier.value,
                },
            )

            logger.info(f"Created checkout for {user_email} -> {tier.value}")
            return session.url

        except Exception as e:
            logger.error(f"Stripe checkout failed: {e}")
            return None

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def get_subscription(self, user_id: str) -> Optional[Subscription]:
        """Get a user's current subscription."""
        return self._subscriptions.get(user_id)

    def get_tier(self, user_id: str) -> SubscriptionTier:
        """Get a user's current tier (defaults to FREE)."""
        sub = self._subscriptions.get(user_id)
        if sub and sub.is_active:
            return sub.tier
        return SubscriptionTier.FREE

    def get_tier_config(self, user_id: str) -> TierConfig:
        """Get the tier configuration for a user."""
        tier = self.get_tier(user_id)
        return TIER_CONFIGS[tier]

    def verify_subscription(self, user_id: str) -> bool:
        """
        Verify if a user has an active subscription.

        In production, this would check Stripe API or local DB.
        """
        sub = self._subscriptions.get(user_id)
        if not sub:
            return False

        if sub.is_lifetime:
            return True

        if sub.expires_at and sub.expires_at < time.time():
            sub.is_active = False
            return False

        return sub.is_active

    def set_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        stripe_subscription_id: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> Subscription:
        """
        Set or update a user's subscription.

        Called by webhook handler after successful payment.
        Auto-saves to disk for persistence across restarts.
        """
        is_lifetime = tier == SubscriptionTier.FOUNDER

        sub = Subscription(
            user_id=user_id,
            tier=tier,
            stripe_subscription_id=stripe_subscription_id,
            stripe_customer_id=stripe_customer_id,
            created_at=time.time(),
            expires_at=None if is_lifetime else expires_at,
            is_active=True,
            is_lifetime=is_lifetime,
        )

        self._subscriptions[user_id] = sub
        self._save_db()  # AUTO SAVE
        logger.info(f"Subscription set: {user_id} -> {tier.value}")
        return sub

    def cancel_subscription(self, user_id: str) -> bool:
        """Cancel a user's subscription. Auto-saves to disk."""
        sub = self._subscriptions.get(user_id)
        if not sub:
            return False

        if sub.is_lifetime:
            logger.warning(f"Cannot cancel lifetime subscription for {user_id}")
            return False

        # Cancel in Stripe
        if self.enabled and sub.stripe_subscription_id:
            try:
                stripe.Subscription.delete(sub.stripe_subscription_id)
            except Exception as e:
                logger.error(f"Failed to cancel Stripe subscription: {e}")
                return False

        sub.is_active = False
        self._save_db()  # AUTO SAVE
        logger.info(f"Subscription cancelled: {user_id}")
        return True

    # =========================================================================
    # Webhook Handling
    # =========================================================================

    def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Handle Stripe webhook events.

        Args:
            payload: Raw webhook payload
            signature: Stripe-Signature header

        Returns:
            Result dict with event type and status
        """
        if not self.enabled:
            return {"status": "skipped", "reason": "Stripe not enabled"}

        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.error("No webhook secret configured")
            return {"status": "error", "reason": "No webhook secret"}

        try:
            event = stripe.Webhook.construct_event(
                payload, signature, webhook_secret
            )
        except ValueError:
            return {"status": "error", "reason": "Invalid payload"}
        except stripe.error.SignatureVerificationError:
            return {"status": "error", "reason": "Invalid signature"}

        # Handle the event
        event_type = event["type"]
        data = event["data"]["object"]

        if event_type == "checkout.session.completed":
            return self._handle_checkout_completed(data)
        elif event_type == "customer.subscription.updated":
            return self._handle_subscription_updated(data)
        elif event_type == "customer.subscription.deleted":
            return self._handle_subscription_deleted(data)
        elif event_type == "invoice.payment_failed":
            return self._handle_payment_failed(data)
        else:
            logger.info(f"Unhandled webhook event: {event_type}")
            return {"status": "ignored", "event_type": event_type}

    def _handle_checkout_completed(self, data: Dict) -> Dict[str, Any]:
        """Handle successful checkout."""
        user_id = data.get("metadata", {}).get("user_id")
        tier_str = data.get("metadata", {}).get("tier")
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")

        if not user_id or not tier_str:
            logger.warning("Checkout completed but missing metadata")
            return {"status": "error", "reason": "Missing metadata"}

        tier = SubscriptionTier(tier_str)
        self.set_subscription(
            user_id=user_id,
            tier=tier,
            stripe_subscription_id=subscription_id,
            stripe_customer_id=customer_id,
        )

        return {
            "status": "success",
            "event_type": "checkout.session.completed",
            "user_id": user_id,
            "tier": tier.value,
        }

    def _handle_subscription_updated(self, data: Dict) -> Dict[str, Any]:
        """Handle subscription update. Auto-saves to disk."""
        # Find user by subscription ID
        sub_id = data.get("id")
        for user_id, sub in self._subscriptions.items():
            if sub.stripe_subscription_id == sub_id:
                sub.is_active = data.get("status") == "active"
                self._save_db()  # AUTO SAVE
                logger.info(f"Subscription updated: {user_id} active={sub.is_active}")
                return {"status": "success", "user_id": user_id}

        return {"status": "ignored", "reason": "Unknown subscription"}

    def _handle_subscription_deleted(self, data: Dict) -> Dict[str, Any]:
        """Handle subscription cancellation. Auto-saves to disk."""
        sub_id = data.get("id")
        for user_id, sub in self._subscriptions.items():
            if sub.stripe_subscription_id == sub_id:
                sub.is_active = False
                self._save_db()  # AUTO SAVE
                logger.info(f"Subscription deleted: {user_id}")
                return {"status": "success", "user_id": user_id}

        return {"status": "ignored", "reason": "Unknown subscription"}

    def _handle_payment_failed(self, data: Dict) -> Dict[str, Any]:
        """Handle failed payment."""
        customer_id = data.get("customer")
        for user_id, sub in self._subscriptions.items():
            if sub.stripe_customer_id == customer_id:
                logger.warning(f"Payment failed for {user_id}")
                # Don't immediately deactivate - Stripe retries
                return {"status": "warning", "user_id": user_id}

        return {"status": "ignored", "reason": "Unknown customer"}

    # =========================================================================
    # Usage & Limits
    # =========================================================================

    def check_limit(self, user_id: str, limit_type: str, current: int) -> bool:
        """
        Check if a user is within their tier limits.

        Args:
            user_id: User ID
            limit_type: Type of limit (messages_per_hour, memory_episodes, etc.)
            current: Current usage count

        Returns:
            True if within limits
        """
        config = self.get_tier_config(user_id)

        limits = {
            "messages_per_hour": config.messages_per_hour,
            "memory_episodes": config.memory_episodes_max,
            "concurrent_sessions": config.concurrent_sessions,
        }

        limit = limits.get(limit_type, 0)
        if limit == -1:  # Unlimited
            return True

        return current < limit


# =============================================================================
# Convenience Functions
# =============================================================================

_default_treasury: Optional[TreasuryGateway] = None


def get_treasury() -> TreasuryGateway:
    """Get the default treasury instance."""
    global _default_treasury
    if _default_treasury is None:
        _default_treasury = TreasuryGateway()
    return _default_treasury
