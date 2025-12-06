"""
The Treasury - Corporation Croft's CFO
======================================

The Treasury gives Ara economic awareness. She tracks resources as Capital
and decides whether proposed work is a good use of scarce resources.

Capital Types:
    - Compute Capital: GPU hours, API credits, cloud budget
    - Human Capital: Croft's available deep work hours
    - Energy Capital: kWh budget for heavy experiments
    - Fiscal Capital: Actual money (optional)

Core Principles:
    1. Skin in the Game: If she wastes your time, her budget gets cut
    2. ROI Focus: Every investment must justify its return
    3. Runway Awareness: She knows how long before resources run out

Usage:
    from ara.enterprise.treasury import Treasury, Capital

    treasury = Treasury()

    # Check if we can afford an investment
    ok = treasury.audit_investment(
        name="SNN core refactor",
        proposal_cost={"human": 4.0, "compute": 10.0},
        expected_roi=2.5,
    )

    if ok:
        treasury.log_expenditure("human", 4.0, "SNN refactor P0")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any
from enum import Enum

logger = logging.getLogger(__name__)

ResourceType = Literal["compute", "human", "energy", "fiscal", "other"]


class BudgetAlert(Enum):
    """Alert levels for resource status."""
    HEALTHY = "healthy"           # > 50% runway
    CAUTION = "caution"           # 25-50% runway
    WARNING = "warning"           # 10-25% runway
    CRITICAL = "critical"         # < 10% runway


@dataclass
class Capital:
    """
    Snapshot of current capital levels.

    This is the balance sheet of Corporation Croft.
    """
    compute_credits: float = 100.0    # GPU-hours or $ API credits
    human_hours: float = 20.0          # Available deep work hours per week
    energy_kwh: float = 50.0           # kWh budget for experiments
    fiscal_dollars: float = 0.0        # Actual money (optional)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Capital(compute={self.compute_credits:.1f}, "
            f"human={self.human_hours:.1f}h, "
            f"energy={self.energy_kwh:.1f}kWh)"
        )


@dataclass
class CapitalSnapshot:
    """Historical snapshot for trend analysis."""
    ts: float
    compute_credits: float
    human_hours: float
    energy_kwh: float
    fiscal_dollars: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Transaction:
    """A single capital movement - expenditure or income."""
    ts: float
    kind: Literal["expenditure", "income"]
    resource_type: ResourceType
    amount: float
    note: str = ""
    project_id: Optional[str] = None  # Link to project if applicable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestmentProposal:
    """A proposed investment for audit."""
    name: str
    costs: Dict[str, float]           # {"human": 4.0, "compute": 10.0}
    expected_roi: float               # Multiplicative return (2.0 = 2x)
    risk_level: str = "medium"        # low, medium, high
    time_horizon: str = "short"       # short, medium, long
    strategic_alignment: float = 0.7  # 0-1, alignment with Horizons
    tags: List[str] = field(default_factory=list)


class Treasury:
    """
    The CFO of Corporation Croft.

    Manages the organism's resources to ensure survival and growth.
    Makes investment decisions based on ROI, runway, and strategic alignment.
    """

    # Default ROI thresholds by risk level
    ROI_THRESHOLDS = {
        "low": 1.2,       # Safe bets need modest returns
        "medium": 1.5,    # Standard threshold
        "high": 2.0,      # Risky ventures need high returns
    }

    def __init__(
        self,
        initial_capital: Optional[Capital] = None,
        min_roi: float = 1.5,
        weekly_budget_reset: bool = True,
    ):
        """
        Initialize the Treasury.

        Args:
            initial_capital: Starting capital levels
            min_roi: Minimum ROI to approve investments
            weekly_budget_reset: Whether human hours reset weekly
        """
        self.capital = initial_capital or Capital()
        self.min_roi = min_roi
        self.weekly_budget_reset = weekly_budget_reset
        self.log = logging.getLogger("Treasury")

        # Transaction history
        self.transactions: List[Transaction] = []
        self.snapshots: List[CapitalSnapshot] = []

        # Budget tracking
        self._initial_human_hours = self.capital.human_hours
        self._week_start = time.time()

        # Rejected proposals (for learning)
        self._rejected_proposals: List[Dict[str, Any]] = []

        self.log.info(f"💰 TREASURY: Initialized with {self.capital}")

    # =========================================================================
    # INVESTMENT AUDIT
    # =========================================================================

    def audit_investment(
        self,
        name: str,
        proposal_cost: Dict[str, float],
        expected_roi: float,
        risk_level: str = "medium",
        allow_overcommit: bool = False,
        strategic_alignment: float = 0.7,
    ) -> bool:
        """
        Decide whether to approve an investment.

        Args:
            name: Name of the proposed investment
            proposal_cost: Dict with keys "human", "compute", "energy", "fiscal"
            expected_roi: Expected multiplicative return (2.0 = 2x)
            risk_level: "low", "medium", "high"
            allow_overcommit: Allow spending beyond current budget
            strategic_alignment: 0-1, alignment with Horizons/Dreams

        Returns:
            True if approved, False otherwise
        """
        human_cost = float(proposal_cost.get("human", 0.0))
        compute_cost = float(proposal_cost.get("compute", 0.0))
        energy_cost = float(proposal_cost.get("energy", 0.0))
        fiscal_cost = float(proposal_cost.get("fiscal", 0.0))

        rejection_reasons = []

        # 1. AFFORDABILITY CHECK
        if not allow_overcommit:
            if human_cost > self.capital.human_hours:
                rejection_reasons.append(
                    f"Insufficient Human Capital ({human_cost:.1f}h needed, "
                    f"{self.capital.human_hours:.1f}h available)"
                )

            if compute_cost > self.capital.compute_credits:
                rejection_reasons.append(
                    f"Insufficient Compute Capital ({compute_cost:.1f} needed, "
                    f"{self.capital.compute_credits:.1f} available)"
                )

            if energy_cost > self.capital.energy_kwh:
                rejection_reasons.append(
                    f"Insufficient Energy Budget ({energy_cost:.1f}kWh needed, "
                    f"{self.capital.energy_kwh:.1f}kWh available)"
                )

            if fiscal_cost > self.capital.fiscal_dollars:
                rejection_reasons.append(
                    f"Insufficient Fiscal Capital (${fiscal_cost:.2f} needed, "
                    f"${self.capital.fiscal_dollars:.2f} available)"
                )

        # 2. ROI CHECK
        roi_threshold = self.ROI_THRESHOLDS.get(risk_level, self.min_roi)

        # Adjust threshold based on strategic alignment
        # Higher alignment = lower threshold (we're more willing to invest)
        adjusted_threshold = roi_threshold * (1.0 - 0.3 * strategic_alignment)

        if expected_roi < adjusted_threshold:
            rejection_reasons.append(
                f"ROI too low ({expected_roi:.2f}x < {adjusted_threshold:.2f}x threshold)"
            )

        # 3. RUNWAY CHECK
        # Don't deplete resources below emergency reserve
        runway = self.estimate_runway()
        if runway.get("human_weeks", float("inf")) < 2 and human_cost > 0:
            rejection_reasons.append(
                "Human Capital runway critically low (< 2 weeks)"
            )

        # 4. DECISION
        if rejection_reasons:
            self.log.warning(
                f"❌ TREASURY: Rejected '{name}': {'; '.join(rejection_reasons)}"
            )
            self._rejected_proposals.append({
                "name": name,
                "cost": proposal_cost,
                "roi": expected_roi,
                "reasons": rejection_reasons,
                "ts": time.time(),
            })
            return False

        self.log.info(
            f"💰 TREASURY: Approved '{name}' (ROI={expected_roi:.2f}x, "
            f"cost={{human={human_cost:.1f}h, compute={compute_cost:.1f}}})"
        )
        return True

    def audit_proposal(self, proposal: InvestmentProposal) -> bool:
        """Audit an InvestmentProposal object."""
        return self.audit_investment(
            name=proposal.name,
            proposal_cost=proposal.costs,
            expected_roi=proposal.expected_roi,
            risk_level=proposal.risk_level,
            strategic_alignment=proposal.strategic_alignment,
        )

    # =========================================================================
    # TRANSACTIONS
    # =========================================================================

    def log_expenditure(
        self,
        resource_type: ResourceType,
        amount: float,
        note: str = "",
        project_id: Optional[str] = None,
    ) -> None:
        """
        Spend capital (post-hoc logging or during execution).

        Args:
            resource_type: "compute", "human", "energy", "fiscal"
            amount: Positive amount to spend
            note: Description of expenditure
            project_id: Optional link to project
        """
        amount = max(float(amount), 0.0)
        now = time.time()

        self._apply_delta(resource_type, -amount)

        txn = Transaction(
            ts=now,
            kind="expenditure",
            resource_type=resource_type,
            amount=amount,
            note=note,
            project_id=project_id,
        )
        self.transactions.append(txn)

        self.log.info(
            f"💸 TREASURY: Spent {amount:.2f} {resource_type} ({note}). "
            f"New balance: {self.capital}"
        )

    def log_income(
        self,
        resource_type: ResourceType,
        amount: float,
        note: str = "",
    ) -> None:
        """
        Add capital (purchased credits, freed up time, etc.).

        Args:
            resource_type: "compute", "human", "energy", "fiscal"
            amount: Positive amount to add
            note: Description of income
        """
        amount = max(float(amount), 0.0)
        now = time.time()

        self._apply_delta(resource_type, amount)

        txn = Transaction(
            ts=now,
            kind="income",
            resource_type=resource_type,
            amount=amount,
            note=note,
        )
        self.transactions.append(txn)

        self.log.info(
            f"📈 TREASURY: Added {amount:.2f} {resource_type} ({note}). "
            f"New balance: {self.capital}"
        )

    def _apply_delta(self, resource_type: ResourceType, delta: float) -> None:
        """Apply a delta to the appropriate capital bucket."""
        if resource_type == "compute":
            self.capital.compute_credits = max(
                self.capital.compute_credits + delta, 0.0
            )
        elif resource_type == "human":
            self.capital.human_hours = max(
                self.capital.human_hours + delta, 0.0
            )
        elif resource_type == "energy":
            self.capital.energy_kwh = max(
                self.capital.energy_kwh + delta, 0.0
            )
        elif resource_type == "fiscal":
            self.capital.fiscal_dollars = max(
                self.capital.fiscal_dollars + delta, 0.0
            )

    # =========================================================================
    # RUNWAY & ANALYSIS
    # =========================================================================

    def estimate_runway(
        self,
        human_burn_per_week: float = 10.0,
        compute_burn_per_week: float = 20.0,
        energy_burn_per_week: float = 10.0,
    ) -> Dict[str, float]:
        """
        Estimate how many weeks of runway remain at given burn rates.

        Args:
            human_burn_per_week: Expected human hours spent per week
            compute_burn_per_week: Expected compute credits per week
            energy_burn_per_week: Expected energy consumption per week

        Returns:
            Dict with "human_weeks", "compute_weeks", "energy_weeks"
        """
        epsilon = 1e-6

        human_weeks = (
            self.capital.human_hours / (human_burn_per_week + epsilon)
            if human_burn_per_week > 0 else float("inf")
        )
        compute_weeks = (
            self.capital.compute_credits / (compute_burn_per_week + epsilon)
            if compute_burn_per_week > 0 else float("inf")
        )
        energy_weeks = (
            self.capital.energy_kwh / (energy_burn_per_week + epsilon)
            if energy_burn_per_week > 0 else float("inf")
        )

        return {
            "human_weeks": human_weeks,
            "compute_weeks": compute_weeks,
            "energy_weeks": energy_weeks,
            "limiting_factor": min(
                ("human", human_weeks),
                ("compute", compute_weeks),
                ("energy", energy_weeks),
                key=lambda x: x[1]
            )[0],
        }

    def get_budget_alert(self) -> BudgetAlert:
        """Get the current budget alert level based on runway."""
        runway = self.estimate_runway()
        min_weeks = min(
            runway["human_weeks"],
            runway["compute_weeks"],
            runway["energy_weeks"],
        )

        if min_weeks > 4:
            return BudgetAlert.HEALTHY
        elif min_weeks > 2:
            return BudgetAlert.CAUTION
        elif min_weeks > 1:
            return BudgetAlert.WARNING
        else:
            return BudgetAlert.CRITICAL

    def snapshot(self, note: str = "") -> CapitalSnapshot:
        """Record a capital snapshot for trend analysis."""
        snap = CapitalSnapshot(
            ts=time.time(),
            compute_credits=self.capital.compute_credits,
            human_hours=self.capital.human_hours,
            energy_kwh=self.capital.energy_kwh,
            fiscal_dollars=self.capital.fiscal_dollars,
            note=note,
        )
        self.snapshots.append(snap)
        self.log.debug(f"Treasury snapshot: {snap}")
        return snap

    def reset_weekly_budget(self) -> None:
        """Reset human hours for a new week."""
        if self.weekly_budget_reset:
            self.capital.human_hours = self._initial_human_hours
            self._week_start = time.time()
            self.log.info(
                f"📅 TREASURY: Weekly budget reset. Human hours: {self._initial_human_hours}h"
            )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Get a summary for dashboards/boardroom."""
        runway = self.estimate_runway()
        alert = self.get_budget_alert()

        return {
            "capital": self.capital.to_dict(),
            "runway_weeks": runway,
            "alert_level": alert.value,
            "transactions_today": len([
                t for t in self.transactions
                if t.ts > time.time() - 86400
            ]),
            "total_spent_compute": sum(
                t.amount for t in self.transactions
                if t.kind == "expenditure" and t.resource_type == "compute"
            ),
            "total_spent_human": sum(
                t.amount for t in self.transactions
                if t.kind == "expenditure" and t.resource_type == "human"
            ),
        }

    def get_spending_report(self, days: int = 7) -> Dict[str, Any]:
        """Get spending report for the last N days."""
        cutoff = time.time() - (days * 86400)

        recent_txns = [t for t in self.transactions if t.ts > cutoff]
        expenditures = [t for t in recent_txns if t.kind == "expenditure"]

        by_resource = {}
        for t in expenditures:
            if t.resource_type not in by_resource:
                by_resource[t.resource_type] = 0.0
            by_resource[t.resource_type] += t.amount

        return {
            "period_days": days,
            "total_transactions": len(recent_txns),
            "expenditures_by_resource": by_resource,
            "top_expenditures": sorted(
                expenditures,
                key=lambda t: t.amount,
                reverse=True
            )[:5],
        }

    def get_rejected_proposals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent rejected proposals for analysis."""
        return self._rejected_proposals[-limit:]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_treasury: Optional[Treasury] = None


def get_treasury() -> Treasury:
    """Get the default Treasury instance."""
    global _default_treasury
    if _default_treasury is None:
        _default_treasury = Treasury()
    return _default_treasury


def audit_investment(
    name: str,
    proposal_cost: Dict[str, float],
    expected_roi: float,
) -> bool:
    """Convenience function to audit an investment."""
    return get_treasury().audit_investment(name, proposal_cost, expected_roi)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ResourceType',
    'BudgetAlert',
    'Capital',
    'CapitalSnapshot',
    'Transaction',
    'InvestmentProposal',
    'Treasury',
    'get_treasury',
    'audit_investment',
]
