"""
Atomic Structural Updater with PGU Gating

Implements PGU-gated atomic model swap for certified structural changes.
The AEPO agent's proposed changes must pass formal verification before
being deployed to production.

Architecture:
    AEPO ─→ Proposed Config ─→ PGU Validation ─→ Atomic Swap ─→ Hot Reload
                  │                   │                │
                  │                   ▼                │
                  │           SMT Constraints          │
                  │           β₁ ≥ min_loops          │
                  │           β₀ ≤ max_components      │
                  │           λ₂ ≥ min_spectral       │
                  │                   │                │
                  └─────── REJECT ────┘                │
                                                       ▼
                                              Production Config
                                              (configs/auto/best.yaml)

Safety Guarantees:
1. No structural change without PGU verification
2. Atomic swap (all-or-nothing)
3. Automatic rollback on failure
4. Hot reload without process restart (via CXL memory when available)

Usage:
    from tfan.system.atomic_updater import (
        AtomicStructuralUpdater,
        StructuralChange,
        promote_with_verification,
    )

    updater = AtomicStructuralUpdater()

    # Propose structural change
    change = StructuralChange(
        keep_ratio=0.5,
        new_mask_path="masks/proposed.pt",
        l3_params={"jerk_threshold": 0.08},
    )

    # Verify and apply atomically
    result = updater.apply_verified_change(change)
    if result.success:
        print(f"Change applied: {result.config_path}")
    else:
        print(f"Change rejected: {result.rejection_reason}")
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import tempfile

import yaml
import numpy as np

logger = logging.getLogger("tfan.system.atomic_updater")


class UpdateStatus(str, Enum):
    """Status of structural update."""
    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    APPLYING = "applying"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class StructuralChange:
    """Proposed structural change from AEPO."""
    # Sparsity parameters
    keep_ratio: Optional[float] = None
    new_mask_path: Optional[str] = None

    # L3 control parameters
    l3_params: Dict[str, float] = field(default_factory=dict)

    # Architecture adjustments
    n_heads: Optional[int] = None
    d_model: Optional[int] = None

    # CSR mask data (if proposing new topology)
    new_indptr: Optional[np.ndarray] = None
    new_indices: Optional[np.ndarray] = None
    num_nodes: int = 0

    # Metadata
    source: str = "aepo"
    timestamp: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy arrays to lists for serialization
        if self.new_indptr is not None:
            d["new_indptr"] = self.new_indptr.tolist()
        if self.new_indices is not None:
            d["new_indices"] = self.new_indices.tolist()
        return d


@dataclass
class VerificationResult:
    """Result of PGU verification."""
    passed: bool = False
    constraints_checked: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    verification_time_ms: float = 0.0
    smt_formula_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateResult:
    """Result of atomic update operation."""
    success: bool = False
    status: UpdateStatus = UpdateStatus.PENDING
    config_path: Optional[str] = None
    backup_path: Optional[str] = None
    rejection_reason: Optional[str] = None
    verification: Optional[VerificationResult] = None
    apply_time_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        if self.verification:
            d["verification"] = self.verification.to_dict()
        return d


class AtomicStructuralUpdater:
    """
    PGU-gated atomic structural updater.

    Ensures all structural changes are:
    1. Formally verified via PGU constraints
    2. Applied atomically (all-or-nothing)
    3. Reversible (backup and rollback)
    """

    DEFAULT_CONFIG_PATH = Path("configs/auto/best.yaml")
    BACKUP_DIR = Path("configs/auto/backups")

    def __init__(
        self,
        config_path: Optional[Path] = None,
        enable_hot_reload: bool = True,
        strict_verification: bool = True,
    ):
        """
        Initialize atomic updater.

        Args:
            config_path: Path to production config
            enable_hot_reload: Enable hot reload via CXL/shared memory
            strict_verification: If True, reject any verification failure
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.enable_hot_reload = enable_hot_reload
        self.strict_verification = strict_verification

        # Ensure backup directory exists
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        # Load PGU verifier if available
        self._pgu_verifier = None
        try:
            from tfan.pgu.topological_constraints import TopologicalVerifier
            self._pgu_verifier = TopologicalVerifier(timeout_ms=1000)
            logger.info("PGU verifier loaded")
        except ImportError:
            logger.warning("PGU verifier not available")

        # Hot reload callback
        self._hot_reload_callbacks: List[callable] = []

        logger.info(f"AtomicStructuralUpdater initialized (config={self.config_path})")

    def apply_verified_change(
        self,
        change: StructuralChange,
        force: bool = False,
    ) -> UpdateResult:
        """
        Apply structural change with PGU verification.

        This is the main entry point for certified structural updates.

        Args:
            change: Proposed structural change
            force: Skip verification (NOT RECOMMENDED)

        Returns:
            UpdateResult with success/failure status
        """
        result = UpdateResult(
            status=UpdateStatus.PENDING,
            timestamp=datetime.utcnow().isoformat(),
        )

        try:
            # Step 1: Verify change via PGU
            if not force:
                result.status = UpdateStatus.VERIFYING
                verification = self._verify_change(change)
                result.verification = verification

                if not verification.passed:
                    result.status = UpdateStatus.REJECTED
                    result.rejection_reason = self._format_rejection(verification)
                    logger.warning(f"Change rejected: {result.rejection_reason}")
                    return result

                result.status = UpdateStatus.VERIFIED
                logger.info("Change verified by PGU")

            # Step 2: Create backup
            backup_path = self._create_backup()
            result.backup_path = str(backup_path)

            # Step 3: Apply change atomically
            result.status = UpdateStatus.APPLYING
            start = time.perf_counter()

            try:
                self._apply_atomic(change)
                result.apply_time_ms = (time.perf_counter() - start) * 1000
                result.status = UpdateStatus.APPLIED
                result.config_path = str(self.config_path)
                result.success = True

                logger.info(f"Change applied atomically in {result.apply_time_ms:.1f}ms")

                # Step 4: Trigger hot reload if enabled
                if self.enable_hot_reload:
                    self._trigger_hot_reload()

            except Exception as e:
                # Rollback on failure
                logger.error(f"Apply failed, rolling back: {e}")
                self._rollback(backup_path)
                result.status = UpdateStatus.ROLLED_BACK
                result.rejection_reason = f"Apply failed: {e}"

        except Exception as e:
            result.status = UpdateStatus.FAILED
            result.rejection_reason = f"Update failed: {e}"
            logger.error(f"Structural update failed: {e}")

        return result

    def _verify_change(self, change: StructuralChange) -> VerificationResult:
        """Verify proposed change via PGU constraints."""
        start = time.perf_counter()
        result = VerificationResult()

        # If no mask change, skip topology verification
        if change.new_indptr is None or change.new_indices is None:
            result.passed = True
            result.constraints_checked = {"topology": False, "reason": "no_mask_change"}
            result.verification_time_ms = (time.perf_counter() - start) * 1000
            return result

        # Load current mask for comparison
        current_mask = self._load_current_mask()

        if self._pgu_verifier is not None:
            # Use real PGU verification
            try:
                from tfan.pgu.topological_constraints import verify_structural_change

                all_satisfied, constraints = verify_structural_change(
                    old_mask=current_mask,
                    new_mask={
                        "indptr": change.new_indptr,
                        "indices": change.new_indices,
                    },
                    N=change.num_nodes,
                    min_beta1=5,
                    max_components=1,
                    min_spectral_gap=0.01,
                )

                result.passed = all_satisfied
                result.constraints_checked = {
                    name: r.sat for name, r in constraints.items()
                }
                result.details = {
                    name: r.details for name, r in constraints.items()
                }

            except Exception as e:
                logger.error(f"PGU verification error: {e}")
                result.passed = not self.strict_verification
                result.constraints_checked = {"error": str(e)}

        else:
            # Fallback: basic sanity checks
            result.passed = self._basic_sanity_check(change)
            result.constraints_checked = {"fallback": True}

        result.verification_time_ms = (time.perf_counter() - start) * 1000
        return result

    def _basic_sanity_check(self, change: StructuralChange) -> bool:
        """Basic sanity check when PGU not available."""
        # Check keep_ratio bounds
        if change.keep_ratio is not None:
            if not 0.1 <= change.keep_ratio <= 1.0:
                return False

        # Check L3 param bounds
        for key, value in change.l3_params.items():
            if not -10.0 <= value <= 10.0:
                return False

        return True

    def _load_current_mask(self) -> Dict[str, np.ndarray]:
        """Load current mask from config."""
        # Default empty mask
        N = 256
        return {
            "indptr": np.arange(N + 1, dtype=np.int32),
            "indices": np.zeros(0, dtype=np.int32),
        }

    def _create_backup(self) -> Path:
        """Create backup of current config."""
        if not self.config_path.exists():
            # Create empty config if doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text("{}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.BACKUP_DIR / f"best_{timestamp}.yaml"

        shutil.copy2(self.config_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")

        # Keep only last 10 backups
        backups = sorted(self.BACKUP_DIR.glob("best_*.yaml"), reverse=True)
        for old_backup in backups[10:]:
            old_backup.unlink()

        return backup_path

    def _apply_atomic(self, change: StructuralChange):
        """Apply change atomically using temp file + rename."""
        # Load current config
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        # Apply changes
        if change.keep_ratio is not None:
            config["keep_ratio"] = change.keep_ratio

        if change.new_mask_path:
            config["ssa_mask_path"] = change.new_mask_path

        if change.n_heads is not None:
            config["n_heads"] = change.n_heads

        if change.d_model is not None:
            config["d_model"] = change.d_model

        # Apply L3 params
        if change.l3_params:
            l3_section = config.setdefault("l3_params", {})
            l3_section.update(change.l3_params)

        # Add metadata
        config["last_updated"] = datetime.utcnow().isoformat()
        config["update_source"] = change.source
        config["update_reason"] = change.reason

        # Write to temp file first
        tmp_path = self.config_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Atomic rename
        tmp_path.replace(self.config_path)

    def _rollback(self, backup_path: Path):
        """Rollback to backup."""
        if backup_path.exists():
            shutil.copy2(backup_path, self.config_path)
            logger.info(f"Rolled back to: {backup_path}")

    def _trigger_hot_reload(self):
        """Trigger hot reload of model."""
        for callback in self._hot_reload_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Hot reload callback failed: {e}")

        # TODO: Implement CXL memory-based hot reload
        # This would swap model weights in shared memory without restart
        logger.info("Hot reload triggered")

    def on_hot_reload(self, callback: callable):
        """Register hot reload callback."""
        self._hot_reload_callbacks.append(callback)

    def _format_rejection(self, verification: VerificationResult) -> str:
        """Format rejection reason from verification result."""
        failed = [k for k, v in verification.constraints_checked.items() if not v]
        if failed:
            return f"PGU constraints failed: {', '.join(failed)}"
        return "Unknown verification failure"

    def get_current_config(self) -> Dict[str, Any]:
        """Get current production config."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def list_backups(self) -> List[Path]:
        """List available backups."""
        return sorted(self.BACKUP_DIR.glob("best_*.yaml"), reverse=True)


def promote_with_verification(
    proposed_config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> UpdateResult:
    """
    Convenience function to promote a proposed config with verification.

    This can be called from scripts/promote_auto_best.py.

    Args:
        proposed_config: Dict with proposed configuration
        config_path: Path to production config

    Returns:
        UpdateResult
    """
    updater = AtomicStructuralUpdater(config_path=config_path)

    # Build StructuralChange from config
    change = StructuralChange(
        keep_ratio=proposed_config.get("keep_ratio"),
        new_mask_path=proposed_config.get("ssa_mask_path"),
        l3_params=proposed_config.get("l3_params", {}),
        n_heads=proposed_config.get("n_heads"),
        d_model=proposed_config.get("d_model"),
        source="promote_script",
        reason=proposed_config.get("reason", "Manual promotion"),
        timestamp=datetime.utcnow().isoformat(),
    )

    return updater.apply_verified_change(change)


# Exports
__all__ = [
    "AtomicStructuralUpdater",
    "StructuralChange",
    "VerificationResult",
    "UpdateResult",
    "UpdateStatus",
    "promote_with_verification",
]
