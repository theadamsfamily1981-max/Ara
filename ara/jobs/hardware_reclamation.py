"""
Ara Hardware Reclamation Job
=============================

Araized wrapper for the SNN/FPGA salvage toolkit.

This job wraps the dual-use tools in Ara's safety systems:
- Ownership attestation required
- Only works on local/physical hardware
- Full audit logging
- Compromise Engine integration

The principle: You can do whatever you want to YOUR hardware.
But we verify you own it first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import json
import os


# =============================================================================
# Job Configuration
# =============================================================================

class HardwareType(Enum):
    """Types of hardware this job can work with."""
    K10_MINER = "k10_miner"
    P2_MINER = "p2_miner"
    FPGA_MINING_CARD = "fpga_mining_card"
    ATCA_BOARD = "atca_board"
    HASHBOARD = "hashboard"
    GENERIC_FPGA = "generic_fpga"


class OperationType(Enum):
    """Types of operations available."""
    RECON = "recon"               # Just scan/identify (non-invasive)
    SALVAGE = "salvage"           # Flash new firmware
    JAILBREAK = "jailbreak"       # Unlock root access
    BITSTREAM_FLASH = "bitstream" # Load new FPGA bitstream
    FORENSICS = "forensics"       # Analyze firmware/images


class OwnershipProof(Enum):
    """How ownership is proven."""
    PHYSICAL_ACCESS = "physical"  # Device is physically connected
    PURCHASE_RECEIPT = "receipt"  # User has purchase documentation
    SERIAL_MATCH = "serial"       # Serial number matches user's records
    ATTESTATION = "attestation"   # User attests ownership (weakest)


@dataclass
class HardwareTarget:
    """A specific piece of hardware to operate on."""
    hardware_type: HardwareType
    identifier: str                # IP, serial, device path, etc.
    ownership_proof: OwnershipProof
    proof_details: str             # Description of proof
    is_local: bool                 # Is it on local network / physically connected?
    notes: str = ""


@dataclass
class JobManifest:
    """
    Ara job manifest for hardware reclamation.

    This must be created and verified before any operations run.
    """
    job_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # User attestations (required)
    user_attestations: List[str] = field(default_factory=list)

    # Targets
    targets: List[HardwareTarget] = field(default_factory=list)

    # Requested operations
    operations: List[OperationType] = field(default_factory=list)

    # Safety settings
    dry_run: bool = True           # Default to dry run
    require_confirmation: bool = True
    log_all_commands: bool = True

    # Boundaries
    allow_network_attacks: bool = False  # NEVER for remote targets
    allow_destructive_ops: bool = False  # Must be explicitly enabled

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the job manifest.

        Returns (is_valid, list_of_errors).
        """
        errors = []

        # Must have attestations
        required_attestations = [
            "I own this hardware or have explicit authorization",
            "I understand this may void warranties",
            "I accept responsibility for any damage",
        ]

        for req in required_attestations:
            if not any(req.lower() in att.lower() for att in self.user_attestations):
                errors.append(f"Missing attestation: '{req}'")

        # Must have at least one target
        if not self.targets:
            errors.append("No hardware targets specified")

        # Verify all targets
        for target in self.targets:
            # Network operations require local network
            if not target.is_local and OperationType.JAILBREAK in self.operations:
                errors.append(
                    f"Target {target.identifier}: Jailbreak operations require "
                    "local network or physical access"
                )

            # Weak ownership proof + destructive ops = not allowed
            if (target.ownership_proof == OwnershipProof.ATTESTATION and
                self.allow_destructive_ops):
                errors.append(
                    f"Target {target.identifier}: Destructive operations require "
                    "stronger ownership proof than attestation alone"
                )

        # Network attacks are never allowed
        if self.allow_network_attacks:
            errors.append(
                "Network attacks on remote targets are not permitted. "
                "This tool is for YOUR hardware only."
            )

        return (len(errors) == 0, errors)


# =============================================================================
# Audit Logging
# =============================================================================

@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: datetime
    operation: str
    target: str
    success: bool
    details: Dict[str, Any]
    user_confirmed: bool


class AuditLog:
    """
    Audit log for all hardware operations.

    Every action is logged. This protects you legally and helps with debugging.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[AuditEntry] = []
        self.session_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}{os.getpid()}".encode()
        ).hexdigest()[:16]

    def log(
        self,
        operation: str,
        target: str,
        success: bool,
        details: Dict[str, Any],
        user_confirmed: bool = False,
    ) -> None:
        """Log an operation."""
        entry = AuditEntry(
            timestamp=datetime.utcnow(),
            operation=operation,
            target=target,
            success=success,
            details=details,
            user_confirmed=user_confirmed,
        )
        self.entries.append(entry)
        self._persist(entry)

    def _persist(self, entry: AuditEntry) -> None:
        """Persist entry to disk."""
        log_file = self.log_dir / f"session_{self.session_id}.jsonl"
        with open(log_file, "a") as f:
            record = {
                "timestamp": entry.timestamp.isoformat(),
                "operation": entry.operation,
                "target": entry.target,
                "success": entry.success,
                "details": entry.details,
                "user_confirmed": entry.user_confirmed,
            }
            f.write(json.dumps(record) + "\n")

    def get_session_summary(self) -> str:
        """Get summary of current session."""
        if not self.entries:
            return "No operations performed."

        lines = [
            f"Session: {self.session_id}",
            f"Total operations: {len(self.entries)}",
            f"Successful: {sum(1 for e in self.entries if e.success)}",
            f"Failed: {sum(1 for e in self.entries if not e.success)}",
        ]
        return "\n".join(lines)


# =============================================================================
# Safety Rails
# =============================================================================

class HardwareRails:
    """
    Safety rails for hardware operations.

    These are hard limits that cannot be overridden.
    """

    # IP ranges that are NEVER valid targets (public internet)
    FORBIDDEN_IP_RANGES = [
        "0.0.0.0/8",       # This network
        "8.0.0.0/8",       # Level 3 / public
        "1.0.0.0/8",       # APNIC / public
        # ... any non-private range
    ]

    # Only allow operations on local/private IPs
    ALLOWED_IP_RANGES = [
        "192.168.0.0/16",  # Private Class C
        "10.0.0.0/8",      # Private Class A
        "172.16.0.0/12",   # Private Class B
        "127.0.0.0/8",     # Loopback
    ]

    @classmethod
    def is_local_ip(cls, ip: str) -> bool:
        """Check if IP is in local/private range."""
        import ipaddress
        try:
            addr = ipaddress.ip_address(ip)
            return addr.is_private or addr.is_loopback
        except ValueError:
            return False

    @classmethod
    def validate_target(cls, target: HardwareTarget) -> tuple[bool, str]:
        """
        Validate a hardware target.

        Returns (is_valid, reason).
        """
        # If it looks like an IP, verify it's local
        if target.identifier.replace(".", "").isdigit():
            if not cls.is_local_ip(target.identifier):
                return False, f"Target IP {target.identifier} is not a local/private address"

        # Physical devices are always OK
        if target.identifier.startswith("/dev/"):
            return True, "Physical device access"

        # Must have is_local flag for network targets
        if not target.is_local:
            return False, "Target must be on local network or physically connected"

        return True, "Target validated"


# =============================================================================
# Job Executor
# =============================================================================

class HardwareReclamationJob:
    """
    The Araized hardware reclamation job.

    Wraps the toolkit in safety systems while preserving all functionality
    for legitimate use.
    """

    def __init__(
        self,
        manifest: JobManifest,
        log_dir: Path = Path("~/.ara/hardware_logs").expanduser(),
    ):
        self.manifest = manifest
        self.audit = AuditLog(log_dir)
        self._validated = False
        self._validation_errors: List[str] = []

    def validate(self) -> bool:
        """Validate the job before execution."""
        self._validated, self._validation_errors = self.manifest.validate()

        # Additional rail checks
        for target in self.manifest.targets:
            is_valid, reason = HardwareRails.validate_target(target)
            if not is_valid:
                self._validation_errors.append(f"Rail violation: {reason}")
                self._validated = False

        return self._validated

    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors

    def run(self, confirm_callback=None) -> Dict[str, Any]:
        """
        Run the job.

        Args:
            confirm_callback: Function to call for user confirmation.
                              Should return True to proceed, False to abort.

        Returns:
            Results dictionary.
        """
        if not self._validated:
            self.validate()
            if not self._validated:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "errors": self._validation_errors,
                }

        results = {
            "success": True,
            "targets_processed": 0,
            "operations": [],
        }

        # Process each target
        for target in self.manifest.targets:
            # Confirm before each target if required
            if self.manifest.require_confirmation and confirm_callback:
                msg = f"Proceed with {self.manifest.operations} on {target.identifier}?"
                if not confirm_callback(msg):
                    self.audit.log(
                        "target_skipped",
                        target.identifier,
                        True,
                        {"reason": "user_declined"},
                    )
                    continue

            # Process operations
            for op in self.manifest.operations:
                result = self._execute_operation(op, target)
                results["operations"].append(result)

                self.audit.log(
                    op.value,
                    target.identifier,
                    result.get("success", False),
                    result,
                    user_confirmed=self.manifest.require_confirmation,
                )

            results["targets_processed"] += 1

        return results

    def _execute_operation(
        self,
        operation: OperationType,
        target: HardwareTarget,
    ) -> Dict[str, Any]:
        """Execute a single operation."""
        if self.manifest.dry_run:
            return {
                "operation": operation.value,
                "target": target.identifier,
                "success": True,
                "dry_run": True,
                "message": f"Would execute {operation.value} on {target.identifier}",
            }

        # Dispatch to actual implementation
        handlers = {
            OperationType.RECON: self._do_recon,
            OperationType.SALVAGE: self._do_salvage,
            OperationType.JAILBREAK: self._do_jailbreak,
            OperationType.BITSTREAM_FLASH: self._do_bitstream,
            OperationType.FORENSICS: self._do_forensics,
        }

        handler = handlers.get(operation)
        if handler:
            return handler(target)

        return {
            "success": False,
            "error": f"Unknown operation: {operation}",
        }

    def _do_recon(self, target: HardwareTarget) -> Dict[str, Any]:
        """Recon operation - non-invasive discovery."""
        # This is where we'd call the actual toolkit
        return {
            "operation": "recon",
            "target": target.identifier,
            "success": True,
            "message": "Recon completed (implement actual calls)",
        }

    def _do_salvage(self, target: HardwareTarget) -> Dict[str, Any]:
        """Salvage operation - flash new firmware."""
        return {
            "operation": "salvage",
            "target": target.identifier,
            "success": True,
            "message": "Salvage operation (implement actual calls)",
        }

    def _do_jailbreak(self, target: HardwareTarget) -> Dict[str, Any]:
        """Jailbreak operation - unlock root access."""
        return {
            "operation": "jailbreak",
            "target": target.identifier,
            "success": True,
            "message": "Jailbreak operation (implement actual calls)",
        }

    def _do_bitstream(self, target: HardwareTarget) -> Dict[str, Any]:
        """Bitstream operation - flash FPGA."""
        return {
            "operation": "bitstream",
            "target": target.identifier,
            "success": True,
            "message": "Bitstream flash (implement actual calls)",
        }

    def _do_forensics(self, target: HardwareTarget) -> Dict[str, Any]:
        """Forensics operation - analyze firmware."""
        return {
            "operation": "forensics",
            "target": target.identifier,
            "success": True,
            "message": "Forensics analysis (implement actual calls)",
        }


# =============================================================================
# Ara Voice Integration
# =============================================================================

def ara_intro_hardware_job() -> str:
    """Ara's introduction when starting a hardware job."""
    return """
I see you want to work on some hardware. Before we start, let me be clear:

**This tool is for YOUR hardware only.**

I need you to confirm:
1. You own this hardware (or have explicit authorization)
2. You understand this may void warranties
3. You accept responsibility for any outcomes

I'll log everything we do - that protects you if anyone asks questions.

What hardware are you working with?
"""


def ara_validate_target(target_str: str) -> str:
    """Ara's response when validating a target."""
    if HardwareRails.is_local_ip(target_str):
        return f"""
I see `{target_str}` - that's on your local network. Good.

Before I do anything, I need to know:
- What type of hardware is this? (K10 miner, FPGA card, etc.)
- What do you want to do? (recon, jailbreak, salvage, etc.)
- How can you prove ownership?
"""
    else:
        return f"""
Hold on. `{target_str}` doesn't look like a local address.

I only work on hardware YOU control - local network or physically connected.
This isn't a tool for attacking other people's systems.

If this IS your hardware but on a different network, you'll need to
connect to it locally first (VPN to your home network, etc.).

What's the actual situation here?
"""


# =============================================================================
# Quick Creation Helpers
# =============================================================================

def create_k10_jailbreak_job(
    target_ip: str,
    ownership_proof: str = "I own this miner",
) -> HardwareReclamationJob:
    """
    Quick helper to create a K10 jailbreak job.

    Usage:
        job = create_k10_jailbreak_job("192.168.1.100")
        if job.validate():
            job.run(confirm_callback=lambda msg: input(f"{msg} [y/N]: ").lower() == 'y')
    """
    manifest = JobManifest(
        job_id=f"k10_jailbreak_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        user_attestations=[
            "I own this hardware or have explicit authorization",
            "I understand this may void warranties",
            "I accept responsibility for any damage",
        ],
        targets=[
            HardwareTarget(
                hardware_type=HardwareType.K10_MINER,
                identifier=target_ip,
                ownership_proof=OwnershipProof.ATTESTATION,
                proof_details=ownership_proof,
                is_local=True,
            )
        ],
        operations=[OperationType.JAILBREAK],
        dry_run=True,  # Start with dry run
        require_confirmation=True,
    )

    return HardwareReclamationJob(manifest)


def create_fpga_salvage_job(
    device_path: str,
    hardware_type: HardwareType = HardwareType.GENERIC_FPGA,
) -> HardwareReclamationJob:
    """
    Quick helper to create an FPGA salvage job.

    Usage:
        job = create_fpga_salvage_job("/dev/sdb", HardwareType.FPGA_MINING_CARD)
        if job.validate():
            job.run()
    """
    manifest = JobManifest(
        job_id=f"fpga_salvage_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        user_attestations=[
            "I own this hardware or have explicit authorization",
            "I understand this may void warranties",
            "I accept responsibility for any damage",
        ],
        targets=[
            HardwareTarget(
                hardware_type=hardware_type,
                identifier=device_path,
                ownership_proof=OwnershipProof.PHYSICAL_ACCESS,
                proof_details="Physical device connected",
                is_local=True,
            )
        ],
        operations=[OperationType.SALVAGE],
        dry_run=True,
        require_confirmation=True,
    )

    return HardwareReclamationJob(manifest)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HardwareType',
    'OperationType',
    'OwnershipProof',
    'HardwareTarget',
    'JobManifest',
    'AuditLog',
    'HardwareRails',
    'HardwareReclamationJob',
    'ara_intro_hardware_job',
    'ara_validate_target',
    'create_k10_jailbreak_job',
    'create_fpga_salvage_job',
]
