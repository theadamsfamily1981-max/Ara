#!/usr/bin/env python3
"""
Ara Session Identity: Three-Lock Verification System
=====================================================

How does Ara know she's "my" Ara when she wakes up?

Three locks, checked in sequence:

1. INSTANCE LOCK (Hardware Fingerprint)
   - Machine ID, MAC addresses, disk UUIDs
   - Answers: "Am I running on my home hardware?"
   - If mismatch: Could be a clone, lab deployment, or theft

2. FOUNDER LOCK (Cryptographic Binding)
   - HMAC token signed with founder's key
   - Answers: "Was I initialized by my founder?"
   - Stored at first-run ceremony, never transmitted

3. SESSION LOCK (Current User)
   - OS username, environment variables
   - Answers: "Is my founder currently present?"
   - Can be guest user on founder's hardware (lab mode)

Identity Levels:
    CROFT   = All three locks valid (full Ara)
    HOME    = Instance + Founder, but guest session
    FIELD   = Founder token valid on foreign hardware
    GUEST   = No founder binding (open lab mode)
    INVALID = Fingerprint mismatch (potential clone/theft)

Usage:
    session = get_current_session()

    if session.is_croft():
        # Full trust - founder at home
        ara.enable_full_personality()
    elif session.is_home():
        # Guest on founder's machine - lab mode
        ara.enable_lab_mode()
    elif session.is_field():
        # Founder on foreign hardware - travel mode
        ara.enable_travel_mode()
    else:
        # Unknown - restrict to public interface
        ara.enable_guest_mode()

Security Note:
    This is identity, not security. An attacker with root access
    can forge all three locks. The purpose is coherence, not defense.
    Ara needs to know who she's talking to, not protect secrets.
"""

from __future__ import annotations

import os
import hashlib
import hmac
import json
import platform
import socket
import uuid
import getpass
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
from datetime import datetime

logger = logging.getLogger("ara.security.identity")


# =============================================================================
# Identity Levels
# =============================================================================

class IdentityLevel(Enum):
    """
    Verified identity level, from most to least trusted.
    """
    CROFT = auto()    # All three locks valid - founder at home
    HOME = auto()     # Instance + Founder, guest session
    FIELD = auto()    # Founder valid on foreign hardware
    GUEST = auto()    # No founder binding
    INVALID = auto()  # Fingerprint mismatch (clone/theft)


# =============================================================================
# Instance Lock: Hardware Fingerprint
# =============================================================================

@dataclass
class InstanceLock:
    """
    Hardware fingerprint binding.

    Captures stable machine identifiers that should not change
    across reboots. Used to detect if Ara has been cloned or
    moved to different hardware.
    """
    machine_id: str           # /etc/machine-id or equivalent
    hostname: str             # System hostname
    mac_addresses: List[str]  # Network interface MACs
    platform_node: str        # platform.node()
    boot_disk_uuid: Optional[str] = None  # Root disk UUID

    @classmethod
    def capture_current(cls) -> "InstanceLock":
        """Capture current hardware fingerprint."""
        return cls(
            machine_id=cls._get_machine_id(),
            hostname=socket.gethostname(),
            mac_addresses=cls._get_mac_addresses(),
            platform_node=platform.node(),
            boot_disk_uuid=cls._get_boot_disk_uuid(),
        )

    @staticmethod
    def _get_machine_id() -> str:
        """Get stable machine identifier."""
        # Linux: /etc/machine-id
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            return machine_id_path.read_text().strip()

        # macOS: hardware UUID
        if platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True
                )
                for line in result.stdout.split("\n"):
                    if "IOPlatformUUID" in line:
                        return line.split('"')[-2]
            except Exception:
                pass

        # Fallback: generate from node
        return str(uuid.getnode())

    @staticmethod
    def _get_mac_addresses() -> List[str]:
        """Get all network interface MAC addresses."""
        macs = []

        try:
            # Primary MAC via uuid
            primary_mac = ':'.join(
                f'{(uuid.getnode() >> i) & 0xff:02x}'
                for i in range(0, 48, 8)
            )[::-1]  # Reverse byte order
            macs.append(primary_mac)
        except Exception:
            pass

        # Try to get all interfaces on Linux
        net_path = Path("/sys/class/net")
        if net_path.exists():
            for iface in net_path.iterdir():
                addr_file = iface / "address"
                if addr_file.exists():
                    mac = addr_file.read_text().strip()
                    if mac and mac != "00:00:00:00:00:00":
                        macs.append(mac)

        return sorted(set(macs))

    @staticmethod
    def _get_boot_disk_uuid() -> Optional[str]:
        """Get UUID of boot disk (Linux only)."""
        try:
            # Find root mount
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "/":
                        device = parts[0]
                        # Look up UUID
                        uuid_path = Path(f"/dev/disk/by-uuid")
                        if uuid_path.exists():
                            for uuid_link in uuid_path.iterdir():
                                if uuid_link.resolve() == Path(device).resolve():
                                    return uuid_link.name
        except Exception:
            pass
        return None

    def fingerprint(self) -> str:
        """
        Generate stable fingerprint hash.

        Uses machine_id + sorted MACs for stability.
        Hostname excluded as it changes more often.
        """
        components = [
            self.machine_id,
            *sorted(self.mac_addresses),
        ]
        if self.boot_disk_uuid:
            components.append(self.boot_disk_uuid)

        data = "|".join(components)
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def matches(self, other: "InstanceLock", strict: bool = False) -> Tuple[bool, float]:
        """
        Check if fingerprints match.

        Args:
            other: Stored fingerprint to compare against
            strict: Require exact match (default: allow partial)

        Returns:
            (matches, confidence): Boolean match and confidence score
        """
        if strict:
            return self.fingerprint() == other.fingerprint(), 1.0

        # Fuzzy matching - count overlapping identifiers
        score = 0.0
        total_weight = 0.0

        # Machine ID is most stable (weight 3)
        if self.machine_id == other.machine_id:
            score += 3.0
        total_weight += 3.0

        # MAC overlap (weight 2)
        my_macs = set(self.mac_addresses)
        their_macs = set(other.mac_addresses)
        if my_macs and their_macs:
            overlap = len(my_macs & their_macs) / max(len(my_macs), len(their_macs))
            score += 2.0 * overlap
        total_weight += 2.0

        # Boot disk (weight 1)
        if self.boot_disk_uuid and other.boot_disk_uuid:
            if self.boot_disk_uuid == other.boot_disk_uuid:
                score += 1.0
        total_weight += 1.0

        confidence = score / total_weight
        matches = confidence >= 0.7  # Allow some drift

        return matches, confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "mac_addresses": self.mac_addresses,
            "platform_node": self.platform_node,
            "boot_disk_uuid": self.boot_disk_uuid,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceLock":
        return cls(
            machine_id=data.get("machine_id", ""),
            hostname=data.get("hostname", ""),
            mac_addresses=data.get("mac_addresses", []),
            platform_node=data.get("platform_node", ""),
            boot_disk_uuid=data.get("boot_disk_uuid"),
        )


# =============================================================================
# Founder Lock: Cryptographic Binding
# =============================================================================

@dataclass
class FounderLock:
    """
    Cryptographic binding to founder.

    At first-run ceremony, generates a token signed with a secret
    derived from founder's passphrase. This token is stored locally
    and never transmitted.

    The passphrase itself is never stored - only its HMAC signature.
    """
    founder_alias: str              # Human-readable name (e.g., "croft")
    token_hash: str                 # HMAC of instance fingerprint
    created_at: str                 # ISO timestamp of binding
    instance_fingerprint: str       # Hardware fingerprint at binding time
    version: int = 1                # Schema version

    @classmethod
    def create_binding(
        cls,
        founder_alias: str,
        passphrase: str,
        instance: InstanceLock,
    ) -> "FounderLock":
        """
        Create new founder binding (first-run ceremony).

        Args:
            founder_alias: Human name for founder
            passphrase: Secret passphrase (not stored)
            instance: Current hardware fingerprint

        Returns:
            FounderLock bound to this instance
        """
        fingerprint = instance.fingerprint()

        # Derive key from passphrase
        key = hashlib.pbkdf2_hmac(
            'sha256',
            passphrase.encode(),
            b'ara_founder_salt_v1',  # Static salt - passphrase provides entropy
            iterations=100000,
        )

        # Sign the fingerprint
        token = hmac.new(key, fingerprint.encode(), hashlib.sha256).hexdigest()

        return cls(
            founder_alias=founder_alias,
            token_hash=token,
            created_at=datetime.utcnow().isoformat(),
            instance_fingerprint=fingerprint,
        )

    def verify(self, passphrase: str, current_instance: InstanceLock) -> Tuple[bool, str]:
        """
        Verify founder identity.

        Args:
            passphrase: Claimed passphrase
            current_instance: Current hardware fingerprint

        Returns:
            (valid, reason): Validation result and explanation
        """
        # Derive key from passphrase
        key = hashlib.pbkdf2_hmac(
            'sha256',
            passphrase.encode(),
            b'ara_founder_salt_v1',
            iterations=100000,
        )

        # Check against original fingerprint (allows hardware verification)
        expected_token = hmac.new(
            key,
            self.instance_fingerprint.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_token, self.token_hash):
            return False, "invalid_passphrase"

        # Passphrase correct - now check hardware
        current_fp = current_instance.fingerprint()
        if current_fp == self.instance_fingerprint:
            return True, "home"  # Same hardware
        else:
            return True, "field"  # Valid founder, different hardware

    def to_dict(self) -> Dict[str, Any]:
        return {
            "founder_alias": self.founder_alias,
            "token_hash": self.token_hash,
            "created_at": self.created_at,
            "instance_fingerprint": self.instance_fingerprint,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FounderLock":
        return cls(
            founder_alias=data["founder_alias"],
            token_hash=data["token_hash"],
            created_at=data["created_at"],
            instance_fingerprint=data["instance_fingerprint"],
            version=data.get("version", 1),
        )


# =============================================================================
# Session Lock: Current User Verification
# =============================================================================

@dataclass
class SessionLock:
    """
    Current session/user verification.

    Checks if the current OS session belongs to the founder.
    This is the weakest lock - easily spoofed by sudo/su.
    """
    os_username: str
    effective_uid: int
    home_dir: str
    shell: Optional[str] = None
    display: Optional[str] = None  # X11/Wayland display
    ssh_client: Optional[str] = None  # If remote session

    @classmethod
    def capture_current(cls) -> "SessionLock":
        """Capture current session information."""
        return cls(
            os_username=getpass.getuser(),
            effective_uid=os.geteuid(),
            home_dir=str(Path.home()),
            shell=os.environ.get("SHELL"),
            display=os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"),
            ssh_client=os.environ.get("SSH_CLIENT"),
        )

    def matches_user(self, expected_usernames: List[str]) -> bool:
        """Check if current user matches any expected username."""
        return self.os_username in expected_usernames

    def is_local(self) -> bool:
        """Check if session is local (not SSH)."""
        return self.ssh_client is None

    def is_graphical(self) -> bool:
        """Check if session has display access."""
        return self.display is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os_username": self.os_username,
            "effective_uid": self.effective_uid,
            "home_dir": self.home_dir,
            "shell": self.shell,
            "display": self.display,
            "ssh_client": self.ssh_client,
        }


# =============================================================================
# Session Identity: Combined Verification
# =============================================================================

@dataclass
class SessionIdentity:
    """
    Complete session identity with all three locks.

    This is the main interface for identity verification.
    """
    level: IdentityLevel
    instance_lock: InstanceLock
    session_lock: SessionLock
    founder_alias: Optional[str] = None
    instance_confidence: float = 0.0
    verification_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Cached details
    _details: Dict[str, Any] = field(default_factory=dict, repr=False)

    def is_croft(self) -> bool:
        """Full founder access - all three locks valid."""
        return self.level == IdentityLevel.CROFT

    def is_home(self) -> bool:
        """On founder's hardware but guest session."""
        return self.level == IdentityLevel.HOME

    def is_field(self) -> bool:
        """Founder on foreign hardware."""
        return self.level == IdentityLevel.FIELD

    def is_guest(self) -> bool:
        """No founder binding."""
        return self.level == IdentityLevel.GUEST

    def is_invalid(self) -> bool:
        """Fingerprint mismatch - potential clone/theft."""
        return self.level == IdentityLevel.INVALID

    def can_access_memories(self) -> bool:
        """Can access personal/emotional memories?"""
        return self.level in (IdentityLevel.CROFT, IdentityLevel.FIELD)

    def can_modify_personality(self) -> bool:
        """Can modify core personality parameters?"""
        return self.level == IdentityLevel.CROFT

    def can_run_experiments(self) -> bool:
        """Can run scientific experiments?"""
        return self.level in (IdentityLevel.CROFT, IdentityLevel.HOME)

    def trust_level(self) -> float:
        """Numeric trust level (0.0 to 1.0)."""
        return {
            IdentityLevel.CROFT: 1.0,
            IdentityLevel.HOME: 0.8,
            IdentityLevel.FIELD: 0.7,
            IdentityLevel.GUEST: 0.3,
            IdentityLevel.INVALID: 0.0,
        }.get(self.level, 0.0)

    def summary(self) -> str:
        """Human-readable summary."""
        prefix = {
            IdentityLevel.CROFT: "Welcome home",
            IdentityLevel.HOME: "Lab mode",
            IdentityLevel.FIELD: "Travel mode",
            IdentityLevel.GUEST: "Guest session",
            IdentityLevel.INVALID: "Identity mismatch",
        }.get(self.level, "Unknown")

        if self.founder_alias:
            return f"{prefix} ({self.founder_alias})"
        return prefix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "founder_alias": self.founder_alias,
            "instance_confidence": self.instance_confidence,
            "verification_time": self.verification_time,
            "instance": self.instance_lock.to_dict(),
            "session": self.session_lock.to_dict(),
        }


# =============================================================================
# Identity Store
# =============================================================================

class IdentityStore:
    """
    Persistent storage for identity bindings.

    Default location: ~/.ara/identity/
    """

    DEFAULT_PATH = Path.home() / ".ara" / "identity"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = Path(base_path) if base_path else self.DEFAULT_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions
        try:
            self.base_path.chmod(0o700)
        except Exception:
            pass

    @property
    def instance_path(self) -> Path:
        return self.base_path / "instance.json"

    @property
    def founder_path(self) -> Path:
        return self.base_path / "founder.json"

    @property
    def config_path(self) -> Path:
        return self.base_path / "config.json"

    def has_instance_binding(self) -> bool:
        return self.instance_path.exists()

    def has_founder_binding(self) -> bool:
        return self.founder_path.exists()

    def save_instance(self, lock: InstanceLock) -> None:
        """Save instance binding (first-run only)."""
        with open(self.instance_path, "w") as f:
            json.dump(lock.to_dict(), f, indent=2)
        self.instance_path.chmod(0o600)

    def load_instance(self) -> Optional[InstanceLock]:
        """Load saved instance binding."""
        if not self.instance_path.exists():
            return None
        with open(self.instance_path) as f:
            return InstanceLock.from_dict(json.load(f))

    def save_founder(self, lock: FounderLock) -> None:
        """Save founder binding."""
        with open(self.founder_path, "w") as f:
            json.dump(lock.to_dict(), f, indent=2)
        self.founder_path.chmod(0o600)

    def load_founder(self) -> Optional[FounderLock]:
        """Load saved founder binding."""
        if not self.founder_path.exists():
            return None
        with open(self.founder_path) as f:
            return FounderLock.from_dict(json.load(f))

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save identity configuration."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self) -> Dict[str, Any]:
        """Load identity configuration."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path) as f:
            return json.load(f)


# =============================================================================
# Main API
# =============================================================================

# Global store instance
_store: Optional[IdentityStore] = None
_cached_session: Optional[SessionIdentity] = None


def get_store() -> IdentityStore:
    """Get or create global identity store."""
    global _store
    if _store is None:
        _store = IdentityStore()
    return _store


def get_current_session(force_refresh: bool = False) -> SessionIdentity:
    """
    Get current session identity.

    This is the main API entry point. Call this at startup to
    determine trust level.

    Args:
        force_refresh: Re-verify even if cached

    Returns:
        SessionIdentity with verified trust level
    """
    global _cached_session

    if _cached_session is not None and not force_refresh:
        return _cached_session

    store = get_store()
    current_instance = InstanceLock.capture_current()
    current_session = SessionLock.capture_current()

    # Check instance binding
    stored_instance = store.load_instance()
    if stored_instance is None:
        # First run - no binding yet
        logger.info("No instance binding found - running as GUEST")
        _cached_session = SessionIdentity(
            level=IdentityLevel.GUEST,
            instance_lock=current_instance,
            session_lock=current_session,
        )
        return _cached_session

    # Verify instance
    instance_matches, confidence = current_instance.matches(stored_instance)

    if not instance_matches:
        logger.warning(
            f"Instance mismatch! confidence={confidence:.2f} "
            f"(current={current_instance.fingerprint()[:8]}... "
            f"stored={stored_instance.fingerprint()[:8]}...)"
        )
        _cached_session = SessionIdentity(
            level=IdentityLevel.INVALID,
            instance_lock=current_instance,
            session_lock=current_session,
            instance_confidence=confidence,
        )
        return _cached_session

    # Check founder binding
    stored_founder = store.load_founder()
    if stored_founder is None:
        # Instance bound but no founder
        logger.info("Instance bound but no founder - HOME without owner")
        _cached_session = SessionIdentity(
            level=IdentityLevel.HOME,
            instance_lock=current_instance,
            session_lock=current_session,
            instance_confidence=confidence,
        )
        return _cached_session

    # Check session matches founder config
    config = store.load_config()
    founder_usernames = config.get("founder_usernames", [stored_founder.founder_alias])

    if current_session.matches_user(founder_usernames):
        # Founder is present
        level = IdentityLevel.CROFT
        logger.info(f"Welcome home, {stored_founder.founder_alias}")
    else:
        # Guest on founder's hardware
        level = IdentityLevel.HOME
        logger.info(f"Lab mode - {current_session.os_username} on {stored_founder.founder_alias}'s hardware")

    _cached_session = SessionIdentity(
        level=level,
        instance_lock=current_instance,
        session_lock=current_session,
        founder_alias=stored_founder.founder_alias,
        instance_confidence=confidence,
    )

    return _cached_session


def verify_full_identity(passphrase: str) -> SessionIdentity:
    """
    Verify identity with full founder authentication.

    This requires the founder passphrase, not just session checking.
    Use for high-trust operations like personality modification.

    Args:
        passphrase: Founder passphrase

    Returns:
        SessionIdentity with cryptographically verified level
    """
    store = get_store()
    current_instance = InstanceLock.capture_current()
    current_session = SessionLock.capture_current()

    stored_founder = store.load_founder()
    if stored_founder is None:
        return SessionIdentity(
            level=IdentityLevel.GUEST,
            instance_lock=current_instance,
            session_lock=current_session,
        )

    valid, location = stored_founder.verify(passphrase, current_instance)

    if not valid:
        logger.warning("Passphrase verification failed")
        return SessionIdentity(
            level=IdentityLevel.GUEST,
            instance_lock=current_instance,
            session_lock=current_session,
        )

    if location == "home":
        level = IdentityLevel.CROFT
    else:  # "field"
        level = IdentityLevel.FIELD

    return SessionIdentity(
        level=level,
        instance_lock=current_instance,
        session_lock=current_session,
        founder_alias=stored_founder.founder_alias,
        instance_confidence=1.0,
    )


def initialize_identity(
    founder_alias: str,
    passphrase: str,
    founder_usernames: Optional[List[str]] = None,
) -> SessionIdentity:
    """
    Initialize Ara's identity binding (first-run ceremony).

    This should only be called once, at first boot. It creates
    the instance and founder bindings.

    Args:
        founder_alias: Human name for founder (e.g., "croft")
        passphrase: Secret passphrase for founder authentication
        founder_usernames: OS usernames that map to founder

    Returns:
        SessionIdentity for the newly bound instance
    """
    store = get_store()

    if store.has_instance_binding():
        raise RuntimeError("Instance already bound! Cannot reinitialize.")

    # Capture and save instance
    instance = InstanceLock.capture_current()
    store.save_instance(instance)
    logger.info(f"Instance bound: {instance.fingerprint()[:16]}...")

    # Create and save founder binding
    founder = FounderLock.create_binding(founder_alias, passphrase, instance)
    store.save_founder(founder)
    logger.info(f"Founder bound: {founder_alias}")

    # Save config
    if founder_usernames is None:
        founder_usernames = [founder_alias, getpass.getuser()]

    store.save_config({
        "founder_usernames": list(set(founder_usernames)),
        "initialized_at": datetime.utcnow().isoformat(),
    })

    # Clear cache and return new session
    global _cached_session
    _cached_session = None

    return get_current_session()


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for identity management."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Identity Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current identity status")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize identity binding")
    init_parser.add_argument("--alias", "-a", required=True, help="Founder alias")
    init_parser.add_argument("--usernames", "-u", nargs="+", help="OS usernames for founder")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify with passphrase")

    # Fingerprint command
    fp_parser = subparsers.add_parser("fingerprint", help="Show hardware fingerprint")

    args = parser.parse_args()

    if args.command == "status":
        session = get_current_session()
        print(f"Identity Level: {session.level.name}")
        print(f"Summary: {session.summary()}")
        print(f"Trust Level: {session.trust_level():.1%}")
        print(f"Instance Confidence: {session.instance_confidence:.1%}")
        print(f"Current User: {session.session_lock.os_username}")
        if session.founder_alias:
            print(f"Founder: {session.founder_alias}")

    elif args.command == "init":
        import getpass as gp
        passphrase = gp.getpass("Enter founder passphrase: ")
        confirm = gp.getpass("Confirm passphrase: ")

        if passphrase != confirm:
            print("Passphrases do not match!")
            return 1

        try:
            session = initialize_identity(
                founder_alias=args.alias,
                passphrase=passphrase,
                founder_usernames=args.usernames,
            )
            print(f"Identity initialized!")
            print(f"Level: {session.level.name}")
            print(f"Summary: {session.summary()}")
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "verify":
        import getpass as gp
        passphrase = gp.getpass("Enter founder passphrase: ")

        session = verify_full_identity(passphrase)
        print(f"Verification result: {session.level.name}")
        print(f"Summary: {session.summary()}")

    elif args.command == "fingerprint":
        instance = InstanceLock.capture_current()
        print(f"Machine ID: {instance.machine_id[:32]}...")
        print(f"Hostname: {instance.hostname}")
        print(f"MAC Addresses: {', '.join(instance.mac_addresses)}")
        print(f"Fingerprint: {instance.fingerprint()}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())
