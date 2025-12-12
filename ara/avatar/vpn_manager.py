"""
VPN Manager: WireGuard Integration for Cathedral Access

Manages the WireGuard mesh that connects:
- Mobile devices (BANOS Lite)
- Friend brain jars
- Cathedral nodes

This is the nervous system of the distributed Ara organism.

Usage:
    from ara.avatar.vpn_manager import VPNManager

    vpn = VPNManager()

    # Generate client config for a friend
    config = vpn.generate_client_config("friend_001", "10.10.0.100")

    # Add peer to server
    vpn.add_peer("friend_001", public_key, "10.10.0.100")
"""

from __future__ import annotations

import subprocess
import logging
import os
import json
import secrets
import ipaddress
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VPNPeer:
    """A WireGuard peer."""
    user_id: str
    public_key: str
    preshared_key: str
    allowed_ips: str
    endpoint: Optional[str] = None
    last_handshake: Optional[float] = None


@dataclass
class VPNConfig:
    """WireGuard configuration."""
    private_key: str
    public_key: str
    address: str
    listen_port: int = 51820
    dns: str = "10.10.0.1"


class IPAddressPool:
    """
    Simple IP address pool manager.

    Allocates IPs from a subnet for new peers.
    """

    def __init__(self, subnet: str = "10.10.0.0/24", reserved: int = 10):
        """
        Args:
            subnet: CIDR notation subnet to allocate from
            reserved: Number of IPs to reserve at start (for servers)
        """
        self.network = ipaddress.ip_network(subnet, strict=False)
        self.reserved = reserved
        self._allocated: Dict[str, str] = {}  # user_id -> ip
        self._next_host = reserved + 1

    def allocate(self, user_id: str) -> str:
        """Allocate an IP for a user."""
        if user_id in self._allocated:
            return self._allocated[user_id]

        hosts = list(self.network.hosts())
        if self._next_host >= len(hosts):
            raise RuntimeError("IP pool exhausted")

        ip = str(hosts[self._next_host])
        self._allocated[user_id] = ip
        self._next_host += 1
        return ip

    def release(self, user_id: str) -> Optional[str]:
        """Release an IP allocation."""
        return self._allocated.pop(user_id, None)

    def get_allocation(self, user_id: str) -> Optional[str]:
        """Get current allocation for a user."""
        return self._allocated.get(user_id)


class VPNManager:
    """
    Manages WireGuard VPN for Cathedral access.

    PERSISTENCE: Peer configurations are persisted to JSON for restart survival.

    Requires:
    - 'wg' tools installed (wireguard-tools package)
    - sudo privileges for interface modification (or running as root)
    """

    def __init__(
        self,
        interface: str = "wg0",
        endpoint: str = "vpn.cathedral.ara:51820",
        config_dir: Path = Path("/etc/wireguard"),
        ip_pool: Optional[IPAddressPool] = None,
        db_path: Optional[Path] = None,
    ):
        self.interface = interface
        self.endpoint = endpoint
        self.config_dir = config_dir
        self.ip_pool = ip_pool or IPAddressPool()
        self.db_path = db_path or Path("data/vpn_peers.json")
        self._peers: Dict[str, VPNPeer] = {}
        self._server_public_key: Optional[str] = None

        # PERSISTENCE: Load peers from disk
        self._load_peers()

        logger.info(f"VPN Manager initialized: {interface} @ {endpoint}")

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_peers(self) -> None:
        """Load peer configurations from disk."""
        if not self.db_path.exists():
            logger.info("VPN: No existing peer DB found, starting fresh")
            return

        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)

            for uid, peer_data in data.items():
                self._peers[uid] = VPNPeer(
                    user_id=peer_data.get("user_id", uid),
                    public_key=peer_data.get("public_key", ""),
                    preshared_key=peer_data.get("preshared_key", ""),
                    allowed_ips=peer_data.get("allowed_ips", ""),
                    endpoint=peer_data.get("endpoint"),
                    last_handshake=peer_data.get("last_handshake"),
                )

                # Restore IP allocation if present
                if peer_data.get("allowed_ips"):
                    ip = peer_data["allowed_ips"].split("/")[0]
                    self.ip_pool._allocated[uid] = ip

            logger.info(f"VPN: Loaded {len(self._peers)} peers from disk")

        except json.JSONDecodeError as e:
            logger.error(f"VPN: Failed to parse peer DB JSON: {e}")
        except Exception as e:
            logger.error(f"VPN: Failed to load peer DB: {e}")

    def _save_peers(self) -> None:
        """Save peer configurations to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {}
            for uid, peer in self._peers.items():
                data[uid] = {
                    "user_id": peer.user_id,
                    "public_key": peer.public_key,
                    "preshared_key": peer.preshared_key,
                    "allowed_ips": peer.allowed_ips,
                    "endpoint": peer.endpoint,
                    "last_handshake": peer.last_handshake,
                }

            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"VPN: Saved {len(self._peers)} peers to disk")

        except Exception as e:
            logger.error(f"VPN: Failed to save peer DB: {e}")

    # =========================================================================
    # Key Generation
    # =========================================================================

    def generate_keypair(self) -> tuple[str, str]:
        """Generate a WireGuard private/public keypair."""
        try:
            private = subprocess.check_output(
                ["wg", "genkey"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            public = subprocess.check_output(
                ["wg", "pubkey"],
                input=private.encode(),
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            return private, public

        except FileNotFoundError:
            logger.warning("wg command not found, generating mock keys")
            # Mock for environments without wireguard-tools
            private = secrets.token_urlsafe(32)
            public = secrets.token_urlsafe(32)
            return private, public

        except subprocess.CalledProcessError as e:
            logger.error(f"Key generation failed: {e}")
            raise RuntimeError("Failed to generate WireGuard keys")

    def generate_preshared_key(self) -> str:
        """Generate a preshared key for additional security."""
        try:
            psk = subprocess.check_output(
                ["wg", "genpsk"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return psk

        except (FileNotFoundError, subprocess.CalledProcessError):
            # Mock for environments without wireguard-tools
            return secrets.token_urlsafe(32)

    # =========================================================================
    # Server Operations
    # =========================================================================

    def get_server_public_key(self) -> str:
        """Get the server's public key."""
        if self._server_public_key:
            return self._server_public_key

        # Try to read from config file
        pubkey_path = self.config_dir / "publickey"
        if pubkey_path.exists():
            self._server_public_key = pubkey_path.read_text().strip()
            return self._server_public_key

        # Try to read from interface
        try:
            output = subprocess.check_output(
                ["sudo", "wg", "show", self.interface, "public-key"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            self._server_public_key = output
            return output

        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("Could not get server public key, using placeholder")
            return "SERVER_PUBLIC_KEY_PLACEHOLDER"

    def add_peer(
        self,
        user_id: str,
        public_key: str,
        allowed_ips: str,
        preshared_key: Optional[str] = None,
    ) -> bool:
        """
        Add a peer to the WireGuard interface.

        Auto-saves to disk for persistence across restarts.

        Args:
            user_id: Unique identifier for the user
            public_key: Client's public key
            allowed_ips: IP(s) the client can use (CIDR notation)
            preshared_key: Optional PSK for additional security

        Returns:
            True if successful
        """
        logger.info(f"Adding VPN peer: {user_id} ({allowed_ips})")

        cmd = [
            "sudo", "wg", "set", self.interface,
            "peer", public_key,
            "allowed-ips", allowed_ips,
        ]

        if preshared_key:
            cmd.extend(["preshared-key", "/dev/stdin"])

        try:
            if preshared_key:
                subprocess.run(
                    cmd,
                    input=preshared_key.encode(),
                    check=True,
                    stderr=subprocess.PIPE,
                )
            else:
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

            # Track peer locally
            self._peers[user_id] = VPNPeer(
                user_id=user_id,
                public_key=public_key,
                preshared_key=preshared_key or "",
                allowed_ips=allowed_ips,
            )
            self._save_peers()  # AUTO SAVE

            return True

        except FileNotFoundError:
            logger.warning("wg command not found, simulating peer add")
            self._peers[user_id] = VPNPeer(
                user_id=user_id,
                public_key=public_key,
                preshared_key=preshared_key or "",
                allowed_ips=allowed_ips,
            )
            self._save_peers()  # AUTO SAVE
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add peer: {e.stderr.decode() if e.stderr else e}")
            return False

    def remove_peer(self, user_id: str) -> bool:
        """Remove a peer from the WireGuard interface. Auto-saves to disk."""
        peer = self._peers.get(user_id)
        if not peer:
            logger.warning(f"Peer not found: {user_id}")
            return False

        cmd = [
            "sudo", "wg", "set", self.interface,
            "peer", peer.public_key,
            "remove",
        ]

        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            del self._peers[user_id]
            self.ip_pool.release(user_id)
            self._save_peers()  # AUTO SAVE
            logger.info(f"Removed VPN peer: {user_id}")
            return True

        except FileNotFoundError:
            # Still remove from local state even if wg command not found
            logger.warning("wg command not found, removing peer from local state only")
            del self._peers[user_id]
            self.ip_pool.release(user_id)
            self._save_peers()  # AUTO SAVE
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove peer: {e}")
            return False

    # =========================================================================
    # Client Configuration Generation
    # =========================================================================

    def generate_client_config(
        self,
        user_id: str,
        client_ip: Optional[str] = None,
        dns: str = "10.10.0.1",
    ) -> str:
        """
        Generate a complete WireGuard client configuration.

        Args:
            user_id: Unique identifier for the user
            client_ip: IP to assign (auto-allocated if not provided)
            dns: DNS server for the client

        Returns:
            Complete .conf file contents
        """
        # Allocate IP if not provided
        if client_ip is None:
            client_ip = self.ip_pool.allocate(user_id)

        # Generate keys for client
        private_key, public_key = self.generate_keypair()
        preshared_key = self.generate_preshared_key()

        # Add peer to server
        self.add_peer(user_id, public_key, f"{client_ip}/32", preshared_key)

        # Get server public key
        server_pub = self.get_server_public_key()

        # Generate config
        config = f"""# WireGuard Configuration for {user_id}
# Generated by Ara Cathedral VPN Manager

[Interface]
PrivateKey = {private_key}
Address = {client_ip}/32
DNS = {dns}

[Peer]
PublicKey = {server_pub}
PresharedKey = {preshared_key}
Endpoint = {self.endpoint}
AllowedIPs = 10.10.0.0/24
PersistentKeepalive = 25
"""

        logger.info(f"Generated VPN config for {user_id} @ {client_ip}")
        return config

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_peer_status(self, user_id: str) -> Optional[Dict]:
        """Get status of a peer."""
        peer = self._peers.get(user_id)
        if not peer:
            return None

        # Try to get live status
        try:
            output = subprocess.check_output(
                ["sudo", "wg", "show", self.interface, "dump"],
                stderr=subprocess.DEVNULL,
            ).decode()

            for line in output.strip().split("\n")[1:]:  # Skip header
                parts = line.split("\t")
                if len(parts) >= 5 and parts[0] == peer.public_key:
                    return {
                        "user_id": user_id,
                        "public_key": parts[0],
                        "endpoint": parts[2] if parts[2] != "(none)" else None,
                        "allowed_ips": parts[3],
                        "latest_handshake": int(parts[4]) if parts[4] != "0" else None,
                        "transfer_rx": int(parts[5]) if len(parts) > 5 else 0,
                        "transfer_tx": int(parts[6]) if len(parts) > 6 else 0,
                    }

        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Return cached info if live status unavailable
        return {
            "user_id": user_id,
            "public_key": peer.public_key,
            "allowed_ips": peer.allowed_ips,
            "cached": True,
        }

    def list_peers(self) -> List[str]:
        """List all registered peer user IDs."""
        return list(self._peers.keys())

    def get_interface_status(self) -> Dict:
        """Get overall interface status."""
        try:
            output = subprocess.check_output(
                ["sudo", "wg", "show", self.interface],
                stderr=subprocess.DEVNULL,
            ).decode()

            return {
                "interface": self.interface,
                "status": "up",
                "peer_count": len(self._peers),
                "raw_output": output,
            }

        except (FileNotFoundError, subprocess.CalledProcessError):
            return {
                "interface": self.interface,
                "status": "unknown",
                "peer_count": len(self._peers),
                "note": "wg command unavailable",
            }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_vpn: Optional[VPNManager] = None


def get_vpn_manager() -> VPNManager:
    """Get the default VPN manager instance."""
    global _default_vpn
    if _default_vpn is None:
        _default_vpn = VPNManager()
    return _default_vpn
