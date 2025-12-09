"""
UDP Discovery Protocol

Emulate the K10 Management Tool discovery protocol.

The Windows management tool discovers miners via UDP broadcast.
This tool reverse-engineers and replicates that protocol for
programmatic farm management.

Protocol Analysis:
- Broadcast on UDP port 14235 (likely)
- Simple JSON or binary payload
- Miners respond with identity information
"""

import socket
import struct
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MinerInfo:
    """Information about a discovered miner."""
    ip_address: str
    mac_address: str
    hostname: str
    firmware_version: str
    current_algorithm: str
    uptime: int
    hashrate: Optional[float] = None


class DiscoveryProtocol:
    """UDP discovery protocol for K10/P2 miners."""

    # Common discovery ports
    DISCOVERY_PORTS = [14235, 8888, 9999, 5000, 3333]

    # Broadcast address
    BROADCAST_ADDR = '255.255.255.255'

    def __init__(self, discovery_port: int = 14235, timeout: int = 5):
        """
        Initialize discovery protocol.

        Args:
            discovery_port: UDP port for discovery (default: 14235)
            timeout: Response timeout in seconds
        """
        self.discovery_port = discovery_port
        self.timeout = timeout
        self.sock = None

    def create_discovery_packet(self, protocol_version: int = 1) -> bytes:
        """
        Create discovery request packet.

        Protocol formats to try:
        1. JSON: {"command": "discover", "version": 1}
        2. Binary: Magic bytes + version
        3. Simple string: "DISCOVER"

        Args:
            protocol_version: Protocol version number

        Returns:
            Packet bytes
        """
        # Try multiple formats
        formats = []

        # Format 1: JSON
        json_packet = json.dumps({
            'command': 'discover',
            'version': protocol_version,
            'timestamp': int(time.time())
        }).encode('utf-8')
        formats.append(json_packet)

        # Format 2: Binary (magic + version)
        # Magic: 0xK10D (K10 Discovery)
        binary_packet = struct.pack('>4sI', b'K10D', protocol_version)
        formats.append(binary_packet)

        # Format 3: Simple string
        string_packet = b'DISCOVER\n'
        formats.append(string_packet)

        # Return first format (can be modified to try all)
        return formats[0]

    def broadcast_discovery(self) -> List[Dict]:
        """
        Broadcast discovery request and collect responses.

        Returns:
            List of miner response dictionaries
        """
        print(f"[*] Broadcasting discovery on port {self.discovery_port}")

        discovered = []

        try:
            # Create UDP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.sock.settimeout(self.timeout)

            # Send discovery packet
            packet = self.create_discovery_packet()
            self.sock.sendto(packet, (self.BROADCAST_ADDR, self.discovery_port))

            print(f"[+] Discovery packet sent ({len(packet)} bytes)")
            print(f"[*] Listening for responses ({self.timeout}s timeout)...")

            # Collect responses
            start_time = time.time()

            while (time.time() - start_time) < self.timeout:
                try:
                    data, addr = self.sock.recvfrom(4096)
                    ip_address = addr[0]

                    print(f"[+] Response from {ip_address} ({len(data)} bytes)")

                    # Try to parse response
                    response = self._parse_response(data, ip_address)

                    if response:
                        discovered.append(response)

                except socket.timeout:
                    break
                except Exception as e:
                    print(f"[!] Error receiving response: {e}")

            print(f"[✓] Discovery complete: {len(discovered)} miner(s) found")

            return discovered

        except Exception as e:
            print(f"[x] Discovery error: {e}")
            return discovered

        finally:
            if self.sock:
                self.sock.close()

    def _parse_response(self, data: bytes, ip_address: str) -> Optional[Dict]:
        """
        Parse miner response packet.

        Args:
            data: Response packet bytes
            ip_address: Source IP address

        Returns:
            Parsed response dictionary or None
        """
        response = {
            'ip_address': ip_address,
            'raw_data': data.hex(),
        }

        try:
            # Try JSON parsing
            text = data.decode('utf-8')
            parsed = json.loads(text)

            response.update(parsed)
            print(f"    Parsed as JSON: {parsed}")

            return response

        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        try:
            # Try binary parsing
            # Assume format: Magic(4) + MAC(6) + Hostname(32) + Version(16) + Algorithm(16)
            if len(data) >= 74:
                magic, mac_bytes, hostname_bytes, version_bytes, algo_bytes = struct.unpack(
                    '>4s6s32s16s16s',
                    data[:74]
                )

                response['mac_address'] = ':'.join(f'{b:02x}' for b in mac_bytes)
                response['hostname'] = hostname_bytes.decode('utf-8', errors='ignore').strip('\x00')
                response['firmware_version'] = version_bytes.decode('utf-8', errors='ignore').strip('\x00')
                response['current_algorithm'] = algo_bytes.decode('utf-8', errors='ignore').strip('\x00')

                print(f"    Parsed as binary structure")
                return response

        except Exception:
            pass

        # Fallback: return raw data
        print(f"    Could not parse response (raw data saved)")
        return response

    def scan_network(self, network: str = "192.168.1.0/24") -> List[Dict]:
        """
        Scan network for K10/P2 miners using multiple discovery methods.

        Args:
            network: CIDR network range

        Returns:
            List of discovered miners
        """
        print(f"[*] Scanning network: {network}")

        discovered = []

        # Method 1: UDP broadcast discovery
        print("\n[*] Method 1: UDP Broadcast Discovery")
        broadcast_results = self.broadcast_discovery()
        discovered.extend(broadcast_results)

        # Method 2: Port scanning (SSH/HTTP on common IPs)
        print("\n[*] Method 2: Port Scanning")

        # Generate IP list from CIDR
        import ipaddress
        net = ipaddress.ip_network(network, strict=False)

        for ip in net.hosts():
            ip_str = str(ip)

            # Quick port check (SSH)
            if self._check_port(ip_str, 22, timeout=0.5):
                print(f"[+] Found SSH on {ip_str}")

                discovered.append({
                    'ip_address': ip_str,
                    'ssh_available': True,
                    'discovery_method': 'port_scan'
                })

        return discovered

    def _check_port(self, ip: str, port: int, timeout: float = 1.0) -> bool:
        """
        Check if a port is open.

        Args:
            ip: IP address
            port: Port number
            timeout: Connection timeout

        Returns:
            True if port is open
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()

            return result == 0

        except Exception:
            return False


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 UDP Discovery Protocol Emulator',
        epilog='''
Example usage:
  # Broadcast discovery on default port
  python discovery.py --broadcast

  # Custom port
  python discovery.py --broadcast --port 8888

  # Network scan
  python discovery.py --scan 192.168.1.0/24

  # Save results to JSON
  python discovery.py --broadcast --output discovered.json
        '''
    )

    parser.add_argument('--broadcast', action='store_true', help='Broadcast discovery')
    parser.add_argument('--scan', type=str, help='Scan CIDR network range')
    parser.add_argument('--port', type=int, default=14235, help='Discovery port (default: 14235)')
    parser.add_argument('--timeout', type=int, default=5, help='Timeout in seconds')
    parser.add_argument('--output', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    discovery = DiscoveryProtocol(discovery_port=args.port, timeout=args.timeout)

    results = []

    if args.broadcast:
        results = discovery.broadcast_discovery()

    elif args.scan:
        results = discovery.scan_network(args.scan)

    else:
        parser.print_help()
        exit(1)

    # Display results
    if results:
        print("\n" + "=" * 70)
        print("Discovered Miners")
        print("=" * 70)

        for i, miner in enumerate(results, 1):
            print(f"\n{i}. {miner['ip_address']}")
            for key, value in miner.items():
                if key != 'ip_address' and key != 'raw_data':
                    print(f"   {key}: {value}")

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[✓] Results saved to {args.output}")
