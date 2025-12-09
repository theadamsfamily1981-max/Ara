"""
Algorithm Switcher

Automate algorithm switching for K10/P2 miners.

Process:
1. Stop current mining process
2. Load new bitstream (.rbf)
3. Update miner configuration (pool, wallet)
4. Restart mining process
5. Monitor for successful startup

Supported algorithms:
- kHeavyHash (Kaspa - KAS)
- Blake3 (Alephium - ALPH, Ironfish - IRON)
- Sha512256D (Radiant - RXD)
- KarlsenHash (Karlsen - KLS)
"""

from pathlib import Path
from typing import Optional, Dict
import subprocess
import time
import json

from .bitstream_manager import BitstreamManager


class AlgorithmSwitcher:
    """Automate algorithm switching on K10/P2 miners."""

    # Algorithm metadata
    ALGORITHMS = {
        'kaspa': {
            'name': 'Kaspa (KAS)',
            'algorithm': 'kHeavyHash',
            'bitstream_pattern': '*kaspa*.rbf',
            'typical_hashrate': '30.00 GH/s',
            'typical_power': '1700W',
            'miner_binary': '/usr/bin/miner',
            'config_file': '/etc/miner.conf',
        },
        'alephium': {
            'name': 'Alephium (ALPH)',
            'algorithm': 'Blake3',
            'bitstream_pattern': '*aleph*.rbf',
            'typical_hashrate': '54.72 GH/s',
            'typical_power': '1973W',
            'miner_binary': '/usr/bin/miner',
            'config_file': '/etc/miner.conf',
        },
        'radiant': {
            'name': 'Radiant (RXD)',
            'algorithm': 'Sha512256D',
            'bitstream_pattern': '*radiant*.rbf',
            'typical_hashrate': '22.28 GH/s',
            'typical_power': '1236W',
            'miner_binary': '/usr/bin/miner',
            'config_file': '/etc/miner.conf',
        },
        'karlsen': {
            'name': 'Karlsen (KLS)',
            'algorithm': 'KarlsenHash',
            'bitstream_pattern': '*karlsen*.rbf',
            'typical_hashrate': '32.08 GH/s',
            'typical_power': '1484W',
            'miner_binary': '/usr/bin/miner',
            'config_file': '/etc/miner.conf',
        },
    }

    def __init__(self, remote_host: Optional[str] = None):
        """
        Initialize algorithm switcher.

        Args:
            remote_host: Optional SSH host for remote operation
        """
        self.remote_host = remote_host
        self.bitstream_mgr = BitstreamManager(remote_host=remote_host)

    def _run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run command locally or via SSH."""
        if self.remote_host:
            full_command = f"ssh {self.remote_host} '{command}'"
        else:
            full_command = command

        return subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )

    def get_current_algorithm(self) -> Optional[str]:
        """
        Detect currently running algorithm.

        Returns:
            Algorithm name or None
        """
        print("[*] Detecting current algorithm...")

        try:
            # Check running processes
            result = self._run_command("ps aux | grep -i miner | grep -v grep", check=False)

            if result.returncode == 0:
                output = result.stdout.lower()

                for algo_key, algo_info in self.ALGORITHMS.items():
                    if algo_key in output or algo_info['algorithm'].lower() in output:
                        print(f"[+] Current algorithm: {algo_info['name']}")
                        return algo_key

            print("[!] Could not detect current algorithm")
            return None

        except Exception as e:
            print(f"[x] Error detecting algorithm: {e}")
            return None

    def stop_miner(self) -> bool:
        """
        Stop current mining process.

        Returns:
            True if successful
        """
        print("[*] Stopping miner process...")

        try:
            # Try graceful stop via init script
            result = self._run_command("/etc/init.d/miner stop", check=False)

            if result.returncode == 0:
                print("[✓] Miner stopped gracefully")
                return True

            # Fallback: killall
            print("[!] Init script failed, trying killall...")
            self._run_command("killall -9 miner", check=False)

            time.sleep(2)

            # Verify stopped
            check = self._run_command("pgrep miner", check=False)

            if check.returncode != 0:  # No process found
                print("[✓] Miner process terminated")
                return True
            else:
                print("[x] Miner process still running")
                return False

        except Exception as e:
            print(f"[x] Error stopping miner: {e}")
            return False

    def start_miner(self) -> bool:
        """
        Start mining process.

        Returns:
            True if successful
        """
        print("[*] Starting miner process...")

        try:
            # Try init script
            result = self._run_command("/etc/init.d/miner start", check=False)

            if result.returncode == 0:
                print("[✓] Miner started via init script")
            else:
                # Fallback: direct execution
                print("[!] Init script failed, trying direct execution...")
                self._run_command("/usr/bin/miner &", check=False)

            time.sleep(5)

            # Verify running
            check = self._run_command("pgrep miner", check=False)

            if check.returncode == 0:  # Process found
                print("[✓] Miner process running")
                return True
            else:
                print("[x] Miner process did not start")
                return False

        except Exception as e:
            print(f"[x] Error starting miner: {e}")
            return False

    def switch_algorithm(
        self,
        target_algorithm: str,
        bitstream_path: Optional[Path] = None,
        pool_url: Optional[str] = None,
        wallet_address: Optional[str] = None
    ) -> bool:
        """
        Switch to a different mining algorithm.

        Args:
            target_algorithm: Algorithm name (kaspa, alephium, radiant, karlsen)
            bitstream_path: Custom bitstream path (optional, auto-detected if None)
            pool_url: Mining pool URL (optional)
            wallet_address: Wallet address (optional)

        Returns:
            True if successful
        """
        if target_algorithm not in self.ALGORITHMS:
            print(f"[x] Unknown algorithm: {target_algorithm}")
            print(f"[+] Supported: {', '.join(self.ALGORITHMS.keys())}")
            return False

        algo_info = self.ALGORITHMS[target_algorithm]

        print("=" * 70)
        print(f"Switching to {algo_info['name']}")
        print("=" * 70)
        print()
        print(f"Algorithm: {algo_info['algorithm']}")
        print(f"Typical Hashrate: {algo_info['typical_hashrate']}")
        print(f"Typical Power: {algo_info['typical_power']}")
        print()

        # Step 1: Stop miner
        if not self.stop_miner():
            print("[x] Failed to stop miner")
            return False

        print()

        # Step 2: Find bitstream
        if not bitstream_path:
            print(f"[*] Searching for bitstream matching: {algo_info['bitstream_pattern']}")

            available = self.bitstream_mgr.list_available_bitstreams()

            import fnmatch
            matches = [bs for bs in available if fnmatch.fnmatch(bs.name.lower(), algo_info['bitstream_pattern'])]

            if not matches:
                print(f"[x] No bitstream found matching pattern: {algo_info['bitstream_pattern']}")
                return False

            bitstream_path = matches[0]
            print(f"[+] Selected bitstream: {bitstream_path}")

        print()

        # Step 3: Load bitstream
        if not self.bitstream_mgr.load_bitstream(bitstream_path):
            print("[x] Failed to load bitstream")
            return False

        print()

        # Step 4: Update config (if pool/wallet provided)
        if pool_url or wallet_address:
            self._update_miner_config(algo_info['config_file'], pool_url, wallet_address)
            print()

        # Step 5: Start miner
        if not self.start_miner():
            print("[x] Failed to start miner")
            return False

        print()
        print("=" * 70)
        print(f"[✓] Successfully switched to {algo_info['name']}")
        print("=" * 70)
        print()
        print("Monitor logs with: tail -f /var/log/miner.log")
        print()

        return True

    def _update_miner_config(
        self,
        config_file: Path,
        pool_url: Optional[str] = None,
        wallet_address: Optional[str] = None
    ):
        """
        Update miner configuration file.

        Args:
            config_file: Path to config file
            pool_url: Mining pool URL
            wallet_address: Wallet address
        """
        print(f"[*] Updating miner config: {config_file}")

        try:
            # Read current config
            result = self._run_command(f"cat {config_file}", check=False)

            if result.returncode == 0:
                config_text = result.stdout
            else:
                print(f"[!] Could not read config file, creating new one")
                config_text = ""

            # Update fields
            if pool_url:
                # Simple text replacement (platform-specific logic may vary)
                config_text = f"POOL={pool_url}\n" + config_text
                print(f"[+] Set pool: {pool_url}")

            if wallet_address:
                config_text = f"WALLET={wallet_address}\n" + config_text
                print(f"[+] Set wallet: {wallet_address}")

            # Write back
            write_cmd = f"cat > {config_file} <<'EOF'\n{config_text}\nEOF"
            self._run_command(write_cmd)

            print(f"[✓] Config updated")

        except Exception as e:
            print(f"[x] Error updating config: {e}")

    def get_miner_status(self) -> Dict:
        """
        Get current miner status.

        Returns:
            Dictionary with status information
        """
        print("[*] Checking miner status...")

        status = {
            'running': False,
            'algorithm': None,
            'hashrate': None,
            'uptime': None,
        }

        try:
            # Check if process is running
            result = self._run_command("pgrep miner", check=False)
            status['running'] = (result.returncode == 0)

            if status['running']:
                # Try to get hashrate from logs
                log_check = self._run_command("tail -20 /var/log/miner.log | grep -i hashrate", check=False)

                if log_check.returncode == 0:
                    # Parse hashrate (platform-specific)
                    status['hashrate'] = log_check.stdout.strip()

                # Detect algorithm
                status['algorithm'] = self.get_current_algorithm()

            print(f"[+] Miner running: {status['running']}")
            print(f"[+] Algorithm: {status['algorithm']}")
            print(f"[+] Hashrate: {status['hashrate']}")

            return status

        except Exception as e:
            print(f"[x] Error getting status: {e}")
            return status


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 Algorithm Switcher',
        epilog='''
Example usage:
  # Switch to Alephium
  python algorithm_switcher.py --switch alephium

  # With pool configuration
  python algorithm_switcher.py --switch alephium \\
      --pool stratum+tcp://pool.alephium.com:20032 \\
      --wallet <YOUR_WALLET>

  # Remote operation
  python algorithm_switcher.py --remote root@192.168.1.100 --switch kaspa

  # Check status
  python algorithm_switcher.py --status
        '''
    )

    parser.add_argument('--remote', help='Remote SSH host (user@host)')
    parser.add_argument('--switch', choices=['kaspa', 'alephium', 'radiant', 'karlsen'],
                        help='Switch to algorithm')
    parser.add_argument('--bitstream', type=Path, help='Custom bitstream path')
    parser.add_argument('--pool', help='Mining pool URL')
    parser.add_argument('--wallet', help='Wallet address')
    parser.add_argument('--status', action='store_true', help='Show miner status')

    args = parser.parse_args()

    switcher = AlgorithmSwitcher(remote_host=args.remote)

    if args.status:
        status = switcher.get_miner_status()
        print()
        print(f"Status: {json.dumps(status, indent=2)}")

    elif args.switch:
        success = switcher.switch_algorithm(
            args.switch,
            bitstream_path=args.bitstream,
            pool_url=args.pool,
            wallet_address=args.wallet
        )
        exit(0 if success else 1)

    else:
        parser.print_help()
