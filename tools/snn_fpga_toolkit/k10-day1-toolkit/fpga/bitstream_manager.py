"""
FPGA Bitstream Manager

Interface to Linux FPGA Manager subsystem for Stratix 10 SoC.

Sysfs paths:
- /sys/class/fpga_manager/fpga0/state
- /sys/class/fpga_manager/fpga0/firmware
- /sys/class/fpga_manager/fpga0/name

Bitstream locations:
- /lib/firmware/ (standard Linux location)
- /opt/bitstreams/ (custom K10 location)
"""

from pathlib import Path
from typing import Optional, List
import subprocess
import time
import hashlib


class BitstreamManager:
    """Manage FPGA bitstream loading on K10/P2."""

    # Standard Linux paths
    FPGA_MANAGER_PATH = Path("/sys/class/fpga_manager/fpga0")
    FIRMWARE_DIR = Path("/lib/firmware")

    # Custom K10 paths (common locations)
    CUSTOM_FIRMWARE_DIRS = [
        Path("/opt/bitstreams"),
        Path("/opt/firmware"),
        Path("/usr/share/fpga"),
        Path("/root/bitstreams"),
    ]

    def __init__(self, remote_host: Optional[str] = None):
        """
        Initialize bitstream manager.

        Args:
            remote_host: Optional SSH host for remote operation (user@host)
        """
        self.remote_host = remote_host
        self.fpga_manager = self.FPGA_MANAGER_PATH
        self.firmware_dir = self.FIRMWARE_DIR

    def _run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """
        Run command locally or via SSH.

        Args:
            command: Shell command
            check: Raise exception on non-zero exit

        Returns:
            CompletedProcess result
        """
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

    def get_fpga_state(self) -> Optional[str]:
        """
        Get current FPGA state from sysfs.

        States:
        - operating: FPGA is configured and running
        - unknown: State cannot be determined
        - power_off: FPGA is powered off
        - programming: FPGA is being programmed

        Returns:
            State string or None if unavailable
        """
        state_file = self.fpga_manager / "state"

        try:
            result = self._run_command(f"cat {state_file}")
            state = result.stdout.strip()
            print(f"[+] FPGA state: {state}")
            return state

        except subprocess.CalledProcessError as e:
            print(f"[x] Error reading FPGA state: {e.stderr}")
            return None

    def get_fpga_name(self) -> Optional[str]:
        """
        Get FPGA manager name.

        Returns:
            Name string or None
        """
        name_file = self.fpga_manager / "name"

        try:
            result = self._run_command(f"cat {name_file}")
            name = result.stdout.strip()
            print(f"[+] FPGA manager: {name}")
            return name

        except subprocess.CalledProcessError as e:
            print(f"[x] Error reading FPGA name: {e.stderr}")
            return None

    def list_available_bitstreams(self) -> List[Path]:
        """
        List all .rbf files in firmware directories.

        Returns:
            List of bitstream file paths
        """
        bitstreams = []

        # Check standard firmware dir
        try:
            result = self._run_command(f"find {self.firmware_dir} -name '*.rbf' 2>/dev/null", check=False)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        bitstreams.append(Path(line))

        except Exception as e:
            pass

        # Check custom dirs
        for custom_dir in self.CUSTOM_FIRMWARE_DIRS:
            try:
                result = self._run_command(f"find {custom_dir} -name '*.rbf' 2>/dev/null", check=False)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            bitstreams.append(Path(line))

            except Exception as e:
                pass

        print(f"[+] Found {len(bitstreams)} bitstream files")

        for bs in bitstreams:
            print(f"    - {bs}")

        return bitstreams

    def load_bitstream(self, bitstream_path: Path, timeout: int = 30) -> bool:
        """
        Load bitstream via FPGA manager sysfs interface.

        Args:
            bitstream_path: Path to .rbf file (relative to /lib/firmware)
            timeout: Maximum wait time in seconds

        Returns:
            True if successful
        """
        # Extract filename (must be relative to /lib/firmware)
        if str(bitstream_path).startswith(str(self.firmware_dir)):
            firmware_name = str(bitstream_path.relative_to(self.firmware_dir))
        else:
            firmware_name = bitstream_path.name

        print(f"[*] Loading bitstream: {firmware_name}")

        firmware_file = self.fpga_manager / "firmware"

        try:
            # Write firmware name to sysfs
            command = f"echo '{firmware_name}' > {firmware_file}"
            self._run_command(command)

            print(f"[+] Bitstream load initiated")

            # Poll state until operating or timeout
            start_time = time.time()

            while (time.time() - start_time) < timeout:
                state = self.get_fpga_state()

                if state == "operating":
                    print(f"[✓] Bitstream loaded successfully ({time.time() - start_time:.1f}s)")
                    return True

                elif state == "programming":
                    print(f"[*] Programming... ({time.time() - start_time:.1f}s)")
                    time.sleep(1)

                else:
                    print(f"[!] Unexpected state: {state}")
                    time.sleep(1)

            print(f"[x] Timeout waiting for bitstream load")
            return False

        except subprocess.CalledProcessError as e:
            print(f"[x] Error loading bitstream: {e.stderr}")
            return False

    def copy_bitstream_to_firmware(self, source: Path, dest_name: Optional[str] = None) -> bool:
        """
        Copy bitstream file to /lib/firmware.

        Args:
            source: Source .rbf file path
            dest_name: Destination filename (default: same as source)

        Returns:
            True if successful
        """
        if not dest_name:
            dest_name = source.name

        dest_path = self.firmware_dir / dest_name

        print(f"[*] Copying {source} → {dest_path}")

        try:
            if self.remote_host:
                # Use scp for remote copy
                command = f"scp {source} {self.remote_host}:{dest_path}"
                subprocess.run(command, shell=True, check=True, capture_output=True)
            else:
                # Local copy
                import shutil
                shutil.copy2(source, dest_path)

            print(f"[✓] Bitstream copied successfully")
            return True

        except Exception as e:
            print(f"[x] Error copying bitstream: {e}")
            return False

    def verify_bitstream(self, bitstream_path: Path) -> dict:
        """
        Verify bitstream file integrity and format.

        Args:
            bitstream_path: Path to .rbf file

        Returns:
            Dictionary with verification results
        """
        print(f"[*] Verifying bitstream: {bitstream_path}")

        results = {
            'exists': False,
            'size_mb': 0.0,
            'md5': None,
            'format': 'unknown',
        }

        try:
            # Check existence
            if self.remote_host:
                check = self._run_command(f"test -f {bitstream_path} && echo 'exists'", check=False)
                results['exists'] = 'exists' in check.stdout
            else:
                results['exists'] = bitstream_path.exists()

            if not results['exists']:
                print(f"[x] File does not exist")
                return results

            # Get size
            if self.remote_host:
                size_result = self._run_command(f"stat -c %s {bitstream_path}")
                size_bytes = int(size_result.stdout.strip())
            else:
                size_bytes = bitstream_path.stat().st_size

            results['size_mb'] = size_bytes / (1024 * 1024)
            print(f"[+] Size: {results['size_mb']:.2f} MB")

            # Calculate MD5
            if self.remote_host:
                md5_result = self._run_command(f"md5sum {bitstream_path}")
                results['md5'] = md5_result.stdout.split()[0]
            else:
                with open(bitstream_path, 'rb') as f:
                    results['md5'] = hashlib.md5(f.read()).hexdigest()

            print(f"[+] MD5: {results['md5']}")

            # Detect format (basic heuristic)
            if results['size_mb'] > 10 and results['size_mb'] < 150:
                results['format'] = 'rbf_stratix10_likely'
                print(f"[+] Format: Likely Stratix 10 RBF")
            else:
                results['format'] = 'rbf_unknown_size'
                print(f"[!] Format: Unknown (unusual size)")

            return results

        except Exception as e:
            print(f"[x] Verification error: {e}")
            return results

    def backup_current_bitstream(self, backup_dir: Path) -> Optional[Path]:
        """
        Backup currently loaded bitstream (if identifiable).

        Args:
            backup_dir: Directory to save backup

        Returns:
            Path to backup file or None
        """
        print(f"[*] Backing up current bitstream to {backup_dir}")

        # Try to identify current bitstream from miner logs or config
        # This is platform-specific and may require custom logic

        print(f"[!] Automatic backup not yet implemented")
        print(f"[!] Manually identify and copy current .rbf file")

        return None


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 FPGA Bitstream Manager'
    )

    parser.add_argument('--remote', help='Remote SSH host (user@host)')
    parser.add_argument('--state', action='store_true', help='Show FPGA state')
    parser.add_argument('--list', action='store_true', help='List available bitstreams')
    parser.add_argument('--load', type=Path, help='Load bitstream (.rbf file)')
    parser.add_argument('--copy', type=Path, help='Copy bitstream to firmware dir')
    parser.add_argument('--verify', type=Path, help='Verify bitstream file')

    args = parser.parse_args()

    manager = BitstreamManager(remote_host=args.remote)

    if args.state:
        state = manager.get_fpga_state()
        name = manager.get_fpga_name()

    elif args.list:
        bitstreams = manager.list_available_bitstreams()

    elif args.load:
        success = manager.load_bitstream(args.load)
        exit(0 if success else 1)

    elif args.copy:
        success = manager.copy_bitstream_to_firmware(args.copy)
        exit(0 if success else 1)

    elif args.verify:
        results = manager.verify_bitstream(args.verify)
        print()
        print(f"Verification Results: {results}")

    else:
        parser.print_help()
