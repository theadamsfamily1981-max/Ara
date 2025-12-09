"""
Method B: Offline Shadow File Modification

Most reliable "Day 1" jailbreak method for K10/P2 miners.

Process:
1. Power down miner and extract SD card
2. Mount root partition on Linux workstation
3. Edit /etc/shadow to remove root password hash
4. Enable SSH root login in /etc/ssh/sshd_config
5. Disable password reset scripts in /etc/rc.local
6. Reinsert SD card and boot with blank root password
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import os


class ShadowEditor:
    """Offline SD card modification for privilege escalation."""

    def __init__(self, device: str, dry_run: bool = False):
        """
        Initialize shadow editor.

        Args:
            device: SD card device (e.g., /dev/sdb)
            dry_run: If True, only simulate changes without writing
        """
        self.device = device
        self.dry_run = dry_run
        self.mount_point = None
        self.boot_partition = None
        self.root_partition = None

    def identify_partitions(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify boot (FAT32) and root (ext3/ext4) partitions.

        Returns:
            Tuple of (boot_partition, root_partition)
        """
        print(f"[*] Analyzing partition table on {self.device}")

        try:
            # Run fdisk to list partitions
            result = subprocess.run(
                ['fdisk', '-l', self.device],
                capture_output=True,
                text=True,
                check=True
            )

            lines = result.stdout.split('\n')

            boot_part = None
            root_part = None

            for line in lines:
                if self.device in line:
                    # Parse partition line
                    # Example: /dev/sdb2  *     526336  15523839  14997504  83  Linux

                    if 'FAT' in line or '0b' in line or '0c' in line or 'W95' in line:
                        # FAT32 partition (boot)
                        parts = line.split()
                        boot_part = parts[0]
                        print(f"[+] Found boot partition: {boot_part} (FAT32)")

                    elif '83' in line or 'Linux' in line:
                        # Linux ext partition (root)
                        parts = line.split()
                        root_part = parts[0]
                        print(f"[+] Found root partition: {root_part} (ext3/ext4)")

            if not root_part:
                print("[!] Could not identify root partition automatically")
                print("[!] Listing all partitions:")
                print(result.stdout)

            self.boot_partition = boot_part
            self.root_partition = root_part

            return boot_part, root_part

        except subprocess.CalledProcessError as e:
            print(f"[x] Error running fdisk: {e}")
            return None, None
        except Exception as e:
            print(f"[x] Unexpected error: {e}")
            return None, None

    def mount_partition(self, partition: str, mount_point: Optional[Path] = None) -> Optional[Path]:
        """
        Mount a partition.

        Args:
            partition: Partition device (e.g., /dev/sdb3)
            mount_point: Custom mount point (default: temp directory)

        Returns:
            Path to mount point or None
        """
        if self.dry_run:
            print(f"[DRY-RUN] Would mount {partition}")
            return Path("/tmp/fake_mount")

        if not mount_point:
            mount_point = Path(tempfile.mkdtemp(prefix='k10_root_'))

        print(f"[*] Mounting {partition} to {mount_point}")

        try:
            # Ensure mount point exists
            mount_point.mkdir(parents=True, exist_ok=True)

            # Mount
            subprocess.run(
                ['mount', partition, str(mount_point)],
                check=True,
                capture_output=True
            )

            print(f"[✓] Mounted successfully")
            return mount_point

        except subprocess.CalledProcessError as e:
            print(f"[x] Mount failed: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"[x] Unexpected error: {e}")
            return None

    def unmount_partition(self, mount_point: Path) -> bool:
        """
        Unmount a partition.

        Args:
            mount_point: Path to mounted filesystem

        Returns:
            True if successful
        """
        if self.dry_run:
            print(f"[DRY-RUN] Would unmount {mount_point}")
            return True

        print(f"[*] Unmounting {mount_point}")

        try:
            subprocess.run(['umount', str(mount_point)], check=True, capture_output=True)
            print(f"[✓] Unmounted successfully")

            # Remove temp directory
            if mount_point.exists() and '/tmp/' in str(mount_point):
                mount_point.rmdir()

            return True

        except subprocess.CalledProcessError as e:
            print(f"[x] Unmount failed: {e.stderr.decode()}")
            return False

    def edit_shadow_file(self, root_mount: Path) -> bool:
        """
        Remove root password hash from /etc/shadow.

        Args:
            root_mount: Path to mounted root filesystem

        Returns:
            True if successful
        """
        shadow_file = root_mount / 'etc' / 'shadow'

        if not shadow_file.exists():
            print(f"[x] Shadow file not found: {shadow_file}")
            return False

        print(f"[*] Editing {shadow_file}")

        # Backup original
        backup_file = shadow_file.with_suffix('.shadow.backup')

        if self.dry_run:
            print(f"[DRY-RUN] Would backup to {backup_file}")
            print(f"[DRY-RUN] Would remove root password hash")
            return True

        try:
            shutil.copy2(shadow_file, backup_file)
            print(f"[+] Created backup: {backup_file}")

            # Read shadow file
            with open(shadow_file, 'r') as f:
                lines = f.readlines()

            # Modify root entry
            modified = False
            new_lines = []

            for line in lines:
                if line.startswith('root:'):
                    # Parse: root:$6$salt$hash:18000:0:99999:7:::
                    parts = line.split(':')

                    if len(parts) >= 2 and parts[1]:  # Has a password hash
                        print(f"[+] Found root password hash: {parts[1][:20]}...")
                        parts[1] = ''  # Remove hash (blank password)
                        line = ':'.join(parts)
                        modified = True
                        print(f"[✓] Removed root password hash")

                new_lines.append(line)

            if not modified:
                print("[!] Root entry not found or already has blank password")
                return False

            # Write modified shadow file
            with open(shadow_file, 'w') as f:
                f.writelines(new_lines)

            print(f"[✓] Shadow file updated")
            return True

        except Exception as e:
            print(f"[x] Error editing shadow file: {e}")
            # Restore backup
            if backup_file.exists():
                shutil.copy2(backup_file, shadow_file)
                print(f"[+] Restored from backup")
            return False

    def enable_ssh_root_login(self, root_mount: Path) -> bool:
        """
        Enable root login in /etc/ssh/sshd_config.

        Args:
            root_mount: Path to mounted root filesystem

        Returns:
            True if successful
        """
        sshd_config = root_mount / 'etc' / 'ssh' / 'sshd_config'

        if not sshd_config.exists():
            print(f"[!] SSH config not found: {sshd_config} (may not be an issue)")
            return True  # Not critical

        print(f"[*] Configuring SSH root login in {sshd_config}")

        if self.dry_run:
            print(f"[DRY-RUN] Would enable PermitRootLogin yes")
            return True

        try:
            # Backup
            backup_file = sshd_config.with_suffix('.sshd_config.backup')
            shutil.copy2(sshd_config, backup_file)

            # Read config
            with open(sshd_config, 'r') as f:
                lines = f.readlines()

            # Modify
            new_lines = []
            modified = False

            for line in lines:
                stripped = line.strip()

                if stripped.startswith('#PermitRootLogin') or stripped.startswith('PermitRootLogin'):
                    new_lines.append('PermitRootLogin yes\n')
                    modified = True
                    print(f"[+] Set PermitRootLogin yes")
                else:
                    new_lines.append(line)

            # If no existing entry, append
            if not modified:
                new_lines.append('\n# K10 Day 1 Jailbreak\n')
                new_lines.append('PermitRootLogin yes\n')
                print(f"[+] Added PermitRootLogin yes")

            # Write
            with open(sshd_config, 'w') as f:
                f.writelines(new_lines)

            print(f"[✓] SSH config updated")
            return True

        except Exception as e:
            print(f"[x] Error editing SSH config: {e}")
            return False

    def disable_password_reset_scripts(self, root_mount: Path) -> bool:
        """
        Disable any scripts in /etc/rc.local that might reset passwords.

        Args:
            root_mount: Path to mounted root filesystem

        Returns:
            True if successful
        """
        rc_local = root_mount / 'etc' / 'rc.local'

        if not rc_local.exists():
            print(f"[!] rc.local not found (may not exist)")
            return True

        print(f"[*] Checking {rc_local} for password reset scripts")

        if self.dry_run:
            print(f"[DRY-RUN] Would analyze rc.local")
            return True

        try:
            with open(rc_local, 'r') as f:
                content = f.read()

            # Check for suspicious commands
            suspicious = ['passwd', 'chpasswd', 'shadow', 'usermod']

            if any(cmd in content for cmd in suspicious):
                print(f"[!] Suspicious commands found in rc.local:")
                print(content)
                print(f"[!] Consider commenting out these lines")

                # Backup
                backup_file = rc_local.with_suffix('.rc_local.backup')
                shutil.copy2(rc_local, backup_file)
                print(f"[+] Created backup: {backup_file}")

            else:
                print(f"[✓] No password reset scripts detected")

            return True

        except Exception as e:
            print(f"[x] Error checking rc.local: {e}")
            return False

    def install_ssh_key(self, root_mount: Path, public_key: str) -> bool:
        """
        Install SSH public key for passwordless authentication.

        Args:
            root_mount: Path to mounted root filesystem
            public_key: SSH public key string

        Returns:
            True if successful
        """
        ssh_dir = root_mount / 'root' / '.ssh'
        authorized_keys = ssh_dir / 'authorized_keys'

        print(f"[*] Installing SSH key to {authorized_keys}")

        if self.dry_run:
            print(f"[DRY-RUN] Would install SSH key")
            return True

        try:
            # Create .ssh directory if it doesn't exist
            ssh_dir.mkdir(parents=True, exist_ok=True)
            ssh_dir.chmod(0o700)

            # Append key
            with open(authorized_keys, 'a') as f:
                f.write(f"\n{public_key}\n")

            authorized_keys.chmod(0o600)

            print(f"[✓] SSH key installed")
            return True

        except Exception as e:
            print(f"[x] Error installing SSH key: {e}")
            return False

    def run_full_jailbreak(self, ssh_pubkey: Optional[str] = None) -> bool:
        """
        Execute complete offline jailbreak sequence.

        Args:
            ssh_pubkey: Optional SSH public key to install

        Returns:
            True if successful
        """
        print("=" * 70)
        print(f"K10 Offline Shadow Edit Jailbreak (Method B)")
        print(f"Target Device: {self.device}")
        print("=" * 70)
        print()

        if self.dry_run:
            print("[!] DRY-RUN MODE: No changes will be written")
            print()

        # Step 1: Identify partitions
        boot_part, root_part = self.identify_partitions()

        if not root_part:
            print("[x] Cannot proceed without root partition")
            return False

        print()

        # Step 2: Mount root partition
        mount_point = self.mount_partition(root_part)

        if not mount_point:
            print("[x] Failed to mount root partition")
            return False

        self.mount_point = mount_point

        try:
            print()

            # Step 3: Edit shadow file
            if not self.edit_shadow_file(mount_point):
                print("[x] Failed to edit shadow file")
                return False

            print()

            # Step 4: Enable SSH root login
            self.enable_ssh_root_login(mount_point)

            print()

            # Step 5: Check for password reset scripts
            self.disable_password_reset_scripts(mount_point)

            print()

            # Step 6: Install SSH key (optional)
            if ssh_pubkey:
                self.install_ssh_key(mount_point, ssh_pubkey)
                print()

            print("=" * 70)
            print("[✓] JAILBREAK COMPLETE")
            print("=" * 70)
            print()
            print("Next steps:")
            print("1. Unmount the SD card")
            print("2. Reinsert into K10/P2 miner")
            print("3. Boot the miner")
            print("4. SSH login: ssh root@<IP>  (no password)")
            print()

            return True

        finally:
            # Always unmount
            if mount_point and not self.dry_run:
                print()
                self.unmount_partition(mount_point)


# Command-line interface
if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='K10/P2 Offline Shadow Edit Jailbreak (Method B)',
        epilog='''
Example usage:
  # Dry-run (safe, no changes)
  sudo python method_b_shadow.py /dev/sdb --dry-run

  # Execute jailbreak
  sudo python method_b_shadow.py /dev/sdb

  # With SSH key installation
  sudo python method_b_shadow.py /dev/sdb --ssh-key ~/.ssh/id_rsa.pub
        '''
    )

    parser.add_argument('device', help='SD card device (e.g., /dev/sdb)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without writing')
    parser.add_argument('--ssh-key', type=Path, help='SSH public key file to install')

    args = parser.parse_args()

    # Check root privileges
    if os.geteuid() != 0 and not args.dry_run:
        print("[x] This script requires root privileges")
        print("[+] Run with: sudo python method_b_shadow.py ...")
        sys.exit(1)

    # Load SSH key if provided
    ssh_pubkey = None
    if args.ssh_key:
        if args.ssh_key.exists():
            ssh_pubkey = args.ssh_key.read_text().strip()
            print(f"[+] Loaded SSH public key from {args.ssh_key}")
        else:
            print(f"[x] SSH key file not found: {args.ssh_key}")
            sys.exit(1)

    # Execute jailbreak
    editor = ShadowEditor(args.device, dry_run=args.dry_run)
    success = editor.run_full_jailbreak(ssh_pubkey=ssh_pubkey)

    sys.exit(0 if success else 1)
