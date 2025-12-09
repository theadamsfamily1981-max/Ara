"""
ISO Extractor

Extract FPGA bitstreams from vendor-provided ISO files.

The K10/P2 firmware distribution includes "ALGO Files" as ISO images.
These ISOs contain:
- .rbf bitstreams
- Configuration files
- Update scripts

This tool mounts ISOs and extracts bitstreams for manual deployment.
"""

from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import tempfile
import shutil
import json


class ISOExtractor:
    """Extract bitstreams from K10/P2 ISO files."""

    def __init__(self):
        """Initialize ISO extractor."""
        self.mount_point = None

    def mount_iso(self, iso_path: Path, mount_point: Optional[Path] = None) -> Optional[Path]:
        """
        Mount ISO file.

        Args:
            iso_path: Path to .iso file
            mount_point: Custom mount point (default: temp directory)

        Returns:
            Path to mount point or None
        """
        if not iso_path.exists():
            print(f"[x] ISO file not found: {iso_path}")
            return None

        if not mount_point:
            mount_point = Path(tempfile.mkdtemp(prefix='k10_iso_'))

        print(f"[*] Mounting {iso_path} to {mount_point}")

        try:
            # Create mount point
            mount_point.mkdir(parents=True, exist_ok=True)

            # Mount ISO
            subprocess.run(
                ['mount', '-o', 'loop,ro', str(iso_path), str(mount_point)],
                check=True,
                capture_output=True
            )

            print(f"[✓] Mounted successfully")
            self.mount_point = mount_point

            return mount_point

        except subprocess.CalledProcessError as e:
            print(f"[x] Mount failed: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"[x] Unexpected error: {e}")
            return None

    def unmount_iso(self, mount_point: Optional[Path] = None) -> bool:
        """
        Unmount ISO.

        Args:
            mount_point: Path to mounted ISO (default: self.mount_point)

        Returns:
            True if successful
        """
        if not mount_point:
            mount_point = self.mount_point

        if not mount_point:
            print("[!] No mount point specified")
            return False

        print(f"[*] Unmounting {mount_point}")

        try:
            subprocess.run(['umount', str(mount_point)], check=True, capture_output=True)
            print(f"[✓] Unmounted successfully")

            # Remove temp directory
            if '/tmp/' in str(mount_point):
                mount_point.rmdir()

            self.mount_point = None
            return True

        except subprocess.CalledProcessError as e:
            print(f"[x] Unmount failed: {e.stderr.decode()}")
            return False

    def analyze_iso_contents(self, mount_point: Path) -> Dict:
        """
        Analyze contents of mounted ISO.

        Args:
            mount_point: Path to mounted ISO

        Returns:
            Dictionary with analysis results
        """
        print(f"[*] Analyzing ISO contents at {mount_point}")

        analysis = {
            'bitstreams': [],
            'config_files': [],
            'scripts': [],
            'other_files': [],
        }

        try:
            # Find all files
            for item in mount_point.rglob('*'):
                if item.is_file():
                    ext = item.suffix.lower()
                    rel_path = item.relative_to(mount_point)

                    # Classify file
                    if ext == '.rbf':
                        analysis['bitstreams'].append({
                            'path': str(rel_path),
                            'size_mb': item.stat().st_size / (1024 * 1024),
                            'full_path': str(item)
                        })

                    elif ext in ['.conf', '.cfg', '.json', '.ini']:
                        analysis['config_files'].append(str(rel_path))

                    elif ext in ['.sh', '.py']:
                        analysis['scripts'].append(str(rel_path))

                    else:
                        analysis['other_files'].append(str(rel_path))

            # Print summary
            print(f"[+] Found {len(analysis['bitstreams'])} bitstream(s)")
            for bs in analysis['bitstreams']:
                print(f"    - {bs['path']} ({bs['size_mb']:.2f} MB)")

            print(f"[+] Found {len(analysis['config_files'])} config file(s)")
            print(f"[+] Found {len(analysis['scripts'])} script(s)")

            return analysis

        except Exception as e:
            print(f"[x] Error analyzing ISO: {e}")
            return analysis

    def extract_bitstreams(self, mount_point: Path, output_dir: Path) -> List[Path]:
        """
        Extract all bitstreams from mounted ISO.

        Args:
            mount_point: Path to mounted ISO
            output_dir: Destination directory

        Returns:
            List of extracted file paths
        """
        print(f"[*] Extracting bitstreams to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = []

        try:
            # Find all .rbf files
            rbf_files = list(mount_point.rglob('*.rbf'))

            for rbf_file in rbf_files:
                # Copy to output directory
                dest_path = output_dir / rbf_file.name

                print(f"[+] Copying {rbf_file.name}...")
                shutil.copy2(rbf_file, dest_path)

                extracted.append(dest_path)

            print(f"[✓] Extracted {len(extracted)} bitstream(s)")

            return extracted

        except Exception as e:
            print(f"[x] Extraction error: {e}")
            return extracted

    def extract_all_files(self, mount_point: Path, output_dir: Path) -> bool:
        """
        Extract all files from ISO (full extraction).

        Args:
            mount_point: Path to mounted ISO
            output_dir: Destination directory

        Returns:
            True if successful
        """
        print(f"[*] Extracting all files to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy entire directory tree
            for item in mount_point.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(mount_point)
                    dest_path = output_dir / rel_path

                    # Create parent directories
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(item, dest_path)

            print(f"[✓] Extraction complete")
            return True

        except Exception as e:
            print(f"[x] Extraction error: {e}")
            return False

    def generate_extraction_report(self, analysis: Dict, output_path: Path):
        """
        Generate JSON report of extracted files.

        Args:
            analysis: Analysis dictionary from analyze_iso_contents()
            output_path: Path to save report
        """
        print(f"[*] Generating extraction report: {output_path}")

        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)

            print(f"[✓] Report saved")

        except Exception as e:
            print(f"[x] Error saving report: {e}")

    def process_iso(
        self,
        iso_path: Path,
        output_dir: Path,
        bitstreams_only: bool = True,
        generate_report: bool = True
    ) -> bool:
        """
        Complete ISO processing workflow.

        Args:
            iso_path: Path to .iso file
            output_dir: Output directory
            bitstreams_only: Extract only .rbf files (default: True)
            generate_report: Generate JSON report (default: True)

        Returns:
            True if successful
        """
        print("=" * 70)
        print(f"Processing ISO: {iso_path.name}")
        print("=" * 70)
        print()

        # Step 1: Mount ISO
        mount_point = self.mount_iso(iso_path)

        if not mount_point:
            print("[x] Failed to mount ISO")
            return False

        try:
            print()

            # Step 2: Analyze contents
            analysis = self.analyze_iso_contents(mount_point)

            print()

            # Step 3: Extract files
            if bitstreams_only:
                extracted = self.extract_bitstreams(mount_point, output_dir)
            else:
                self.extract_all_files(mount_point, output_dir)

            print()

            # Step 4: Generate report
            if generate_report:
                report_path = output_dir / f"{iso_path.stem}_extraction_report.json"
                self.generate_extraction_report(analysis, report_path)

            print()
            print("=" * 70)
            print("[✓] ISO Processing Complete")
            print("=" * 70)
            print()
            print(f"Output directory: {output_dir}")
            print()

            return True

        finally:
            # Always unmount
            self.unmount_iso(mount_point)


# Command-line interface
if __name__ == '__main__':
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description='K10/P2 ISO Bitstream Extractor',
        epilog='''
Example usage:
  # Extract bitstreams only
  sudo python iso_extractor.py algo_alephium.iso -o extracted/alephium

  # Extract all files
  sudo python iso_extractor.py algo_kaspa.iso -o extracted/kaspa --all-files

  # Batch process multiple ISOs
  for iso in *.iso; do
      sudo python iso_extractor.py "$iso" -o "extracted/$(basename "$iso" .iso)"
  done
        '''
    )

    parser.add_argument('iso_file', type=Path, help='ISO file to extract')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--all-files', action='store_true', help='Extract all files (not just bitstreams)')
    parser.add_argument('--no-report', action='store_true', help='Skip JSON report generation')

    args = parser.parse_args()

    # Check root privileges (required for mounting)
    if os.geteuid() != 0:
        print("[x] This script requires root privileges for mounting ISOs")
        print("[+] Run with: sudo python iso_extractor.py ...")
        sys.exit(1)

    # Check ISO exists
    if not args.iso_file.exists():
        print(f"[x] ISO file not found: {args.iso_file}")
        sys.exit(1)

    # Execute extraction
    extractor = ISOExtractor()
    success = extractor.process_iso(
        args.iso_file,
        args.output,
        bitstreams_only=not args.all_files,
        generate_report=not args.no_report
    )

    sys.exit(0 if success else 1)
