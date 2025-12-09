"""
Auto Jailbreak Orchestrator

Intelligent jailbreak tool that attempts all methods in optimal sequence.

Decision Tree:
1. Try Method A (Network) first - non-invasive, fast
2. If Method A fails, guide user to Method B (Offline Shadow Edit)
3. Provide Method C (UART) as fallback for eMMC-based systems

This orchestrator prioritizes speed and success probability.
"""

from typing import Optional, Dict
from pathlib import Path
import sys

from .method_a_network import NetworkJailbreak


class AutoJailbreak:
    """Automatic jailbreak orchestrator for K10/P2 miners."""

    def __init__(self, target_ip: Optional[str] = None):
        """
        Initialize auto jailbreak tool.

        Args:
            target_ip: Target IP address (required for Method A)
        """
        self.target_ip = target_ip
        self.results = {
            'method_a_attempted': False,
            'method_a_success': False,
            'method_b_recommended': False,
            'method_c_recommended': False,
            'final_status': 'not_started',
        }

    def attempt_method_a(self) -> bool:
        """
        Attempt Method A: Network-based jailbreak.

        Returns:
            True if successful
        """
        if not self.target_ip:
            print("[!] Method A requires target IP address")
            return False

        print("=" * 70)
        print("Attempting Method A: Network-Based Jailbreak")
        print("=" * 70)
        print()

        self.results['method_a_attempted'] = True

        # Initialize Method A
        jailbreak = NetworkJailbreak(self.target_ip)

        # Run full attack
        attack_results = jailbreak.run_full_attack()

        if attack_results['success']:
            self.results['method_a_success'] = True
            self.results['final_status'] = 'success_method_a'

            print()
            print("=" * 70)
            print("SUCCESS! Root Access Obtained via Network")
            print("=" * 70)
            print()

            if attack_results['ssh_client']:
                print("[+] SSH client available")
                print(f"[+] Connect with: ssh root@{self.target_ip}")
                print()

            return True

        else:
            print()
            print("[!] Method A failed - all network-based attempts exhausted")
            print()

            return False

    def recommend_method_b(self):
        """Provide guidance for Method B: Offline Shadow Edit."""
        self.results['method_b_recommended'] = True
        self.results['final_status'] = 'method_b_recommended'

        print("=" * 70)
        print("Recommended: Method B (Offline Shadow Edit)")
        print("=" * 70)
        print()
        print("Method B is the most reliable Day 1 jailbreak for K10/P2.")
        print()
        print("Requirements:")
        print("  - Physical access to the miner")
        print("  - Linux workstation with SD card reader")
        print("  - Removable SD card (standard for K10/P2)")
        print()
        print("Steps:")
        print("  1. Power down the miner")
        print("  2. Remove the MicroSD card from the control board")
        print("  3. Insert SD card into Linux workstation")
        print("  4. Run the shadow editor:")
        print()
        print("     sudo python -m jailbreak.method_b_shadow /dev/sdb")
        print()
        print("  5. Follow the on-screen prompts")
        print("  6. Reinsert SD card and boot miner")
        print("  7. SSH with blank password: ssh root@<IP>")
        print()
        print("Success Rate: ~95% (requires SD card access)")
        print("=" * 70)
        print()

    def recommend_method_c(self):
        """Provide guidance for Method C: UART Injection."""
        self.results['method_c_recommended'] = True
        self.results['final_status'] = 'method_c_recommended'

        print("=" * 70)
        print("Fallback: Method C (UART Console Injection)")
        print("=" * 70)
        print()
        print("Method C is for devices with soldered eMMC (rare for K10/P2).")
        print()
        print("Requirements:")
        print("  - USB-to-TTL adapter (3.3V logic, e.g., FTDI, CH340)")
        print("  - Physical access to open the case")
        print("  - UART header on control board (GND, TX, RX)")
        print()
        print("Steps:")
        print("  1. Locate UART header (usually 3-4 pin connector)")
        print("  2. Connect adapter: GND→GND, TX→RX, RX→TX")
        print("  3. Run the UART injector:")
        print()
        print("     python -m jailbreak.method_c_uboot /dev/ttyUSB0")
        print()
        print("  4. Power cycle the miner")
        print("  5. Follow the interactive prompts")
        print()
        print("Success Rate: ~80% (requires UART access and U-Boot knowledge)")
        print("=" * 70)
        print()

    def run(self, interactive: bool = True) -> Dict:
        """
        Run complete auto jailbreak sequence.

        Args:
            interactive: Prompt user for decisions

        Returns:
            Results dictionary
        """
        print("=" * 70)
        print("K10/P2 Auto Jailbreak - Day 1 Toolkit")
        print("=" * 70)
        print()

        if self.target_ip:
            print(f"[+] Target IP: {self.target_ip}")
            print()

        # Decision: Attempt Method A if IP is available
        if self.target_ip:
            print("[*] Strategy: Attempt Method A (Network) first")
            print("[*] Method A is non-invasive and fastest")
            print()

            if interactive:
                response = input("Proceed with Method A? [Y/n]: ").strip().lower()
                if response and response != 'y':
                    print("[!] Skipping Method A")
                else:
                    if self.attempt_method_a():
                        return self.results

            else:
                if self.attempt_method_a():
                    return self.results

        else:
            print("[!] No target IP provided - skipping Method A")
            print()

        # Method A failed or skipped - recommend Method B
        print()
        print("[*] Recommending Method B: Offline Shadow Edit")
        print()

        if interactive:
            response = input("View Method B instructions? [Y/n]: ").strip().lower()
            if not response or response == 'y':
                print()
                self.recommend_method_b()

                response = input("Try Method C (UART) instead? [y/N]: ").strip().lower()
                if response == 'y':
                    print()
                    self.recommend_method_c()
        else:
            self.recommend_method_b()

        return self.results

    def generate_report(self) -> str:
        """
        Generate summary report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("Auto Jailbreak Report")
        report.append("=" * 70)
        report.append("")

        if self.target_ip:
            report.append(f"Target IP: {self.target_ip}")

        report.append(f"Final Status: {self.results['final_status']}")
        report.append("")

        report.append("Methods Attempted:")
        report.append(f"  - Method A (Network): {'✓ Success' if self.results['method_a_success'] else ('Attempted, Failed' if self.results['method_a_attempted'] else 'Not Attempted')}")
        report.append(f"  - Method B (Shadow Edit): {'Recommended' if self.results['method_b_recommended'] else 'Not Recommended'}")
        report.append(f"  - Method C (UART): {'Recommended' if self.results['method_c_recommended'] else 'Not Recommended'}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 Auto Jailbreak - Intelligent Method Selection',
        epilog='''
Example usage:
  # Auto-detect and guide
  python auto_jailbreak.py

  # With target IP (attempts Method A first)
  python auto_jailbreak.py --target 192.168.1.100

  # Non-interactive mode
  python auto_jailbreak.py --target 192.168.1.100 --non-interactive
        '''
    )

    parser.add_argument('--target', help='Target IP address')
    parser.add_argument('--non-interactive', action='store_true', help='Non-interactive mode')

    args = parser.parse_args()

    # Run auto jailbreak
    auto = AutoJailbreak(target_ip=args.target)
    results = auto.run(interactive=not args.non_interactive)

    # Print report
    print()
    print(auto.generate_report())

    # Exit code
    sys.exit(0 if results['final_status'].startswith('success') else 1)
