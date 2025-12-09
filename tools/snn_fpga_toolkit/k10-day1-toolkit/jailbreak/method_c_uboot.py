"""
Method C: U-Boot Environment Injection via UART Console

Hardware-based jailbreak for K10/P2 when SD card is inaccessible.

Requirements:
- USB-to-TTL adapter (3.3V logic)
- Physical access to UART header on control board
- Serial terminal software (minicom, screen, or Python serial)

Process:
1. Connect UART adapter (GND, RX→TX, TX→RX)
2. Open serial console (115200 baud, 8N1)
3. Interrupt U-Boot autoboot sequence
4. Inject kernel parameter: init=/bin/sh
5. Boot to root shell, remount RW, set password
"""

import serial
import time
from typing import Optional, List
from pathlib import Path
import sys


class UBootInjector:
    """UART-based U-Boot injection for privilege escalation."""

    def __init__(self, serial_port: str, baudrate: int = 115200):
        """
        Initialize U-Boot injector.

        Args:
            serial_port: Serial device (e.g., /dev/ttyUSB0, COM3)
            baudrate: Baud rate (default: 115200)
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None
        self.timeout = 1.0

    def connect(self) -> bool:
        """
        Open serial connection.

        Returns:
            True if successful
        """
        print(f"[*] Connecting to {self.serial_port} at {self.baudrate} baud")

        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )

            print(f"[✓] Serial connection established")
            return True

        except serial.SerialException as e:
            print(f"[x] Failed to open serial port: {e}")
            return False
        except Exception as e:
            print(f"[x] Unexpected error: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[✓] Serial connection closed")

    def send_command(self, command: str, wait_time: float = 0.5) -> str:
        """
        Send command to serial console.

        Args:
            command: Command string
            wait_time: Time to wait for response

        Returns:
            Response text
        """
        if not self.ser or not self.ser.is_open:
            print(f"[x] Serial port not open")
            return ""

        try:
            # Send command
            self.ser.write(f"{command}\n".encode('utf-8'))
            self.ser.flush()

            # Wait for response
            time.sleep(wait_time)

            # Read available data
            response = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')

            return response

        except Exception as e:
            print(f"[x] Error sending command: {e}")
            return ""

    def wait_for_pattern(self, pattern: str, timeout: float = 30.0) -> Optional[str]:
        """
        Wait for a specific pattern in serial output.

        Args:
            pattern: Text pattern to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Full output up to pattern, or None if timeout
        """
        if not self.ser or not self.ser.is_open:
            return None

        print(f"[*] Waiting for pattern: '{pattern}' (timeout: {timeout}s)")

        buffer = ""
        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
                    sys.stdout.write(chunk)  # Echo to console
                    sys.stdout.flush()

                    if pattern in buffer:
                        print(f"\n[✓] Pattern detected: '{pattern}'")
                        return buffer

                time.sleep(0.1)

            print(f"\n[x] Timeout waiting for pattern: '{pattern}'")
            return None

        except Exception as e:
            print(f"\n[x] Error waiting for pattern: {e}")
            return None

    def interrupt_autoboot(self) -> bool:
        """
        Interrupt U-Boot autoboot sequence.

        User must manually power cycle the device while this is running.

        Returns:
            True if U-Boot prompt detected
        """
        print("=" * 70)
        print("U-Boot Autoboot Interception")
        print("=" * 70)
        print()
        print("[!] ACTION REQUIRED:")
        print("    1. Ensure serial connection is established")
        print("    2. Power cycle the K10/P2 miner NOW")
        print("    3. Watch for 'Hit any key to stop autoboot' message")
        print()
        print("[*] Listening for U-Boot messages...")
        print()

        # Wait for U-Boot banner
        output = self.wait_for_pattern("U-Boot", timeout=60.0)

        if not output:
            print("[x] Did not detect U-Boot. Check connections and power cycle.")
            return False

        # Spam Enter to interrupt autoboot
        print("\n[*] Sending interrupt signals...")

        for i in range(10):
            self.ser.write(b"\n")
            time.sleep(0.1)

        time.sleep(1.0)

        # Check for U-Boot prompt
        response = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
        print(response)

        # Look for prompt patterns
        prompts = ['SOCFPGA #', 'U-Boot>', '=>', '#']

        if any(prompt in response for prompt in prompts):
            print(f"[✓] U-Boot command prompt detected!")
            return True
        else:
            print(f"[x] U-Boot prompt not detected. Response:")
            print(response)
            return False

    def inject_init_shell(self) -> bool:
        """
        Inject init=/bin/sh into kernel boot arguments.

        Returns:
            True if successful
        """
        print()
        print("=" * 70)
        print("Injecting init=/bin/sh into bootargs")
        print("=" * 70)
        print()

        # Method 1: Modify bootargs environment variable
        print("[*] Method 1: Modifying bootargs environment variable")
        print()

        commands = [
            # Print current bootargs
            "printenv bootargs",

            # Append init=/bin/sh
            "setenv bootargs \"${bootargs} init=/bin/sh\"",

            # Verify
            "printenv bootargs",
        ]

        for cmd in commands:
            print(f"[>] {cmd}")
            response = self.send_command(cmd, wait_time=1.0)
            print(response)
            print()

        print("[+] Bootargs modified to spawn root shell")
        return True

    def boot_system(self) -> bool:
        """
        Boot the system with modified parameters.

        Returns:
            True if boot initiated
        """
        print()
        print("=" * 70)
        print("Booting System")
        print("=" * 70)
        print()

        print("[*] Executing 'boot' command...")
        self.send_command("boot", wait_time=2.0)

        print()
        print("[*] Waiting for root shell prompt...")
        print("[!] You should see kernel boot messages followed by a '#' prompt")
        print()

        # Wait for shell prompt
        output = self.wait_for_pattern("#", timeout=120.0)

        if output:
            print()
            print("[✓] Root shell detected!")
            return True
        else:
            print()
            print("[!] Did not detect shell prompt. Check serial output.")
            return False

    def provide_shell_instructions(self):
        """Provide instructions for post-boot setup."""
        print()
        print("=" * 70)
        print("Post-Boot Instructions")
        print("=" * 70)
        print()
        print("You now have a root shell. Execute the following commands:")
        print()
        print("1. Remount root filesystem as read-write:")
        print("   # mount -t ext4 -o remount,rw /dev/mmcblk0p3 /")
        print()
        print("2. Set root password:")
        print("   # passwd")
        print("   (Enter new password when prompted)")
        print()
        print("3. Enable SSH root login:")
        print("   # sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config")
        print()
        print("4. Reboot:")
        print("   # reboot -f")
        print()
        print("5. After reboot, SSH login with new password:")
        print("   $ ssh root@<IP>")
        print()
        print("=" * 70)

    def run_interactive_session(self):
        """
        Run interactive serial terminal session.

        Allows manual command entry after automated setup.
        """
        print()
        print("=" * 70)
        print("Interactive Serial Terminal")
        print("=" * 70)
        print()
        print("[+] Type commands and press Enter")
        print("[+] Press Ctrl+C to exit")
        print()

        if not self.ser or not self.ser.is_open:
            print("[x] Serial port not open")
            return

        try:
            while True:
                # Read from serial
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    sys.stdout.write(data)
                    sys.stdout.flush()

                # Check for keyboard input (non-blocking)
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline()
                    self.ser.write(user_input.encode('utf-8'))
                    self.ser.flush()

        except KeyboardInterrupt:
            print()
            print("[+] Exiting interactive session")
        except Exception as e:
            print(f"\n[x] Error in interactive session: {e}")

    def run_full_jailbreak(self, interactive: bool = True) -> bool:
        """
        Execute complete UART-based jailbreak sequence.

        Args:
            interactive: Drop to interactive shell after injection

        Returns:
            True if successful
        """
        print("=" * 70)
        print("K10 U-Boot UART Jailbreak (Method C)")
        print(f"Serial Port: {self.serial_port} @ {self.baudrate} baud")
        print("=" * 70)
        print()

        # Step 1: Connect
        if not self.connect():
            return False

        try:
            # Step 2: Interrupt autoboot
            if not self.interrupt_autoboot():
                print("[x] Failed to interrupt U-Boot")
                return False

            # Step 3: Inject init=/bin/sh
            if not self.inject_init_shell():
                print("[x] Failed to inject boot parameters")
                return False

            # Step 4: Boot system
            if not self.boot_system():
                print("[!] Boot initiated, but shell prompt not detected")
                print("[!] Check serial output manually")

            # Step 5: Provide instructions
            self.provide_shell_instructions()

            # Step 6: Interactive session
            if interactive:
                self.run_interactive_session()

            return True

        finally:
            self.disconnect()


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 U-Boot UART Jailbreak (Method C)',
        epilog='''
Example usage:
  # Linux
  python method_c_uboot.py /dev/ttyUSB0

  # Windows
  python method_c_uboot.py COM3

  # Custom baud rate
  python method_c_uboot.py /dev/ttyUSB0 --baudrate 115200

Hardware setup:
  1. Connect USB-to-TTL adapter (3.3V logic):
     - GND → GND
     - TX  → RX (miner)
     - RX  → TX (miner)
  2. Do NOT connect VCC (power)
  3. Open case to access UART header (usually near HPS/ARM chip)
        '''
    )

    parser.add_argument('serial_port', help='Serial device (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--no-interactive', action='store_true', help='Skip interactive session')

    args = parser.parse_args()

    # Execute jailbreak
    injector = UBootInjector(args.serial_port, args.baudrate)
    success = injector.run_full_jailbreak(interactive=not args.no_interactive)

    sys.exit(0 if success else 1)
