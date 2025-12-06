"""
SMS Backends - The Physical Layer
=================================

Pluggable backends for actually sending/receiving SMS.

Available backends:
    1. IMessageBackend: For Mac + iPhone (same iCloud account)
    2. TwilioBackend: For any phone via Twilio API
    3. MockBackend: For testing without real SMS

iPhone Setup Options:

Option A - Mac as Relay (Recommended):
    - Your Mac is signed into same iCloud as iPhone
    - Ara runs on Linux, talks to Mac via SSH
    - Mac uses AppleScript to send iMessages

Option B - Shortcuts Automation:
    - iPhone runs a Shortcut that polls a server
    - Ara writes messages to a file/API
    - Shortcut reads and sends

Option C - Twilio:
    - Sign up for Twilio ($)
    - Get a phone number
    - Ara sends via API

Each backend implements:
    - send(text) -> bool
    - receive() -> Optional[str]  (if supported)
"""

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Callable


logger = logging.getLogger(__name__)


class SMSBackend(ABC):
    """Abstract base class for SMS backends."""

    @abstractmethod
    def send(self, text: str) -> bool:
        """Send a text message. Returns True if successful."""
        pass

    def receive(self) -> Optional[str]:
        """Check for new incoming messages. Returns message text or None."""
        return None

    def is_available(self) -> bool:
        """Check if this backend is available/configured."""
        return True


class MockBackend(SMSBackend):
    """
    Mock backend for testing.

    Logs messages instead of sending them.
    Can simulate incoming messages.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        simulate_incoming: bool = False,
    ):
        self.log = logging.getLogger("MockSMS")
        self.log_path = log_path or Path("var/lib/banos/sms/mock_messages.txt")
        self.simulate_incoming = simulate_incoming
        self._incoming_queue: List[str] = []

    def send(self, text: str) -> bool:
        """Log the message instead of sending."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] OUTGOING: {text}\n"

        self.log.info(f"Mock SMS: {text[:50]}...")

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(log_line)
            return True
        except Exception as e:
            self.log.error(f"Failed to log mock message: {e}")
            return False

    def receive(self) -> Optional[str]:
        """Return queued simulated messages."""
        if self._incoming_queue:
            return self._incoming_queue.pop(0)
        return None

    def simulate_incoming_message(self, text: str) -> None:
        """Queue a simulated incoming message."""
        self._incoming_queue.append(text)


class IMessageBackend(SMSBackend):
    """
    iMessage backend using Mac as relay.

    Requirements:
        - A Mac signed into the same iCloud as the target iPhone
        - SSH access to the Mac from the Ara host
        - Messages app configured on Mac

    The backend SSHs to the Mac and runs AppleScript to send.
    """

    def __init__(
        self,
        recipient: str = "",           # Phone number or iMessage email
        mac_host: str = "localhost",   # Mac hostname/IP
        mac_user: str = "",            # SSH user on Mac
        mac_ssh_key: Optional[str] = None,  # Path to SSH key
        use_local: bool = False,       # True if running on the Mac itself
    ):
        self.recipient = recipient
        self.mac_host = mac_host
        self.mac_user = mac_user
        self.mac_ssh_key = mac_ssh_key
        self.use_local = use_local
        self.log = logging.getLogger("iMessageBackend")

        if not recipient:
            self.log.warning("No recipient configured for iMessage")

    def send(self, text: str) -> bool:
        """Send via iMessage using AppleScript."""
        if not self.recipient:
            self.log.error("No recipient configured")
            return False

        # Escape quotes in the message
        escaped_text = text.replace('"', '\\"').replace("'", "'\\''")

        # AppleScript to send iMessage
        applescript = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{self.recipient}" of targetService
            send "{escaped_text}" to targetBuddy
        end tell
        '''

        try:
            if self.use_local:
                # Running on the Mac itself
                result = subprocess.run(
                    ["osascript", "-e", applescript],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            else:
                # SSH to Mac and run
                ssh_cmd = self._build_ssh_command()
                result = subprocess.run(
                    ssh_cmd + ["osascript", "-e", applescript],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            if result.returncode == 0:
                self.log.info(f"iMessage sent to {self.recipient}")
                return True
            else:
                self.log.error(f"iMessage failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.log.error("iMessage send timed out")
            return False
        except Exception as e:
            self.log.error(f"iMessage error: {e}")
            return False

    def _build_ssh_command(self) -> List[str]:
        """Build the SSH command prefix."""
        cmd = ["ssh"]
        if self.mac_ssh_key:
            cmd.extend(["-i", self.mac_ssh_key])
        cmd.extend(["-o", "StrictHostKeyChecking=no"])
        cmd.extend(["-o", "ConnectTimeout=10"])
        cmd.append(f"{self.mac_user}@{self.mac_host}")
        return cmd

    def is_available(self) -> bool:
        """Check if we can reach the Mac."""
        if self.use_local:
            # Check if osascript exists
            try:
                result = subprocess.run(
                    ["which", "osascript"],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False
        else:
            # Check SSH connectivity
            try:
                ssh_cmd = self._build_ssh_command()
                result = subprocess.run(
                    ssh_cmd + ["echo", "ok"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return result.returncode == 0 and "ok" in result.stdout
            except Exception:
                return False


class TwilioBackend(SMSBackend):
    """
    Twilio backend for SMS via API.

    Requirements:
        - Twilio account
        - Account SID and Auth Token
        - A Twilio phone number
    """

    def __init__(
        self,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        to_number: str = "",
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_number = to_number
        self.log = logging.getLogger("TwilioBackend")

        self._client = None

        if account_sid and auth_token:
            try:
                from twilio.rest import Client
                self._client = Client(account_sid, auth_token)
                self.log.info("Twilio client initialized")
            except ImportError:
                self.log.warning("twilio package not installed")
            except Exception as e:
                self.log.error(f"Twilio init failed: {e}")

    def send(self, text: str) -> bool:
        """Send SMS via Twilio."""
        if self._client is None:
            self.log.error("Twilio client not initialized")
            return False

        if not self.to_number or not self.from_number:
            self.log.error("Phone numbers not configured")
            return False

        try:
            message = self._client.messages.create(
                body=text,
                from_=self.from_number,
                to=self.to_number,
            )
            self.log.info(f"Twilio SMS sent: {message.sid}")
            return True
        except Exception as e:
            self.log.error(f"Twilio send failed: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Twilio is configured."""
        return self._client is not None and self.to_number and self.from_number


class ShortcutsBackend(SMSBackend):
    """
    Backend that works with iPhone Shortcuts app.

    Instead of sending directly, writes messages to a file that
    an iPhone Shortcut can poll and send.

    Setup:
        1. Create a Shortcut on iPhone that:
           - Reads from a shared location (iCloud, server)
           - Sends any pending messages
           - Marks them as sent
        2. Configure this backend with the shared file path
        3. Run the Shortcut periodically (or on-demand)
    """

    def __init__(
        self,
        outbox_path: Optional[Path] = None,
        inbox_path: Optional[Path] = None,
    ):
        self.outbox_path = outbox_path or Path("var/lib/banos/sms/shortcuts_outbox.json")
        self.inbox_path = inbox_path or Path("var/lib/banos/sms/shortcuts_inbox.json")
        self.log = logging.getLogger("ShortcutsBackend")

    def send(self, text: str) -> bool:
        """Write message to outbox for Shortcut to pick up."""
        import json

        try:
            self.outbox_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing messages
            messages = []
            if self.outbox_path.exists():
                try:
                    with open(self.outbox_path) as f:
                        messages = json.load(f)
                except Exception:
                    messages = []

            # Add new message
            messages.append({
                "text": text,
                "timestamp": time.time(),
                "sent": False,
            })

            # Save
            with open(self.outbox_path, "w") as f:
                json.dump(messages, f, indent=2)

            self.log.info("Message written to Shortcuts outbox")
            return True

        except Exception as e:
            self.log.error(f"Shortcuts outbox write failed: {e}")
            return False

    def receive(self) -> Optional[str]:
        """Check inbox for messages from Shortcut."""
        import json

        if not self.inbox_path.exists():
            return None

        try:
            with open(self.inbox_path) as f:
                messages = json.load(f)

            if not messages:
                return None

            # Get first unprocessed message
            for msg in messages:
                if not msg.get("processed", False):
                    msg["processed"] = True
                    with open(self.inbox_path, "w") as f:
                        json.dump(messages, f, indent=2)
                    return msg.get("text", "")

            return None

        except Exception as e:
            self.log.debug(f"Inbox read error: {e}")
            return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SMSBackend',
    'MockBackend',
    'IMessageBackend',
    'TwilioBackend',
    'ShortcutsBackend',
]
