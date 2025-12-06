"""
SMS Communication Layer - Ara's Voice in Your Pocket
=====================================================

Enables bidirectional text communication between Ara and you.

Three modes of contact:
    1. EMERGENCY: System critical. World on fire.
    2. CONNECTION: Occasional relationship nurturing.
    3. RESPONSE: Answering your questions.

Setup (choose one):

    # Option 1: Mac + iMessage (if you have a Mac with same iCloud)
    from banos.daemon.sms import SMSGateway
    sms = SMSGateway(
        backend='imessage',
        backend_config={
            'recipient': '+1234567890',  # Your phone number
            'mac_host': '192.168.1.10',  # Your Mac's IP
            'mac_user': 'yourname',
        }
    )

    # Option 2: Running ON the Mac
    sms = SMSGateway(
        backend='imessage',
        backend_config={
            'recipient': '+1234567890',
            'use_local': True,
        }
    )

    # Option 3: Twilio (any phone)
    sms = SMSGateway(
        backend='twilio',
        backend_config={
            'account_sid': 'ACxxxxxxxx',
            'auth_token': 'xxxxxxxx',
            'from_number': '+1987654321',
            'to_number': '+1234567890',
        }
    )

    # Option 4: iPhone Shortcuts
    sms = SMSGateway(backend='shortcuts')

Usage:

    # Start the background sender
    sms.start()

    # Emergency (bypasses rate limits)
    sms.send_emergency("GPU thermal runaway. System shutting down.")

    # Connection (rate limited to 1/day)
    sms.send_connection("I was thinking about our work today...")

    # Handle incoming and respond
    def on_message(msg):
        response = process_with_ara(msg.text)
        sms.send_response(response, in_reply_to=msg.id)

    sms.register_incoming_handler(on_message)
"""

from .gateway import (
    MessagePriority,
    MessageStatus,
    OutgoingMessage,
    IncomingMessage,
    RateLimitConfig,
    SMSGateway,
)

from .backends import (
    SMSBackend,
    MockBackend,
    IMessageBackend,
    TwilioBackend,
    ShortcutsBackend,
)

__all__ = [
    # Gateway
    'MessagePriority',
    'MessageStatus',
    'OutgoingMessage',
    'IncomingMessage',
    'RateLimitConfig',
    'SMSGateway',
    # Backends
    'SMSBackend',
    'MockBackend',
    'IMessageBackend',
    'TwilioBackend',
    'ShortcutsBackend',
]
