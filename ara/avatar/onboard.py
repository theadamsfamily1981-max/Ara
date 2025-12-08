#!/usr/bin/env python3
"""
Friend Onboarding CLI

Invite friends to use Ara via the Cathedral.
Generates VPN credentials and provisions their Brain Jar.

Usage:
    # Invite a friend (free tier)
    python -m ara.avatar.onboard --friend alice --name "Alice Chen"

    # Invite with Pro tier
    python -m ara.avatar.onboard --friend bob --name "Bob Smith" --tier pro

    # List all friends
    python -m ara.avatar.onboard --list

    # Check friend status
    python -m ara.avatar.onboard --status alice

    # Remove friend
    python -m ara.avatar.onboard --remove alice
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import json
from pathlib import Path

from ara.avatar.cathedral import CathedralServer, SubscriptionTier


def print_vpn_config(config: str, user_id: str):
    """Print VPN config in a nice format."""
    print("\n" + "=" * 60)
    print(f"VPN Configuration for {user_id}")
    print("=" * 60)
    print(config)
    print("=" * 60)
    print("\nSave this to a .conf file and import into WireGuard.")
    print("Replace <GENERATED_CLIENT_PRIVATE_KEY> with your private key.")


def print_disclosures(disclosures: str):
    """Print user disclosures."""
    print("\n" + "-" * 60)
    print("IMPORTANT: Share these disclosures with the user")
    print("-" * 60)
    # Print first 1000 chars of disclosures
    if len(disclosures) > 1000:
        print(disclosures[:1000] + "...\n[truncated - full disclosures in onboarding URL]")
    else:
        print(disclosures)


async def invite_friend(
    server: CathedralServer,
    friend_id: str,
    display_name: str,
    tier: str,
) -> None:
    """Invite a new friend."""
    # Map tier string to enum
    tier_map = {
        "free": SubscriptionTier.FREE,
        "pro": SubscriptionTier.PRO,
        "power": SubscriptionTier.POWER,
        "founder": SubscriptionTier.FOUNDER,
    }

    if tier.lower() not in tier_map:
        print(f"Error: Invalid tier '{tier}'. Choose from: free, pro, power")
        return

    tier_enum = tier_map[tier.lower()]

    print(f"\nProvisioning Brain Jar for {display_name}...")
    print(f"  User ID: {friend_id}")
    print(f"  Tier: {tier_enum.value}")

    result = await server.provision_user(
        user_id=friend_id,
        display_name=display_name,
        tier=tier_enum,
        invited_by="founder",
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    print(f"\n✅ Successfully invited {display_name}!")
    print(f"   Status: {result['status']}")
    print(f"   Onboarding URL: {result['onboarding_url']}")

    # Print VPN config
    print_vpn_config(result['vpn_config'], friend_id)

    # Print disclosures
    print_disclosures(result['disclosures'])

    print("\n📋 Next Steps:")
    print("   1. Send the VPN config to your friend")
    print("   2. Have them read and accept the disclosures")
    print("   3. They can then connect via WireGuard and chat!")


async def list_friends(server: CathedralServer) -> None:
    """List all invited friends."""
    users = server.list_users()

    if not users:
        print("\nNo friends invited yet.")
        print("Use: python -m ara.avatar.onboard --friend <id> --name <name>")
        return

    print(f"\n{'User ID':<20} {'Name':<20} {'Tier':<10} {'Consent':<10} {'Last Active'}")
    print("-" * 80)

    for user in users:
        import time
        if user['last_active']:
            last_active = time.strftime("%Y-%m-%d %H:%M", time.localtime(user['last_active']))
        else:
            last_active = "Never"

        consent = "✓" if user['consent_given'] else "✗"

        print(f"{user['user_id']:<20} {user['display_name']:<20} {user['tier']:<10} {consent:<10} {last_active}")

    print(f"\nTotal: {len(users)} friends")


async def check_status(server: CathedralServer, user_id: str) -> None:
    """Check status of a specific friend."""
    stats = server.get_user_stats(user_id)

    if "error" in stats:
        print(f"\nError: {stats['error']}")
        return

    import time

    print(f"\n=== Status for {stats['display_name']} ===")
    print(f"  User ID: {stats['user_id']}")
    print(f"  Tier: {stats['tier']}")
    print(f"  Consent Given: {'Yes' if stats['consent_given'] else 'No'}")
    print(f"  Message Count: {stats['message_count']}")
    print(f"  Memory Count: {stats['memory_count']}")
    print(f"  Created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(stats['created_at']))}")
    print(f"  Last Active: {time.strftime('%Y-%m-%d %H:%M', time.localtime(stats['last_active']))}")
    print(f"  Session Active: {'Yes' if stats['session_active'] else 'No'}")


async def remove_friend(server: CathedralServer, user_id: str, confirm: bool) -> None:
    """Remove a friend (with confirmation)."""
    stats = server.get_user_stats(user_id)

    if "error" in stats:
        print(f"\nError: {stats['error']}")
        return

    if not confirm:
        print(f"\n⚠️  WARNING: This will remove {stats['display_name']} and their data.")
        print(f"   Messages: {stats['message_count']}")
        print(f"   Memories: {stats['memory_count']}")
        print("\n   To confirm, add --confirm flag.")
        return

    result = await server.deprovision_user(user_id)

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    print(f"\n✅ Removed {user_id}")


async def show_stats(server: CathedralServer) -> None:
    """Show Cathedral statistics."""
    stats = server.get_cathedral_stats()

    print("\n=== Cathedral Statistics ===")
    print(f"  Total Users: {stats['total_users']}")
    print(f"  Active (24h): {stats['active_24h']}")
    print(f"  Total Messages: {stats['total_messages']}")
    print(f"  Total Memories: {stats['total_memories']}")
    print(f"\n  Tier Breakdown:")
    for tier, count in stats['tier_breakdown'].items():
        print(f"    {tier}: {count}")


async def main():
    parser = argparse.ArgumentParser(
        description="Ara Friend Onboarding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Invite a friend
    python -m ara.avatar.onboard --friend alice --name "Alice Chen" --tier pro

    # List all friends
    python -m ara.avatar.onboard --list

    # Check friend status
    python -m ara.avatar.onboard --status alice
        """,
    )

    parser.add_argument(
        "--friend", "-f",
        type=str,
        help="User ID for the friend (e.g., 'alice', 'friend_001')",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Display name for the friend",
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        default="free",
        help="Subscription tier: free, pro, power (default: free)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all invited friends",
    )
    parser.add_argument(
        "--status", "-s",
        type=str,
        help="Check status of a specific friend",
    )
    parser.add_argument(
        "--remove", "-r",
        type=str,
        help="Remove a friend",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm removal (required with --remove)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show Cathedral statistics",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/var/ara/cathedral"),
        help="Data directory for Cathedral",
    )

    args = parser.parse_args()

    # Initialize server
    server = CathedralServer(data_dir=args.data_dir)

    # Route to action
    if args.list:
        await list_friends(server)
    elif args.status:
        await check_status(server, args.status)
    elif args.remove:
        await remove_friend(server, args.remove, args.confirm)
    elif args.stats:
        await show_stats(server)
    elif args.friend:
        if not args.name:
            print("Error: --name is required when inviting a friend")
            sys.exit(1)
        await invite_friend(server, args.friend, args.name, args.tier)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
