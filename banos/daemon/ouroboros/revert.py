#!/usr/bin/env python3
"""
Emergency Revert Script for Ouroboros
=====================================

This script is the nuclear option. It:
1. Kills any running Ouroboros processes
2. Rolls back all in-memory mutations
3. Optionally wipes the mutations/ directory
4. Disables Ouroboros via environment

Usage:
    # Soft revert (rollback in-memory only)
    python -m banos.daemon.ouroboros.revert

    # Hard revert (also wipe mutations/ directory)
    python -m banos.daemon.ouroboros.revert --hard

    # Full reset (reinstall from git, wipe everything)
    python -m banos.daemon.ouroboros.revert --full-reset

Keep this script OUTSIDE the Ouroboros system so it can't be mutated.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REVERT] %(message)s"
)
log = logging.getLogger(__name__)


def disable_ouroboros() -> None:
    """Set environment to disable Ouroboros."""
    os.environ["OUROBOROS_ENABLED"] = "0"
    os.environ["OUROBOROS_AUTO_APPLY"] = "0"

    log.info("Disabled Ouroboros via environment")

    # Also try to write to a persistent location
    try:
        disable_file = Path("/var/lib/ara/ouroboros_disabled")
        disable_file.parent.mkdir(parents=True, exist_ok=True)
        disable_file.write_text("1")
        log.info(f"Wrote disable flag to {disable_file}")
    except PermissionError:
        log.warning("Cannot write persistent disable flag (no permission)")


def rollback_in_memory() -> int:
    """Attempt to rollback any in-memory mutations."""
    try:
        from banos.daemon.ouroboros import AntifragilitySystem

        system = AntifragilitySystem(Path("/home/user/Ara"))
        results = system.emergency_rollback()

        success_count = sum(1 for r in results if r.success)
        log.info(f"Rolled back {success_count}/{len(results)} mutations")

        return success_count

    except ImportError:
        log.warning("Cannot import Ouroboros - may not be installed")
        return 0
    except Exception as e:
        log.error(f"Rollback failed: {e}")
        return 0


def wipe_mutations_dir(repo_root: Path) -> None:
    """Wipe the mutations directory."""
    mutations_dir = repo_root / "mutations"

    if mutations_dir.exists():
        log.warning(f"Wiping {mutations_dir}")
        shutil.rmtree(mutations_dir)
        log.info("Mutations directory wiped")
    else:
        log.info("No mutations directory found")


def reset_from_git(repo_root: Path, branch: str = "main") -> bool:
    """Hard reset to a git branch."""
    log.warning(f"Resetting {repo_root} to {branch}")

    try:
        # Stash any changes
        subprocess.run(
            ["git", "stash"],
            cwd=str(repo_root),
            check=False,
        )

        # Fetch latest
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=str(repo_root),
            check=True,
        )

        # Hard reset
        subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=str(repo_root),
            check=True,
        )

        log.info(f"Reset to origin/{branch}")
        return True

    except subprocess.CalledProcessError as e:
        log.error(f"Git reset failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Emergency revert for Ouroboros self-modification"
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Also wipe the mutations/ directory",
    )
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help="Full reset: git reset --hard + wipe mutations",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="/home/user/Ara",
        help="Path to Ara repository",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Branch to reset to (for --full-reset)",
    )

    args = parser.parse_args()
    repo_root = Path(args.repo)

    print("=" * 60)
    print("OUROBOROS EMERGENCY REVERT")
    print("=" * 60)

    # Step 1: Always disable Ouroboros first
    disable_ouroboros()

    # Step 2: Rollback in-memory mutations
    print("\n[1/3] Rolling back in-memory mutations...")
    rollback_in_memory()

    # Step 3: Wipe mutations if requested
    if args.hard or args.full_reset:
        print("\n[2/3] Wiping mutations directory...")
        wipe_mutations_dir(repo_root)
    else:
        print("\n[2/3] Skipping mutations wipe (use --hard to wipe)")

    # Step 4: Git reset if requested
    if args.full_reset:
        print(f"\n[3/3] Resetting to origin/{args.branch}...")
        if not reset_from_git(repo_root, args.branch):
            print("WARNING: Git reset failed!")
            sys.exit(1)
    else:
        print("\n[3/3] Skipping git reset (use --full-reset)")

    print("\n" + "=" * 60)
    print("REVERT COMPLETE")
    print("=" * 60)
    print("\nOuroboros is now DISABLED.")
    print("To re-enable, set OUROBOROS_ENABLED=1")

    if args.full_reset:
        print(f"\nRepository reset to origin/{args.branch}")
        print("You may need to reinstall dependencies: pip install -e .")


if __name__ == "__main__":
    main()
