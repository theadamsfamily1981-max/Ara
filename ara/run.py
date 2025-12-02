"""
Ara Launcher - Start Brain and Avatar together

Usage:
    python -m ara.run              # Start both processes
    python -m ara.run --brain-only # Start only the brain server
    python -m ara.run --avatar-only # Start only the avatar (brain must be running)
    python -m ara.run --text       # Use text mode instead of voice
"""

import argparse
import subprocess
import sys
import time
import signal
import os

def main():
    parser = argparse.ArgumentParser(description="Ara Launcher")
    parser.add_argument("--brain-only", action="store_true", help="Start only brain server")
    parser.add_argument("--avatar-only", action="store_true", help="Start only avatar (brain must be running)")
    parser.add_argument("--text", action="store_true", help="Use text mode for avatar")
    parser.add_argument("--mode", default="MODE_B", help="Hardware mode (MODE_A, MODE_B, MODE_C)")
    parser.add_argument("--port", type=int, default=8008, help="Brain server port")
    args = parser.parse_args()

    processes = []

    def cleanup(signum=None, frame=None):
        """Clean up child processes."""
        print("\nShutting down...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start brain server
        if not args.avatar_only:
            print(f"Starting Ara Brain Server (mode={args.mode}, port={args.port})...")
            env = os.environ.copy()
            env["ARA_MODE"] = args.mode

            brain_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "ara.server.core_server",
                    "--mode", args.mode,
                    "--port", str(args.port)
                ],
                env=env
            )
            processes.append(brain_proc)

            # Wait for server to start
            print("Waiting for brain to initialize...")
            time.sleep(3)

            if brain_proc.poll() is not None:
                print("Brain server failed to start!")
                return 1

        if args.brain_only:
            print("Brain server running. Press Ctrl+C to stop.")
            brain_proc.wait()
            return 0

        # Start avatar
        print("Starting Ara Avatar...")
        avatar_args = [sys.executable, "-m", "ara.avatar.loop"]
        if args.text:
            avatar_args.append("--text")

        avatar_proc = subprocess.Popen(avatar_args)
        processes.append(avatar_proc)

        # Wait for avatar to exit
        avatar_proc.wait()

    except KeyboardInterrupt:
        pass

    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
