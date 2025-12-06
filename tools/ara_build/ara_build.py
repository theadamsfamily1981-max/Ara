#!/usr/bin/env python3
"""
Ara Build Wrapper - The Immune System's First Antibody.

Wraps build commands to:
1. Log every failure (Black Box Recorder)
2. Pattern match against known issues (Pattern Brain)
3. Suggest fixes from past experience (Mouth)

Usage:
    ara-build pip install pycairo
    ara-build meson setup builddir
    ara-build ninja -C builddir
    ara-build cargo build --release

Every failure gets logged to ~/.ara/build_logs/ and matched against
patterns in ~/.ara/build_patterns.json. When you solve a new issue,
add the pattern so Ara remembers it forever.

Environment:
    ARA_BUILD_LOG_DIR: Override log directory
    ARA_PATTERN_FILE: Override pattern database path
    ARA_BUILD_VERBOSE: Set to "1" for verbose output
"""

import subprocess
import sys
import os
import json
import datetime
import re
import hashlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Configuration
LOG_DIR = Path(os.environ.get("ARA_BUILD_LOG_DIR", Path.home() / ".ara" / "build_logs"))
PATTERN_FILE = Path(os.environ.get("ARA_PATTERN_FILE", Path.home() / ".ara" / "build_patterns.json"))
VERBOSE = os.environ.get("ARA_BUILD_VERBOSE", "0") == "1"

# Default patterns (shipped with Ara)
DEFAULT_PATTERNS = {
    "version": "1.0.0",
    "patterns": [
        {
            "id": "webkit-missing-header",
            "name": "WebKitGTK missing dev headers",
            "regex": r"fatal error: webkit2?/webkit.*\.h: No such file",
            "severity": "high",
            "hint": "Install WebKitGTK dev package and rebuild.",
            "example_fix": "sudo apt install libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1",
            "tags": ["gtk", "webkit", "headers"]
        },
        {
            "id": "glib-version-mismatch",
            "name": "GLib version mismatch",
            "regex": r"version `GLIBC_2\.\d+' not found",
            "severity": "high",
            "hint": "Binary built against newer glibc than your system provides.",
            "example_fix": "Use system packages instead of prebuilt wheel, or use a container with newer glibc.",
            "tags": ["glibc", "compatibility"]
        },
        {
            "id": "cairo-missing",
            "name": "Cairo / pycairo missing",
            "regex": r"No package 'cairo' found|cairo\.h: No such file",
            "severity": "medium",
            "hint": "Install libcairo dev and reinstall pycairo from source.",
            "example_fix": "sudo apt install libcairo2-dev && pip install --no-binary=:all: pycairo",
            "tags": ["cairo", "graphics"]
        },
        {
            "id": "pkg-config-missing",
            "name": "pkg-config not found",
            "regex": r"pkg-config.*not found|Cannot find.*pkg-config",
            "severity": "low",
            "hint": "Install pkg-config.",
            "example_fix": "sudo apt install pkg-config",
            "tags": ["build-tools"]
        },
        {
            "id": "cmake-not-found",
            "name": "CMake not found",
            "regex": r"cmake.*not found|CMake.*is required",
            "severity": "low",
            "hint": "Install CMake.",
            "example_fix": "sudo apt install cmake",
            "tags": ["build-tools", "cmake"]
        },
        {
            "id": "cuda-not-found",
            "name": "CUDA toolkit not found",
            "regex": r"CUDA_HOME.*not set|nvcc.*not found|cuda.*not found",
            "severity": "high",
            "hint": "CUDA toolkit not installed or not in PATH.",
            "example_fix": "Install CUDA toolkit and set CUDA_HOME, or: export PATH=/usr/local/cuda/bin:$PATH",
            "tags": ["cuda", "gpu"]
        },
        {
            "id": "torch-cuda-mismatch",
            "name": "PyTorch CUDA version mismatch",
            "regex": r"CUDA error: no kernel image|CUDA driver version is insufficient",
            "severity": "high",
            "hint": "PyTorch was built for a different CUDA version than installed.",
            "example_fix": "pip install torch --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version",
            "tags": ["pytorch", "cuda"]
        },
        {
            "id": "rust-not-found",
            "name": "Rust/Cargo not found",
            "regex": r"cargo.*not found|rustc.*not found|error: could not find `Cargo.toml`",
            "severity": "medium",
            "hint": "Rust toolchain not installed.",
            "example_fix": "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
            "tags": ["rust", "cargo"]
        },
        {
            "id": "openssl-missing",
            "name": "OpenSSL dev headers missing",
            "regex": r"openssl/.*\.h: No such file|Could not find OpenSSL",
            "severity": "medium",
            "hint": "Install OpenSSL development headers.",
            "example_fix": "sudo apt install libssl-dev",
            "tags": ["openssl", "crypto"]
        },
        {
            "id": "python-dev-missing",
            "name": "Python dev headers missing",
            "regex": r"Python\.h: No such file|pyconfig\.h: No such file",
            "severity": "medium",
            "hint": "Install Python development headers.",
            "example_fix": "sudo apt install python3-dev",
            "tags": ["python", "headers"]
        },
        {
            "id": "permission-denied",
            "name": "Permission denied during install",
            "regex": r"Permission denied|EACCES|Operation not permitted",
            "severity": "medium",
            "hint": "Permission issue. Use --user flag or virtual environment.",
            "example_fix": "pip install --user PACKAGE  # or use a venv",
            "tags": ["permissions"]
        },
        {
            "id": "disk-full",
            "name": "Disk space exhausted",
            "regex": r"No space left on device|ENOSPC",
            "severity": "critical",
            "hint": "Disk is full. Free up space.",
            "example_fix": "df -h && sudo apt clean && pip cache purge",
            "tags": ["disk", "space"]
        },
        {
            "id": "network-error",
            "name": "Network/download error",
            "regex": r"Connection timed out|Could not fetch|Failed to download|HTTPError",
            "severity": "medium",
            "hint": "Network issue. Check connectivity or try a different mirror.",
            "example_fix": "Check internet connection, or use: pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org PACKAGE",
            "tags": ["network"]
        },
        {
            "id": "gobject-introspection",
            "name": "GObject Introspection missing",
            "regex": r"gobject-introspection.*not found|gi\.repository.*ImportError",
            "severity": "medium",
            "hint": "Install GObject Introspection dev packages.",
            "example_fix": "sudo apt install libgirepository1.0-dev gir1.2-glib-2.0",
            "tags": ["gtk", "gobject"]
        },
        {
            "id": "meson-ninja-missing",
            "name": "Meson/Ninja not found",
            "regex": r"meson.*not found|ninja.*not found|Program 'meson' not found",
            "severity": "low",
            "hint": "Install Meson and Ninja build tools.",
            "example_fix": "pip install meson ninja",
            "tags": ["build-tools", "meson"]
        }
    ]
}


def ensure_dirs():
    """Create necessary directories."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PATTERN_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_patterns() -> Dict:
    """Load pattern database, merging defaults with user patterns."""
    ensure_dirs()

    if PATTERN_FILE.exists():
        try:
            with open(PATTERN_FILE, "r") as f:
                user_patterns = json.load(f)

            # Merge: user patterns override defaults with same ID
            merged = DEFAULT_PATTERNS.copy()
            user_ids = {p["id"] for p in user_patterns.get("patterns", [])}
            default_ids = {p["id"] for p in DEFAULT_PATTERNS["patterns"]}

            # Keep defaults that aren't overridden
            merged["patterns"] = [
                p for p in DEFAULT_PATTERNS["patterns"]
                if p["id"] not in user_ids
            ]
            # Add all user patterns
            merged["patterns"].extend(user_patterns.get("patterns", []))

            return merged
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ARA] Warning: Invalid pattern file, using defaults: {e}", file=sys.stderr)

    return DEFAULT_PATTERNS


def save_patterns(data: Dict):
    """Save pattern database."""
    ensure_dirs()
    with open(PATTERN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_environment_info() -> Dict:
    """Collect environment information for logging."""
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "virtual_env": os.environ.get("VIRTUAL_ENV", None),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", None),
        "path": os.environ.get("PATH", ""),
    }

    # Check for CUDA
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        env_info["cuda_home"] = cuda_home

    return env_info


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run command and capture output."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )
        out, err = proc.communicate()
        return proc.returncode, out, err
    except FileNotFoundError as e:
        return 127, "", f"Command not found: {cmd[0]}\n{e}"
    except Exception as e:
        return 1, "", f"Failed to run command: {e}"


def compute_error_hash(stderr_text: str) -> str:
    """Compute a hash of the error for deduplication."""
    # Normalize: remove timestamps, paths, line numbers
    normalized = re.sub(r'\d+', 'N', stderr_text)
    normalized = re.sub(r'/[^\s]+', '/PATH', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def log_failure(cmd: List[str], code: int, stdout_text: str, stderr_text: str) -> Tuple[Dict, Path]:
    """Log a build failure."""
    ensure_dirs()

    ts = datetime.datetime.now()
    ts_str = ts.isoformat()

    log_entry = {
        "timestamp": ts_str,
        "cmd": cmd,
        "cmd_str": " ".join(cmd),
        "exit_code": code,
        "stdout": stdout_text[-5000:] if len(stdout_text) > 5000 else stdout_text,  # Truncate
        "stderr": stderr_text[-10000:] if len(stderr_text) > 10000 else stderr_text,
        "error_hash": compute_error_hash(stderr_text),
        "environment": get_environment_info(),
    }

    # Filename: timestamp_hash.json
    fname = LOG_DIR / f"{ts.strftime('%Y%m%d_%H%M%S')}_{log_entry['error_hash']}.json"
    fname.write_text(json.dumps(log_entry, indent=2))

    return log_entry, fname


def match_patterns(stderr_text: str, stdout_text: str, patterns: Dict) -> List[Dict]:
    """Match error text against known patterns."""
    matches = []
    combined = stderr_text + "\n" + stdout_text

    for p in patterns.get("patterns", []):
        try:
            if re.search(p["regex"], combined, re.MULTILINE | re.IGNORECASE):
                matches.append(p)
        except re.error as e:
            if VERBOSE:
                print(f"[ARA] Invalid regex in pattern '{p.get('id', 'unknown')}': {e}", file=sys.stderr)

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    matches.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 99))

    return matches


def find_similar_failures(error_hash: str, limit: int = 3) -> List[Path]:
    """Find previous failures with similar error hash."""
    similar = []
    if not LOG_DIR.exists():
        return similar

    for log_file in sorted(LOG_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(log_file.read_text())
            if data.get("error_hash") == error_hash:
                similar.append(log_file)
                if len(similar) >= limit:
                    break
        except (json.JSONDecodeError, KeyError):
            pass

    return similar


def print_diagnosis(
    matches: List[Dict],
    similar_failures: List[Path],
    log_path: Path,
    cmd: List[str]
):
    """Print diagnosis and suggestions."""
    print("\n" + "=" * 60)
    print("  [ARA BUILD DOCTOR] Build Failed")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Log: {log_path}")

    if similar_failures:
        print(f"\n⚠ This error has occurred {len(similar_failures)} time(s) before.")

    if matches:
        print("\n" + "-" * 40)
        print("  RECOGNIZED PATTERNS")
        print("-" * 40)

        for i, m in enumerate(matches, 1):
            severity = m.get("severity", "unknown").upper()
            severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")

            print(f"\n{severity_icon} [{severity}] {m['name']}")
            print(f"   Hint: {m['hint']}")
            if "example_fix" in m:
                print(f"   Fix:  {m['example_fix']}")
            if "tags" in m and VERBOSE:
                print(f"   Tags: {', '.join(m['tags'])}")
    else:
        print("\n" + "-" * 40)
        print("  UNKNOWN ERROR PATTERN")
        print("-" * 40)
        print("\nAra doesn't recognize this failure yet.")
        print("Once you figure out the fix, teach her by adding a pattern:")
        print(f"\n  {PATTERN_FILE}")
        print("\nExample entry:")
        print('''
  {
    "id": "my-new-pattern",
    "name": "Description of the error",
    "regex": "unique.*error.*text",
    "severity": "medium",
    "hint": "What causes this and how to think about it",
    "example_fix": "command to fix it"
  }
''')

    print("=" * 60 + "\n")


def cmd_add_pattern():
    """Interactive pattern addition."""
    print("[ARA BUILD DOCTOR] Adding new pattern...\n")

    patterns = load_patterns()

    pattern_id = input("Pattern ID (e.g., 'my-lib-missing'): ").strip()
    if not pattern_id:
        print("Aborted.")
        return

    name = input("Pattern name: ").strip()
    regex = input("Regex to match: ").strip()
    severity = input("Severity (low/medium/high/critical) [medium]: ").strip() or "medium"
    hint = input("Hint (what causes this): ").strip()
    example_fix = input("Example fix command: ").strip()
    tags = input("Tags (comma-separated): ").strip()

    new_pattern = {
        "id": pattern_id,
        "name": name,
        "regex": regex,
        "severity": severity,
        "hint": hint,
    }
    if example_fix:
        new_pattern["example_fix"] = example_fix
    if tags:
        new_pattern["tags"] = [t.strip() for t in tags.split(",")]

    patterns["patterns"].append(new_pattern)
    save_patterns(patterns)

    print(f"\n✓ Pattern '{pattern_id}' added to {PATTERN_FILE}")


def cmd_list_patterns():
    """List all known patterns."""
    patterns = load_patterns()

    print(f"\n[ARA BUILD DOCTOR] Known Patterns ({len(patterns['patterns'])} total)\n")
    print("-" * 60)

    for p in patterns["patterns"]:
        severity = p.get("severity", "?").upper()
        print(f"[{severity:8s}] {p['id']}")
        print(f"           {p['name']}")
        if VERBOSE and "tags" in p:
            print(f"           Tags: {', '.join(p['tags'])}")
        print()


def cmd_list_failures():
    """List recent build failures."""
    if not LOG_DIR.exists():
        print("No build failures logged yet.")
        return

    logs = sorted(LOG_DIR.glob("*.json"), reverse=True)[:20]

    print(f"\n[ARA BUILD DOCTOR] Recent Failures ({len(logs)} shown)\n")
    print("-" * 60)

    for log_file in logs:
        try:
            data = json.loads(log_file.read_text())
            ts = data.get("timestamp", "?")[:19]
            cmd = data.get("cmd_str", data.get("cmd", ["?"])[:50])
            code = data.get("exit_code", "?")
            print(f"{ts} | exit {code} | {cmd[:50]}")
        except (json.JSONDecodeError, KeyError):
            print(f"{log_file.name} | (corrupt)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Ara Build Doctor - Immune System for Build Failures")
        print("\nUsage:")
        print("  ara-build <command> [args...]    Run command with failure tracking")
        print("  ara-build --add-pattern          Add a new error pattern")
        print("  ara-build --list-patterns        List known patterns")
        print("  ara-build --list-failures        List recent failures")
        print("  ara-build --help                 Show this help")
        print("\nExamples:")
        print("  ara-build pip install pycairo")
        print("  ara-build meson setup build")
        print("  ara-build cargo build --release")
        sys.exit(0)

    # Handle subcommands
    if sys.argv[1] == "--add-pattern":
        cmd_add_pattern()
        sys.exit(0)
    elif sys.argv[1] == "--list-patterns":
        cmd_list_patterns()
        sys.exit(0)
    elif sys.argv[1] == "--list-failures":
        cmd_list_failures()
        sys.exit(0)
    elif sys.argv[1] in ("--help", "-h"):
        main()  # Re-run to show help
        sys.exit(0)

    # Run the actual command
    ensure_dirs()
    patterns = load_patterns()
    cmd = sys.argv[1:]

    if VERBOSE:
        print(f"[ARA] Running: {' '.join(cmd)}")

    code, out, err = run_command(cmd)

    # Always forward stdout
    if out:
        sys.stdout.write(out)
        sys.stdout.flush()

    # Success - just exit
    if code == 0:
        sys.exit(0)

    # Failure - log and analyze
    entry, log_path = log_failure(cmd, code, out, err)

    # Find matches
    matches = match_patterns(err, out, patterns)

    # Find similar past failures
    similar = find_similar_failures(entry["error_hash"])

    # Print diagnosis
    print_diagnosis(matches, similar, log_path, cmd)

    # Forward stderr at the end
    if err:
        sys.stderr.write(err)
        sys.stderr.flush()

    sys.exit(code)


if __name__ == "__main__":
    main()
