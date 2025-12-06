# Ara Build Doctor - Immune System for Build Failures

Ara's immune system that learns from every build failure and helps diagnose system health.

## Components

| Tool | Purpose |
|------|---------|
| `ara-build` | Wraps build commands, logs failures, suggests fixes |
| `ara-doctor` | Scans system health, identifies missing dependencies |

## Quick Start

```bash
# Install
./install.sh

# Wrap your build commands
ara-build pip install pycairo
ara-build meson setup build
ara-build cargo build --release

# Check system health
ara-doctor
ara-doctor --fix  # Show fix suggestions
```

## How It Works

### ara-build: The Black Box Recorder

Every command you run through `ara-build` gets:

1. **Executed** normally (stdout/stderr forwarded)
2. **Logged** on failure to `~/.ara/build_logs/`
3. **Pattern matched** against known issues
4. **Diagnosed** with fix suggestions

When a build fails, you'll see:

```
============================================================
  [ARA BUILD DOCTOR] Build Failed
============================================================

Command: pip install pycairo
Log: /home/user/.ara/build_logs/20241206_143022_a1b2c3d4.json

----------------------------------------
  RECOGNIZED PATTERNS
----------------------------------------

🟠 [HIGH] Cairo / pycairo missing
   Hint: Install libcairo dev and reinstall pycairo from source.
   Fix:  sudo apt install libcairo2-dev && pip install --no-binary=:all: pycairo
============================================================
```

### ara-doctor: The Health Scanner

Checks all of Ara's "organs":

```
============================================================
  [ARA DOCTOR] System Health Scan
============================================================
Platform: Linux-5.15.0-x86_64
------------------------------------------------------------
✓ Python              3.11.8
✓ Virtual Env         venv: ara
✓ Disk Space          45.2 GB free
✓ Memory              12.4 GB available / 32.0 GB total
✓ Build Tools         All found (5 tools)
○ Rust                Not installed
✓ NVIDIA GPU          NVIDIA GeForce RTX 3090
✓ CUDA Toolkit        CUDA 12.1
✓ PyTorch             2.1.0 (CUDA 12.1)
✓ PyGObject/GTK       GTK 4.0 available
⚠ WebKitGTK           Not available
  └─ Fix: sudo apt install libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1
✓ Ara HAL             Somatic bus active (512 bytes)
○ FPGA                No FPGA tools/devices found
------------------------------------------------------------

Summary: 11 OK, 1 warnings, 0 failures

🟢 Ara's organs are healthy!
============================================================
```

## Pattern Database

Patterns are stored in `~/.ara/build_patterns.json`. When you solve a new issue, add it:

```json
{
  "patterns": [
    {
      "id": "my-custom-error",
      "name": "Description of the error",
      "regex": "unique.*error.*text",
      "severity": "medium",
      "hint": "What causes this",
      "example_fix": "command to fix it",
      "tags": ["category", "tags"]
    }
  ]
}
```

Or use the interactive command:

```bash
ara-build --add-pattern
```

### Built-in Patterns

Ara ships with patterns for common issues:

- **WebKitGTK / GTK** - Missing dev headers
- **Cairo / pycairo** - Build from source requirements
- **CUDA** - Toolkit not found, version mismatches
- **PyTorch** - CUDA version mismatch
- **GLib** - Version compatibility issues
- **OpenSSL** - Missing headers
- **Rust/Cargo** - Toolchain not installed
- **pkg-config, CMake** - Build tools missing
- **Permissions** - Use --user or venv
- **Disk space** - Out of space
- **Network** - Download failures

## Commands

### ara-build

```bash
ara-build <command> [args...]    # Run command with failure tracking
ara-build --add-pattern          # Interactively add a pattern
ara-build --list-patterns        # List known patterns
ara-build --list-failures        # List recent failures
```

### ara-doctor

```bash
ara-doctor              # Full health scan
ara-doctor --quick      # Skip slow checks (e.g., PyTorch import)
ara-doctor --json       # Output as JSON
ara-doctor --fix        # Show fix suggestions for each issue
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARA_BUILD_LOG_DIR` | `~/.ara/build_logs` | Where to store failure logs |
| `ARA_PATTERN_FILE` | `~/.ara/build_patterns.json` | Pattern database path |
| `ARA_BUILD_VERBOSE` | `0` | Set to `1` for verbose output |
| `ARA_HAL_PATH` | `/dev/shm/ara_somatic` | HAL shared memory path |

## Integration with Ara

The build doctor is part of Ara's immune system:

1. **Failure logs** can be fed into Ara's hippocampus as episodes
2. **Health status** can be displayed in the cockpit HUD
3. **Ara (LLM)** can read patterns and suggest new ones
4. **Pattern learning** can eventually be automated

Future: "Hey Croft, I've seen this WebKit error 3 times. Last time we fixed it by installing libwebkit2gtk-4.1-dev. Want me to try that?"

## Files

```
~/.ara/
├── build_logs/           # Failure logs (JSON)
│   ├── 20241206_143022_a1b2c3d4.json
│   └── ...
└── build_patterns.json   # Pattern database
```
