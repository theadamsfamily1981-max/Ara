#!/usr/bin/env bash
#
# T-FAN Topology Screensaver - xscreensaver setup
# Configures xscreensaver integration automatically
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
SCREENSAVER_DIR="$REPO_ROOT/tfan/viz/screensaver"

echo "🎨 Setting up T-FAN Topology Screensaver for xscreensaver..."

# Check if xscreensaver is installed
if ! command -v xscreensaver &> /dev/null; then
    echo "⚠️  xscreensaver not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y xscreensaver xscreensaver-data-extra
fi

# Create wrapper in user's bin
BIN_DIR="$HOME/bin"
WRAPPER="$BIN_DIR/tfan-topo-saver"

mkdir -p "$BIN_DIR"

cat > "$WRAPPER" << EOF
#!/usr/bin/env bash
cd "$SCREENSAVER_DIR"
exec ./xscreensaver-wrapper.sh "\$@"
EOF

chmod +x "$WRAPPER"
chmod +x "$SCREENSAVER_DIR/xscreensaver-wrapper.sh"
chmod +x "$SCREENSAVER_DIR/topo_screensaver.py"
chmod +x "$SCREENSAVER_DIR/metrics_bridge.py"

echo "✓ Created wrapper: $WRAPPER"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "export PATH=\"\$HOME/bin:\$PATH\"" >> "$HOME/.bashrc"
    echo "✓ Added $BIN_DIR to PATH (restart terminal or run: source ~/.bashrc)"
fi

# Configure xscreensaver
XSCREENSAVER_CONF="$HOME/.xscreensaver"

if [[ -f "$XSCREENSAVER_CONF" ]]; then
    # Backup existing config
    cp "$XSCREENSAVER_CONF" "$XSCREENSAVER_CONF.bak"
    echo "✓ Backed up existing xscreensaver config"
fi

# Check if our screensaver is already configured
if ! grep -q "tfan-topo-saver" "$XSCREENSAVER_CONF" 2>/dev/null; then
    # Add our screensaver to the programs list
    if [[ -f "$XSCREENSAVER_CONF" ]]; then
        # Insert after the programs: line
        sed -i '/^programs:/a \
\  "T-FAN Topology Screensaver" tfan-topo-saver \\n\\' "$XSCREENSAVER_CONF"
    else
        # Create minimal config
        cat > "$XSCREENSAVER_CONF" << 'XSCONF'
mode: one
timeout: 0:10:00
cycle: 0:10:00
lock: False
lockTimeout: 0:00:00
passwdTimeout: 0:00:30
visualID: default
installColormap: True
verbose: False
timestamp: True
fade: True
unfade: False
fadeSeconds: 0:00:03
fadeTicks: 20
dpmsEnabled: True
dpmsStandby: 2:00:00
dpmsSuspend: 2:00:00
dpmsOff: 4:00:00

programs: \
  "T-FAN Topology Screensaver" tfan-topo-saver \n\
  blank \n\

XSCONF
    fi
    echo "✓ Added T-FAN screensaver to xscreensaver config"
fi

# Start xscreensaver daemon if not running
if ! pgrep -x xscreensaver > /dev/null; then
    xscreensaver -no-splash &
    echo "✓ Started xscreensaver daemon"
else
    # Restart to pick up new config
    xscreensaver-command -exit
    sleep 1
    xscreensaver -no-splash &
    echo "✓ Restarted xscreensaver daemon"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To configure:"
echo "  1. Run: xscreensaver-demo"
echo "  2. Select 'T-FAN Topology Screensaver' from the list"
echo "  3. Click 'Settings' to adjust timeout"
echo "  4. Click 'Preview' to test"
echo ""
echo "To test now:"
echo "  xscreensaver-command -activate"
echo ""
