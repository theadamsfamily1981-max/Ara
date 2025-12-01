#!/bin/bash
set -e

echo "🚀 Installing T-FAN GNOME Integration..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Directories
EXTENSION_DIR="$HOME/.local/share/gnome-shell/extensions/tfan@quanta-meis-nib-cis"
APP_DIR="$HOME/.local/share/tfan"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICONS_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"

echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$EXTENSION_DIR"
mkdir -p "$APP_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICONS_DIR"
mkdir -p "$HOME/.cache/tfan"

echo -e "${BLUE}Installing GNOME Shell extension...${NC}"
cp extension/extension.js "$EXTENSION_DIR/"
cp extension/metadata.json "$EXTENSION_DIR/"

# Create assets if they don't exist
mkdir -p "$EXTENSION_DIR/assets"
if [ ! -f "$EXTENSION_DIR/assets/tfan-icon.svg" ]; then
    # Create simple icon SVG
    cat > "$EXTENSION_DIR/assets/tfan-icon.svg" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
  <circle cx="16" cy="16" r="14" fill="#667eea" stroke="#764ba2" stroke-width="2"/>
  <path d="M 12 12 L 20 16 L 12 20 Z" fill="white"/>
  <circle cx="16" cy="16" r="3" fill="#fbbf24"/>
</svg>
EOF
fi

if [ ! -f "$EXTENSION_DIR/assets/tfan-logo.svg" ]; then
    cp "$EXTENSION_DIR/assets/tfan-icon.svg" "$EXTENSION_DIR/assets/tfan-logo.svg"
fi

echo -e "${BLUE}Installing main application...${NC}"
cp app/tfan_gnome.py "$APP_DIR/"
chmod +x "$APP_DIR/tfan_gnome.py"

# Create launcher script
cat > "$BIN_DIR/tfan-gnome" << EOF
#!/bin/bash
cd "$APP_DIR"
python3 tfan_gnome.py "\$@"
EOF
chmod +x "$BIN_DIR/tfan-gnome"

echo -e "${BLUE}Installing desktop entry...${NC}"
cp com.quanta.tfan.desktop "$DESKTOP_DIR/"

echo -e "${BLUE}Updating desktop database...${NC}"
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Restart GNOME Shell (Alt+F2, type 'r', press Enter)"
echo "     Or log out and log back in"
echo "  2. Enable the extension:"
echo "     gnome-extensions enable tfan@quanta-meis-nib-cis"
echo "  3. Launch the app:"
echo "     tfan-gnome"
echo ""
echo "The T-FAN indicator should appear in your top bar!"
