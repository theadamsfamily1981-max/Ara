#!/bin/bash
#
# PC Parts Scraper - Easy Installation Script (Linux/Mac)
#
# This script will:
# - Check Python version
# - Create virtual environment
# - Install all dependencies
# - Set up the database
# - Test the installation
#

set -e  # Exit on error

echo "=========================================="
echo "  PC Parts Scraper - Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Change to script directory
cd "$(dirname "$0")"

# Step 1: Check Python
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Found Python $PYTHON_VERSION"

# Check Python version (need 3.8+)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required!"
    echo "You have Python $PYTHON_VERSION"
    echo "Please upgrade: https://www.python.org/"
    exit 1
fi

# Step 2: Create virtual environment
print_status "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
print_success "Virtual environment created"

# Step 3: Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 4: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Step 5: Install dependencies
print_status "Installing Python dependencies (this may take a few minutes)..."
echo ""
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "All Python packages installed successfully"
else
    print_error "Failed to install some packages"
    echo "Try running: pip install -r requirements.txt"
    exit 1
fi

# Step 6: Install Playwright browsers (optional)
echo ""
print_status "Installing Playwright browsers for JavaScript-heavy sites..."
print_warning "This downloads Chrome/Firefox (~300MB). Skip? [y/N]"
read -t 10 -n 1 -r SKIP_PLAYWRIGHT || SKIP_PLAYWRIGHT="n"
echo ""

if [[ ! $SKIP_PLAYWRIGHT =~ ^[Yy]$ ]]; then
    playwright install chromium
    if [ $? -eq 0 ]; then
        print_success "Playwright browsers installed"
    else
        print_warning "Playwright installation failed (optional - you can skip this)"
    fi
else
    print_warning "Skipped Playwright (you can install later with: playwright install)"
fi

# Step 7: Create necessary directories
print_status "Creating data directories..."
mkdir -p data
mkdir -p config
print_success "Directories created"

# Step 8: Initialize database
print_status "Initializing database..."
python3 << EOF
from utils.database import DatabaseManager
db = DatabaseManager('data/pc_parts.db')
print("Database initialized successfully!")
EOF

if [ $? -eq 0 ]; then
    print_success "Database initialized"
else
    print_error "Database initialization failed"
fi

# Step 9: Make scripts executable
print_status "Making scripts executable..."
chmod +x run_gui.sh
chmod +x main.py
print_success "Scripts are now executable"

# Done!
echo ""
echo "=========================================="
print_success "Installation Complete!"
echo "=========================================="
echo ""
echo "🎉 PC Parts Scraper is ready to use!"
echo ""
echo "Quick Start Options:"
echo ""
echo "  1. Web GUI (Easiest - Recommended):"
echo "     ${GREEN}./run_gui.sh${NC}"
echo "     Then open: http://localhost:8501"
echo ""
echo "  2. Command Line:"
echo "     ${GREEN}source venv/bin/activate${NC}  # Activate environment"
echo "     ${GREEN}python main.py scrape --all${NC}  # Run scraper"
echo "     ${GREEN}python main.py rare${NC}  # View rare finds"
echo ""
echo "  3. Test Installation:"
echo "     ${GREEN}python main.py stats${NC}  # Check database stats"
echo ""
echo "Configuration:"
echo "  Edit config/settings.yaml to customize keywords and alerts"
echo ""
echo "Need help? Check README.md"
echo ""
