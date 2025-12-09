# PC Parts Scraper - PowerShell Installation Script (Windows)
#
# This script will:
# - Check Python version
# - Create virtual environment
# - Install all dependencies
# - Set up the database
# - Test the installation
#
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  PC Parts Scraper - Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Functions for colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[✗] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

# Step 1: Check Python
Write-Status "Checking Python installation..."

try {
    $pythonVersion = python --version 2>&1 | Out-String
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        $patch = [int]$matches[3]

        Write-Success "Found Python $major.$minor.$patch"

        # Check version
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Error "Python 3.8 or higher is required!"
            Write-Host "You have Python $major.$minor.$patch"
            Write-Host "Download from: https://www.python.org/downloads/"
            Write-Host ""
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        throw "Could not parse Python version"
    }
} catch {
    Write-Error "Python is not installed or not in PATH!"
    Write-Host ""
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/"
    Write-Host "IMPORTANT: Check 'Add Python to PATH' during installation!"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 2: Create virtual environment
Write-Status "Creating virtual environment..."

if (Test-Path "venv") {
    Write-Warning "Virtual environment already exists. Removing old one..."
    Remove-Item -Recurse -Force venv
}

python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Success "Virtual environment created"

# Step 3: Activate virtual environment
Write-Status "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
Write-Success "Virtual environment activated"

# Step 4: Upgrade pip
Write-Status "Upgrading pip..."
python -m pip install --upgrade pip *> $null
Write-Success "pip upgraded"

# Step 5: Install dependencies
Write-Status "Installing Python dependencies (this may take a few minutes)..."
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Success "All Python packages installed successfully"
} else {
    Write-Error "Failed to install some packages"
    Write-Host "Try running: pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 6: Install Playwright (optional)
Write-Host ""
Write-Status "Installing Playwright browsers for JavaScript-heavy sites..."
Write-Warning "This downloads Chrome/Firefox (~300MB). Skip? (Y/N, default N)"

$timeout = New-TimeSpan -Seconds 10
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$skip = $false

while ($stopwatch.Elapsed -lt $timeout) {
    if ([Console]::KeyAvailable) {
        $key = [Console]::ReadKey($true)
        if ($key.KeyChar -eq 'Y' -or $key.KeyChar -eq 'y') {
            $skip = $true
            break
        }
    }
    Start-Sleep -Milliseconds 100
}

if (!$skip) {
    playwright install chromium
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Playwright browsers installed"
    } else {
        Write-Warning "Playwright installation failed (optional)"
    }
} else {
    Write-Host ""
    Write-Warning "Skipped Playwright (install later with: playwright install)"
}

# Step 7: Create directories
Write-Status "Creating data directories..."
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "config" | Out-Null
Write-Success "Directories created"

# Step 8: Initialize database
Write-Status "Initializing database..."

$initScript = @"
from utils.database import DatabaseManager
db = DatabaseManager('data/pc_parts.db')
print('Database initialized successfully!')
"@

$initScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Success "Database initialized"
} else {
    Write-Error "Database initialization failed"
}

# Done!
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Success "Installation Complete!"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 PC Parts Scraper is ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick Start Options:"
Write-Host ""
Write-Host "  1. Web GUI (Easiest - Recommended):" -ForegroundColor Yellow
Write-Host "     " -NoNewline
Write-Host ".\run_gui.bat" -ForegroundColor Green
Write-Host "     Then open: http://localhost:8501"
Write-Host ""
Write-Host "  2. Command Line:" -ForegroundColor Yellow
Write-Host "     " -NoNewline
Write-Host "venv\Scripts\Activate.ps1" -ForegroundColor Green -NoNewline
Write-Host "  # Activate environment"
Write-Host "     " -NoNewline
Write-Host "python main.py scrape --all" -ForegroundColor Green -NoNewline
Write-Host "  # Run scraper"
Write-Host "     " -NoNewline
Write-Host "python main.py rare" -ForegroundColor Green -NoNewline
Write-Host "  # View rare finds"
Write-Host ""
Write-Host "  3. Test Installation:" -ForegroundColor Yellow
Write-Host "     " -NoNewline
Write-Host "python main.py stats" -ForegroundColor Green -NoNewline
Write-Host "  # Check database stats"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Edit config\settings.yaml to customize keywords and alerts"
Write-Host ""
Write-Host "Need help? Check README.md"
Write-Host ""
Read-Host "Press Enter to exit"
