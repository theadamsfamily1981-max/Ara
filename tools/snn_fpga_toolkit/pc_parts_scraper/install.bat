@echo off
REM PC Parts Scraper - Easy Installation Script (Windows)
REM
REM This script will:
REM - Check Python version
REM - Create virtual environment
REM - Install all dependencies
REM - Set up the database
REM - Test the installation

setlocal enabledelayedexpansion

echo ==========================================
echo   PC Parts Scraper - Installation
echo ==========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is installed
echo [*] Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [+] Found Python %PYTHON_VERSION%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check if Python version is 3.8+
if %PYTHON_MAJOR% LSS 3 (
    echo [X] Python 3.8 or higher is required!
    echo You have Python %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 8 (
    echo [X] Python 3.8 or higher is required!
    echo You have Python %PYTHON_VERSION%
    pause
    exit /b 1
)

REM Create virtual environment
echo [*] Creating virtual environment...
if exist venv (
    echo [!] Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [X] Failed to create virtual environment
    pause
    exit /b 1
)
echo [+] Virtual environment created

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
echo [+] Virtual environment activated

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip > nul 2>&1
echo [+] pip upgraded

REM Install dependencies
echo [*] Installing Python dependencies (this may take a few minutes)...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Failed to install some packages
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [+] All Python packages installed successfully

REM Install Playwright browsers (optional)
echo.
echo [*] Installing Playwright browsers for JavaScript-heavy sites...
echo [!] This downloads Chrome/Firefox (~300MB). Skip? (Y/N, default N)
choice /t 10 /d N /n >nul
if errorlevel 2 (
    echo [!] Skipped Playwright (install later with: playwright install)
) else (
    playwright install chromium
    if errorlevel 1 (
        echo [!] Playwright installation failed (optional - you can skip this)
    ) else (
        echo [+] Playwright browsers installed
    )
)

REM Create directories
echo [*] Creating data directories...
if not exist data mkdir data
if not exist config mkdir config
echo [+] Directories created

REM Initialize database
echo [*] Initializing database...
python -c "from utils.database import DatabaseManager; db = DatabaseManager('data/pc_parts.db'); print('Database initialized!')"
if errorlevel 1 (
    echo [X] Database initialization failed
) else (
    echo [+] Database initialized
)

REM Done!
echo.
echo ==========================================
echo [+] Installation Complete!
echo ==========================================
echo.
echo PC Parts Scraper is ready to use!
echo.
echo Quick Start Options:
echo.
echo   1. Web GUI (Easiest - Recommended):
echo      run_gui.bat
echo      Then open: http://localhost:8501
echo.
echo   2. Command Line:
echo      venv\Scripts\activate    (Activate environment)
echo      python main.py scrape --all    (Run scraper)
echo      python main.py rare    (View rare finds)
echo.
echo   3. Test Installation:
echo      python main.py stats    (Check database stats)
echo.
echo Configuration:
echo   Edit config\settings.yaml to customize keywords and alerts
echo.
echo Need help? Check README.md
echo.
pause
