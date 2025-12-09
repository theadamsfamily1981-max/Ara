@echo off
REM Launch the PC Parts Scraper Web GUI (Windows)

echo Starting PC Parts Scraper Web GUI...
echo The interface will open in your browser at http://localhost:8501
echo.

cd /d "%~dp0"
streamlit run web_gui.py
pause
