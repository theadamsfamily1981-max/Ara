#!/bin/bash
# Launch the PC Parts Scraper Web GUI

echo "Starting PC Parts Scraper Web GUI..."
echo "The interface will open in your browser at http://localhost:8501"
echo ""

cd "$(dirname "$0")"
streamlit run web_gui.py
