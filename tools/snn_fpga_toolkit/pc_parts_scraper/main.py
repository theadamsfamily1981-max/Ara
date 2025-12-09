#!/usr/bin/env python3
"""
PC Parts Scraper - Main Entry Point

An advanced web scraper for finding rare and forgotten PC parts across:
- Government surplus auctions
- Liquidation sales
- Used marketplaces
- Discount retailers
- Electronics recyclers

Usage:
    python main.py scrape --all
    python main.py search -k "voodoo" -k "3dfx"
    python main.py rare
    python main.py stats
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main

if __name__ == '__main__':
    main()
