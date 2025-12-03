# ARA Cockpit UI Guide

The ARA Cockpit is a GTK-based dashboard interface for monitoring and controlling the ARA system.

## Overview

![Cockpit Storage View](screenshots/cockpit-storage.png)

## Navigation Panels

The cockpit features 8 main navigation tiles arranged in a 2x4 grid:

| Panel | Description |
|-------|-------------|
| **ARA** | Main ARA system status and controls |
| **KITTEN** | Kitten companion system status |
| **CHAT** | Chat interface and conversation history |
| **GPU** | GPU utilization and temperature monitoring |
| **CPU/RAM** | CPU and memory usage statistics |
| **NETWORK** | Network connectivity and traffic |
| **STORAGE** | Disk usage and storage analytics |
| **NEURAL** | Neural network status and inference metrics |

## Status Bar

The bottom status bar displays real-time system information:
- **ARA Status**: ONLINE/OFFLINE indicator
- **KITTEN Status**: Companion system state
- **GPU Temperature**: Current GPU temperature
- **System Time**: Current time display

## Panel Views

### Storage Panel

![Storage Panel](screenshots/cockpit-storage.png)

The storage panel displays disk usage for mounted filesystems:
- Mount point path
- Used / Total space
- Usage percentage with progress bar

### Network Panel

![Network Panel](screenshots/cockpit-network-1.png)

The network panel shows connectivity status and traffic statistics.

## Theme

The cockpit uses a dark blue gradient theme with:
- Purple-to-blue gradient tiles
- Cyan accent color for selected items
- Monospace fonts for data display
- Subtle rounded corners on UI elements

## Source Files

- `gnome-tfan/app/cockpit_hud.py` - Main cockpit application
- `gnome-tfan/app/cockpit_theme.css` - CSS styling
- `gnome-tfan/assets/` - Media assets and backgrounds
