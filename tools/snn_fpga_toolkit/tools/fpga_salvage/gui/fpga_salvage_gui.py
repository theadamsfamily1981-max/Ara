#!/usr/bin/env python3
"""
FPGA Salvage Tool - Web GUI
============================

Easy-to-use web interface for repurposing mining and ATCA FPGAs.

Features:
- Step-by-step wizard
- Real-time progress updates
- Hardware auto-detection
- PMIC voltage tuning with sliders
- Safety warnings
- Log viewer

Usage:
    sudo python3 fpga_salvage_gui.py
    Then open: http://localhost:5000

Author: SNN Kernel Team
License: GPL-3.0
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import subprocess
import threading
import time
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpga_salvage import FPGAVendor, FPGASalvage, JTAGInterface, detect_pmic

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fpga-salvage-secret-key-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
salvage_state = {
    'running': False,
    'current_step': 'idle',
    'progress': 0,
    'logs': [],
    'detected_hardware': None,
    'voltage_info': None
}

# ============================================================================
# Logging to Web UI
# ============================================================================

def log_to_ui(message, level='info'):
    """Send log message to web UI"""
    global salvage_state
    log_entry = {
        'timestamp': time.strftime('%H:%M:%S'),
        'level': level,
        'message': message
    }
    salvage_state['logs'].append(log_entry)
    socketio.emit('log', log_entry)

# ============================================================================
# Hardware Detection
# ============================================================================

def detect_hardware():
    """Auto-detect FPGA hardware and JTAG adapter"""
    log_to_ui("Detecting hardware...", 'info')

    detected = {
        'jtag_adapter': None,
        'fpgas': [],
        'pmics': []
    }

    # Detect JTAG adapter (USB device)
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if 'FTDI' in result.stdout or 'ftdi' in result.stdout.lower():
            detected['jtag_adapter'] = 'FTDI FT2232H (Generic JTAG)'
            log_to_ui("✓ Found FTDI JTAG adapter", 'success')
        elif '09fb' in result.stdout:  # Intel USB-Blaster
            detected['jtag_adapter'] = 'Intel USB-Blaster'
            log_to_ui("✓ Found Intel USB-Blaster", 'success')
        elif 'Xilinx' in result.stdout:
            detected['jtag_adapter'] = 'Xilinx Platform Cable'
            log_to_ui("✓ Found Xilinx Platform Cable", 'success')
        else:
            log_to_ui("⚠ No JTAG adapter detected", 'warning')
    except Exception as e:
        log_to_ui(f"Error detecting JTAG: {e}", 'error')

    # Detect PMICs on I2C buses
    for bus in range(0, 4):  # Check I2C buses 0-3
        try:
            pmic = detect_pmic(bus)
            if pmic:
                detected['pmics'].append({
                    'bus': bus,
                    'type': pmic.__class__.__name__,
                    'address': f"0x{pmic.i2c.addr:02x}"
                })
                log_to_ui(f"✓ Found PMIC: {pmic.__class__.__name__} on bus {bus}", 'success')
        except:
            pass

    # Try to detect FPGAs via OpenOCD (requires JTAG connection)
    # This is more complex, skip for now in auto-detection

    log_to_ui("Hardware detection complete", 'info')
    return detected

# ============================================================================
# Salvage Operations
# ============================================================================

def run_salvage(vendor_str, skip_erase):
    """Run salvage operation in background thread"""
    global salvage_state

    try:
        salvage_state['running'] = True
        salvage_state['current_step'] = 'initializing'
        salvage_state['progress'] = 10

        # Map vendor string to enum
        vendor_map = {
            'stratix10': FPGAVendor.INTEL_STRATIX10,
            'virtex': FPGAVendor.XILINX_VIRTEX_ULTRASCALE,
            'kintex': FPGAVendor.XILINX_KINTEX_ULTRASCALE,
        }

        vendor = vendor_map.get(vendor_str)
        if not vendor:
            log_to_ui(f"Unknown vendor: {vendor_str}", 'error')
            return

        log_to_ui(f"Starting salvage for {vendor.value}...", 'info')

        # Create salvage instance
        salvage = FPGASalvage(vendor, skip_erase=skip_erase)

        # Step 1: JTAG Connection
        salvage_state['current_step'] = 'jtag_test'
        salvage_state['progress'] = 20
        socketio.emit('status', salvage_state)

        if not salvage.jtag.test_connection():
            log_to_ui("JTAG connection failed!", 'error')
            return

        log_to_ui("✓ JTAG connection successful", 'success')

        # Step 2: Flash Erase (if not skipped)
        if not skip_erase:
            salvage_state['current_step'] = 'flash_erase'
            salvage_state['progress'] = 40
            socketio.emit('status', salvage_state)

            log_to_ui("⚠ Erasing proprietary flash...", 'warning')
            time.sleep(2)  # Give user time to see warning

            if salvage.jtag.erase_flash():
                log_to_ui("✓ Flash erased successfully", 'success')
            else:
                log_to_ui("⚠ Flash erase failed (continuing anyway)", 'warning')

        # Step 3: Program Bitstream
        salvage_state['current_step'] = 'program_bitstream'
        salvage_state['progress'] = 60
        socketio.emit('status', salvage_state)

        bitstream = salvage._get_diagnostic_bitstream()
        if bitstream.exists():
            log_to_ui(f"Programming bitstream: {bitstream.name}", 'info')
            if salvage.jtag.program_bitstream(bitstream):
                log_to_ui("✓ Bitstream programmed", 'success')
            else:
                log_to_ui("⚠ Bitstream programming failed", 'warning')
        else:
            log_to_ui("⚠ No diagnostic bitstream found (skipping)", 'warning')

        # Step 4: Diagnostics
        salvage_state['current_step'] = 'diagnostics'
        salvage_state['progress'] = 80
        socketio.emit('status', salvage_state)

        log_to_ui("Running hardware diagnostics...", 'info')

        # Run diagnostic tests
        salvage.diag_core = DiagnosticCore(salvage.jtag)

        thermal_ok = salvage.diag_core.check_thermals()
        power_ok = salvage.diag_core.check_power_rails()
        memory_ok = salvage.diag_core.check_memory()

        # Step 5: Complete
        salvage_state['current_step'] = 'complete'
        salvage_state['progress'] = 100
        socketio.emit('status', salvage_state)

        if thermal_ok and power_ok and memory_ok:
            log_to_ui("✅ SALVAGE COMPLETE - Hardware is ready for AI workloads!", 'success')
        else:
            log_to_ui("⚠ Salvage complete with warnings - check diagnostics", 'warning')

    except Exception as e:
        log_to_ui(f"ERROR: {str(e)}", 'error')
        salvage_state['current_step'] = 'error'
    finally:
        salvage_state['running'] = False
        socketio.emit('status', salvage_state)

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current salvage status"""
    return jsonify(salvage_state)

@app.route('/api/detect')
def api_detect():
    """Detect hardware"""
    detected = detect_hardware()
    salvage_state['detected_hardware'] = detected
    return jsonify(detected)

@app.route('/api/start_salvage', methods=['POST'])
def start_salvage():
    """Start salvage operation"""
    data = request.json
    vendor = data.get('vendor', 'stratix10')
    skip_erase = data.get('skip_erase', False)

    if salvage_state['running']:
        return jsonify({'error': 'Salvage already running'}), 400

    # Clear logs
    salvage_state['logs'] = []

    # Start in background thread
    thread = threading.Thread(target=run_salvage, args=(vendor, skip_erase))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/voltage/read', methods=['POST'])
def read_voltage():
    """Read PMIC voltage"""
    data = request.json
    bus = data.get('bus', 0)

    try:
        pmic = detect_pmic(bus)
        if not pmic:
            return jsonify({'error': 'No PMIC detected'}), 404

        voltage = pmic.read_voltage()
        current = pmic.read_current() if hasattr(pmic, 'read_current') else None
        temp = pmic.read_temperature() if hasattr(pmic, 'read_temperature') else None

        info = {
            'voltage': voltage,
            'current': current,
            'temperature': temp,
            'power': voltage * current if (voltage and current) else None
        }

        salvage_state['voltage_info'] = info
        return jsonify(info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voltage/set', methods=['POST'])
def set_voltage():
    """Set PMIC voltage"""
    data = request.json
    bus = data.get('bus', 0)
    voltage = data.get('voltage')

    if not voltage or not (0.5 <= voltage <= 1.0):
        return jsonify({'error': 'Invalid voltage (must be 0.5-1.0V)'}), 400

    try:
        pmic = detect_pmic(bus)
        if not pmic:
            return jsonify({'error': 'No PMIC detected'}), 404

        log_to_ui(f"Setting voltage to {voltage:.3f}V...", 'info')

        if pmic.set_voltage(voltage):
            log_to_ui(f"✓ Voltage set to {voltage:.3f}V", 'success')
            return jsonify({'success': True})
        else:
            log_to_ui("✗ Voltage programming failed", 'error')
            return jsonify({'error': 'Programming failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SocketIO Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    emit('status', salvage_state)
    emit('logs', {'logs': salvage_state['logs']})

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import os

    if os.geteuid() != 0:
        print("=" * 70)
        print("ERROR: This GUI requires root access for JTAG and I2C operations")
        print("=" * 70)
        print("\nPlease run with sudo:")
        print(f"  sudo python3 {sys.argv[0]}")
        print("\nThen open your browser to: http://localhost:5000")
        print("=" * 70)
        sys.exit(1)

    print("=" * 70)
    print("FPGA Salvage Tool - Web GUI")
    print("=" * 70)
    print("\n✓ Server starting...")
    print("\nOpen your browser to:")
    print("  http://localhost:5000")
    print("\nOr from another computer:")
    print(f"  http://{subprocess.check_output(['hostname', '-I']).decode().split()[0]}:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    print()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
