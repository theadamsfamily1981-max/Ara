"""
Rust/Python Struct Compatibility Verifier
==========================================

This module verifies that the Rust `SomaticState` struct and the Python
HAL memory layout are byte-for-byte identical. Run this to catch any
layout mismatches before they cause hard-to-debug bugs.

Usage:
    python -m banos.hal.struct_compat
"""

import struct
import ctypes
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# Expected Memory Layout (Must match somatic.rs exactly)
# =============================================================================

MAGIC = 0x0AFA5011  # Note: Must match Rust somatic.rs
VERSION = 2
SHM_SIZE = 4096

# Section sizes (from Rust compile-time assertions)
EXPECTED_SIZES = {
    'SomaticHeader': 32,
    'AffectiveState': 32,
    'SomaticSensors': 64,
    'HardwareDiag': 32,
    'SystemMetrics': 96,
    'ControlFlags': 32,
    'CouncilState': 32,
    'SomaticState': 320,
}

# Section offsets
EXPECTED_OFFSETS = {
    'header': 0x0000,
    'affect': 0x0020,
    'sensors': 0x0040,
    'hardware': 0x0080,
    'metrics': 0x00A0,
    'control': 0x0100,
    'council': 0x0120,
}


# =============================================================================
# CTypes Struct Definitions (Mirror of Rust structs)
# =============================================================================

class SomaticHeader(ctypes.Structure):
    """Header section: 0x0000-0x001F (32 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('magic', ctypes.c_uint32),        # 0x0000
        ('version', ctypes.c_uint32),      # 0x0004
        ('timestamp_ns', ctypes.c_uint64), # 0x0008
        ('heart_rate', ctypes.c_uint32),   # 0x0010
        ('dream_state', ctypes.c_uint32),  # 0x0014
        ('_pad', ctypes.c_uint8 * 8),      # 0x0018-0x001F
    ]


class AffectiveState(ctypes.Structure):
    """Affective state section: 0x0020-0x003F (32 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('pad_p', ctypes.c_float),         # 0x0020
        ('pad_a', ctypes.c_float),         # 0x0024
        ('pad_d', ctypes.c_float),         # 0x0028
        ('quadrant', ctypes.c_uint8),      # 0x002C
        ('sched_mode', ctypes.c_uint8),    # 0x002D
        ('_reserved', ctypes.c_uint16),    # 0x002E
        ('emotion_x', ctypes.c_float),     # 0x0030
        ('emotion_y', ctypes.c_float),     # 0x0034
        ('_pad', ctypes.c_uint8 * 8),      # 0x0038-0x003F
    ]


class SomaticSensors(ctypes.Structure):
    """Somatic sensors section: 0x0040-0x007F (64 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('pain_raw', ctypes.c_uint32),     # 0x0040
        ('pain_weber', ctypes.c_float),    # 0x0044
        ('entropy', ctypes.c_float),       # 0x0048
        ('flow_x', ctypes.c_float),        # 0x004C
        ('flow_y', ctypes.c_float),        # 0x0050
        ('flow_mag', ctypes.c_float),      # 0x0054
        ('audio_rms', ctypes.c_float),     # 0x0058
        ('audio_pitch', ctypes.c_float),   # 0x005C
        ('_pad', ctypes.c_uint8 * 32),     # 0x0060-0x007F
    ]


class HardwareDiag(ctypes.Structure):
    """Hardware diagnostics section: 0x0080-0x009F (32 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('active_neurons', ctypes.c_uint32),  # 0x0080
        ('total_spikes', ctypes.c_uint32),    # 0x0084
        ('fabric_temp', ctypes.c_uint32),     # 0x0088
        ('thermal_limit', ctypes.c_uint8),    # 0x008C
        ('fabric_online', ctypes.c_uint8),    # 0x008D
        ('dream_active', ctypes.c_uint8),     # 0x008E
        ('_reserved', ctypes.c_uint8),        # 0x008F
        ('_pad', ctypes.c_uint8 * 16),        # 0x0090-0x009F
    ]


class SystemMetrics(ctypes.Structure):
    """System metrics section: 0x00A0-0x00FF (96 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('cpu_temp', ctypes.c_float),      # 0x00A0
        ('gpu_temp', ctypes.c_float),      # 0x00A4
        ('cpu_load', ctypes.c_float),      # 0x00A8
        ('gpu_load', ctypes.c_float),      # 0x00AC
        ('ram_used_pct', ctypes.c_float),  # 0x00B0
        ('vram_used_pct', ctypes.c_float), # 0x00B4
        ('power_draw_w', ctypes.c_float),  # 0x00B8
        ('_pad', ctypes.c_uint8 * 68),     # 0x00BC-0x00FF
    ]


class ControlFlags(ctypes.Structure):
    """Control flags section: 0x0100-0x011F (32 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('avatar_mode', ctypes.c_uint8),      # 0x0100
        ('sim_detail', ctypes.c_uint8),       # 0x0101
        ('force_sleep', ctypes.c_uint8),      # 0x0102
        ('emergency_stop', ctypes.c_uint8),   # 0x0103
        ('critical_temp', ctypes.c_float),    # 0x0104
        ('_pad', ctypes.c_uint8 * 24),        # 0x0108-0x011F
    ]


class CouncilState(ctypes.Structure):
    """Council state section: 0x0120-0x013F (32 bytes)"""
    _pack_ = 1
    _fields_ = [
        ('mask', ctypes.c_uint32),         # 0x0120
        ('stress', ctypes.c_float),        # 0x0124
        ('muse_x', ctypes.c_float),        # 0x0128
        ('muse_y', ctypes.c_float),        # 0x012C
        ('censor_x', ctypes.c_float),      # 0x0130
        ('censor_y', ctypes.c_float),      # 0x0134
        ('scribe_x', ctypes.c_float),      # 0x0138
        ('scribe_y', ctypes.c_float),      # 0x013C
    ]


class SomaticState(ctypes.Structure):
    """Complete somatic state - maps to first 320 bytes of shared memory"""
    _pack_ = 1
    _fields_ = [
        ('header', SomaticHeader),
        ('affect', AffectiveState),
        ('sensors', SomaticSensors),
        ('hardware', HardwareDiag),
        ('metrics', SystemMetrics),
        ('control', ControlFlags),
        ('council', CouncilState),
    ]


# =============================================================================
# Verification Functions
# =============================================================================

def verify_sizes() -> List[str]:
    """Verify struct sizes match Rust definitions."""
    errors = []

    structs = {
        'SomaticHeader': SomaticHeader,
        'AffectiveState': AffectiveState,
        'SomaticSensors': SomaticSensors,
        'HardwareDiag': HardwareDiag,
        'SystemMetrics': SystemMetrics,
        'ControlFlags': ControlFlags,
        'CouncilState': CouncilState,
        'SomaticState': SomaticState,
    }

    for name, struct_cls in structs.items():
        actual = ctypes.sizeof(struct_cls)
        expected = EXPECTED_SIZES[name]
        if actual != expected:
            errors.append(f"{name}: size={actual}, expected={expected}")

    return errors


def verify_offsets() -> List[str]:
    """Verify field offsets within SomaticState."""
    errors = []

    expected = {
        'header': 0x0000,
        'affect': 0x0020,
        'sensors': 0x0040,
        'hardware': 0x0080,
        'metrics': 0x00A0,
        'control': 0x0100,
        'council': 0x0120,
    }

    for field_name, expected_offset in expected.items():
        actual_offset = getattr(SomaticState, field_name).offset
        if actual_offset != expected_offset:
            errors.append(
                f"SomaticState.{field_name}: offset=0x{actual_offset:04X}, "
                f"expected=0x{expected_offset:04X}"
            )

    return errors


def verify_field_offsets_detail() -> List[str]:
    """Verify individual field offsets within each section."""
    errors = []

    # Header field offsets (relative to header start at 0x0000)
    header_fields = [
        ('magic', 0x0000),
        ('version', 0x0004),
        ('timestamp_ns', 0x0008),
        ('heart_rate', 0x0010),
        ('dream_state', 0x0014),
    ]
    for field_name, expected in header_fields:
        actual = getattr(SomaticHeader, field_name).offset
        if actual != expected:
            errors.append(f"SomaticHeader.{field_name}: 0x{actual:04X} != 0x{expected:04X}")

    # Affective field offsets (relative to affect start at 0x0020)
    affect_fields = [
        ('pad_p', 0x0000),   # 0x0020 - 0x0020
        ('pad_a', 0x0004),
        ('pad_d', 0x0008),
        ('quadrant', 0x000C),
        ('sched_mode', 0x000D),
        ('emotion_x', 0x0010),
        ('emotion_y', 0x0014),
    ]
    for field_name, expected in affect_fields:
        actual = getattr(AffectiveState, field_name).offset
        if actual != expected:
            errors.append(f"AffectiveState.{field_name}: 0x{actual:04X} != 0x{expected:04X}")

    # Sensors field offsets (relative to sensors start at 0x0040)
    sensor_fields = [
        ('pain_raw', 0x0000),
        ('pain_weber', 0x0004),
        ('entropy', 0x0008),
        ('flow_x', 0x000C),
        ('flow_y', 0x0010),
        ('flow_mag', 0x0014),
        ('audio_rms', 0x0018),
        ('audio_pitch', 0x001C),
    ]
    for field_name, expected in sensor_fields:
        actual = getattr(SomaticSensors, field_name).offset
        if actual != expected:
            errors.append(f"SomaticSensors.{field_name}: 0x{actual:04X} != 0x{expected:04X}")

    # Council field offsets (relative to council start at 0x0120)
    council_fields = [
        ('mask', 0x0000),
        ('stress', 0x0004),
        ('muse_x', 0x0008),
        ('muse_y', 0x000C),
        ('censor_x', 0x0010),
        ('censor_y', 0x0014),
        ('scribe_x', 0x0018),
        ('scribe_y', 0x001C),
    ]
    for field_name, expected in council_fields:
        actual = getattr(CouncilState, field_name).offset
        if actual != expected:
            errors.append(f"CouncilState.{field_name}: 0x{actual:04X} != 0x{expected:04X}")

    return errors


def create_test_state() -> SomaticState:
    """Create a test state with known values."""
    state = SomaticState()

    # Header
    state.header.magic = MAGIC
    state.header.version = VERSION
    state.header.timestamp_ns = 1234567890123456789
    state.header.heart_rate = 60
    state.header.dream_state = 0

    # Affective
    state.affect.pad_p = 0.5
    state.affect.pad_a = -0.3
    state.affect.pad_d = 0.7
    state.affect.quadrant = 1
    state.affect.sched_mode = 0
    state.affect.emotion_x = 0.1
    state.affect.emotion_y = 0.2

    # Sensors
    state.sensors.pain_raw = 1000
    state.sensors.pain_weber = 0.3
    state.sensors.entropy = 0.2
    state.sensors.flow_x = 0.1
    state.sensors.flow_y = -0.1
    state.sensors.flow_mag = 0.14
    state.sensors.audio_rms = 0.5
    state.sensors.audio_pitch = 440.0

    # Hardware
    state.hardware.active_neurons = 1024
    state.hardware.total_spikes = 1000000
    state.hardware.fabric_temp = 45000  # mC
    state.hardware.thermal_limit = 0
    state.hardware.fabric_online = 1
    state.hardware.dream_active = 0

    # Metrics
    state.metrics.cpu_temp = 55.0
    state.metrics.gpu_temp = 65.0
    state.metrics.cpu_load = 0.3
    state.metrics.gpu_load = 0.8
    state.metrics.ram_used_pct = 0.6
    state.metrics.vram_used_pct = 0.9
    state.metrics.power_draw_w = 450.0

    # Control
    state.control.avatar_mode = 3
    state.control.sim_detail = 1
    state.control.force_sleep = 0
    state.control.emergency_stop = 0
    state.control.critical_temp = 95.0

    # Council
    state.council.mask = 15  # All active
    state.council.stress = 0.2
    state.council.muse_x = 0.7
    state.council.muse_y = 0.7
    state.council.censor_x = 0.3
    state.council.censor_y = 0.7
    state.council.scribe_x = 0.5
    state.council.scribe_y = 0.3

    return state


def state_to_bytes(state: SomaticState) -> bytes:
    """Convert SomaticState to bytes."""
    return bytes(state)


def bytes_to_state(data: bytes) -> SomaticState:
    """Convert bytes to SomaticState."""
    return SomaticState.from_buffer_copy(data)


def print_layout():
    """Print the complete memory layout."""
    print("=" * 70)
    print("SOMATIC STATE MEMORY LAYOUT (Python ctypes)")
    print("=" * 70)
    print(f"Total size: {ctypes.sizeof(SomaticState)} bytes")
    print()

    sections = [
        ('header', SomaticHeader, 0x0000),
        ('affect', AffectiveState, 0x0020),
        ('sensors', SomaticSensors, 0x0040),
        ('hardware', HardwareDiag, 0x0080),
        ('metrics', SystemMetrics, 0x00A0),
        ('control', ControlFlags, 0x0100),
        ('council', CouncilState, 0x0120),
    ]

    for name, struct_cls, base_offset in sections:
        print(f"\n{name.upper()} (0x{base_offset:04X} - 0x{base_offset + ctypes.sizeof(struct_cls) - 1:04X})")
        print("-" * 50)
        for field_name, field_type in struct_cls._fields_:
            if field_name.startswith('_'):
                continue
            offset = getattr(struct_cls, field_name).offset
            size = ctypes.sizeof(field_type)
            abs_offset = base_offset + offset
            print(f"  0x{abs_offset:04X}  {field_name:20s}  ({size} bytes)")


def run_verification() -> bool:
    """Run all verification checks."""
    print("=" * 70)
    print("RUST/PYTHON STRUCT COMPATIBILITY VERIFICATION")
    print("=" * 70)

    all_passed = True

    # Size verification
    print("\n[1] Verifying struct sizes...")
    size_errors = verify_sizes()
    if size_errors:
        print("  FAILED:")
        for e in size_errors:
            print(f"    - {e}")
        all_passed = False
    else:
        print("  PASSED: All struct sizes match Rust definitions")

    # Section offset verification
    print("\n[2] Verifying section offsets...")
    offset_errors = verify_offsets()
    if offset_errors:
        print("  FAILED:")
        for e in offset_errors:
            print(f"    - {e}")
        all_passed = False
    else:
        print("  PASSED: All section offsets correct")

    # Field offset verification
    print("\n[3] Verifying field offsets...")
    field_errors = verify_field_offsets_detail()
    if field_errors:
        print("  FAILED:")
        for e in field_errors:
            print(f"    - {e}")
        all_passed = False
    else:
        print("  PASSED: All field offsets correct")

    # Round-trip test
    print("\n[4] Verifying round-trip serialization...")
    try:
        state = create_test_state()
        data = state_to_bytes(state)
        state2 = bytes_to_state(data)

        if (state.header.magic == state2.header.magic and
            state.affect.pad_p == state2.affect.pad_p and
            state.sensors.pain_weber == state2.sensors.pain_weber and
            state.council.stress == state2.council.stress):
            print("  PASSED: Round-trip serialization works")
        else:
            print("  FAILED: Round-trip values don't match")
            all_passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED - Rust and Python structs are compatible!")
    else:
        print("SOME CHECKS FAILED - Fix layout mismatches before proceeding")
    print("=" * 70)

    return all_passed


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys

    if '--layout' in sys.argv:
        print_layout()
    else:
        success = run_verification()
        sys.exit(0 if success else 1)
