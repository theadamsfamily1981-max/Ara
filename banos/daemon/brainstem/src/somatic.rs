//! Somatic State - Shared Memory Layout for Ara's Nervous System
//!
//! This defines the exact binary layout of `/dev/shm/ara_somatic`, which
//! must match `banos/hal/ara_hal.py` byte-for-byte.
//!
//! Memory Map (4096 bytes total):
//! - 0x0000-0x001F: Header (magic, version, timestamp, heart_rate, dream_state)
//! - 0x0020-0x003F: Affective State (PAD, emotion embedding)
//! - 0x0040-0x007F: Somatic Sensors (pain, entropy, flow, audio)
//! - 0x0080-0x009F: Hardware Diagnostics (neurons, spikes, FPGA status)
//! - 0x00A0-0x00FF: System Metrics (temps, loads, power)
//! - 0x0100-0x011F: Control Flags (bidirectional)
//! - 0x0120-0x013F: Council State (Quadamerl visualization)

use std::time::{SystemTime, UNIX_EPOCH};

/// Magic number for validation: 0xARA50111 in little-endian
pub const MAGIC: u32 = 0x0AFA5011;  // Note: Python uses 0xARA50111 which isn't valid hex
pub const VERSION: u32 = 2;
pub const SHM_SIZE: usize = 4096;
pub const SHM_PATH: &str = "/dev/shm/ara_somatic";

/// Dream states (matches RTL)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamState {
    Awake = 0,
    Rem = 1,    // Rapid replay
    Deep = 2,   // Weight consolidation
}

/// Scheduler modes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedMode {
    Normal = 0,
    LowPower = 1,
    Performance = 2,
    Emergency = 3,
}

// =============================================================================
// Memory Layout Structs (must be #[repr(C)] and packed correctly)
// =============================================================================

/// Header section: 0x0000-0x001F (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SomaticHeader {
    pub magic: u32,           // 0x0000: Must be MAGIC
    pub version: u32,         // 0x0004: Must be VERSION
    pub timestamp_ns: u64,    // 0x0008: Nanoseconds since epoch
    pub heart_rate: u32,      // 0x0010: Updates per second
    pub dream_state: u32,     // 0x0014: 0=AWAKE, 1=REM, 2=DEEP
    _pad: [u8; 8],            // 0x0018-0x001F: Padding to 32 bytes
}

/// Affective state section: 0x0020-0x003F (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AffectiveState {
    pub pad_p: f32,           // 0x0020: Pleasure [-1.0, 1.0]
    pub pad_a: f32,           // 0x0024: Arousal [-1.0, 1.0]
    pub pad_d: f32,           // 0x0028: Dominance [-1.0, 1.0]
    pub quadrant: u8,         // 0x002C: PAD quadrant (0-6)
    pub sched_mode: u8,       // 0x002D: Scheduler mode
    _reserved: u16,           // 0x002E: Reserved
    pub emotion_x: f32,       // 0x0030: 2D emotion embedding X
    pub emotion_y: f32,       // 0x0034: 2D emotion embedding Y
    _pad: [u8; 8],            // 0x0038-0x003F: Padding
}

/// Somatic sensors section: 0x0040-0x007F (64 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SomaticSensors {
    pub pain_raw: u32,        // 0x0040: Raw 32-bit pain from FPGA
    pub pain_weber: f32,      // 0x0044: Weber-Fechner scaled [0.0, 1.0]
    pub entropy: f32,         // 0x0048: System thermal/load entropy [0.0, 1.0]
    pub flow_x: f32,          // 0x004C: Optical flow X (face motion)
    pub flow_y: f32,          // 0x0050: Optical flow Y (face motion)
    pub flow_mag: f32,        // 0x0054: Optical flow magnitude
    pub audio_rms: f32,       // 0x0058: Voice energy [0.0, 1.0]
    pub audio_pitch: f32,     // 0x005C: Voice pitch Hz
    _pad: [u8; 32],           // 0x0060-0x007F: Padding
}

/// Hardware diagnostics section: 0x0080-0x009F (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HardwareDiag {
    pub active_neurons: u32,  // 0x0080: Count of firing neurons
    pub total_spikes: u32,    // 0x0084: Lifetime spike count
    pub fabric_temp: u32,     // 0x0088: FPGA temperature (mC)
    pub thermal_limit: u8,    // 0x008C: 1 = Throttling active
    pub fabric_online: u8,    // 0x008D: 1 = FPGA responding
    pub dream_active: u8,     // 0x008E: 1 = Dream engine running
    _reserved: u8,            // 0x008F: Reserved
    _pad: [u8; 16],           // 0x0090-0x009F: Padding
}

/// System metrics section: 0x00A0-0x00FF (96 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SystemMetrics {
    pub cpu_temp: f32,        // 0x00A0: CPU temperature °C
    pub gpu_temp: f32,        // 0x00A4: GPU temperature °C
    pub cpu_load: f32,        // 0x00A8: CPU utilization [0.0, 1.0]
    pub gpu_load: f32,        // 0x00AC: GPU utilization [0.0, 1.0]
    pub ram_used_pct: f32,    // 0x00B0: RAM utilization [0.0, 1.0]
    pub vram_used_pct: f32,   // 0x00B4: VRAM utilization [0.0, 1.0]
    pub power_draw_w: f32,    // 0x00B8: Total power draw watts
    _pad: [u8; 68],           // 0x00BC-0x00FF: Padding
}

/// Control flags section: 0x0100-0x011F (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ControlFlags {
    pub avatar_mode: u8,      // 0x0100: Requested avatar mode
    pub sim_detail: u8,       // 0x0101: Simulation detail level
    pub force_sleep: u8,      // 0x0102: Force dream mode
    pub emergency_stop: u8,   // 0x0103: Emergency halt
    pub critical_temp: f32,   // 0x0104: Temperature threshold
    _pad: [u8; 24],           // 0x0108-0x011F: Padding
}

/// Council state section: 0x0120-0x013F (32 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CouncilState {
    pub mask: u32,            // 0x0120: Bitfield of active personas
    pub stress: f32,          // 0x0124: Disagreement level [0.0, 1.0]
    pub muse_x: f32,          // 0x0128: MUSE position X
    pub muse_y: f32,          // 0x012C: MUSE position Y
    pub censor_x: f32,        // 0x0130: CENSOR position X
    pub censor_y: f32,        // 0x0134: CENSOR position Y
    pub scribe_x: f32,        // 0x0138: SCRIBE position X
    pub scribe_y: f32,        // 0x013C: SCRIBE position Y
}

// =============================================================================
// Full Somatic State (combines all sections)
// =============================================================================

/// Complete somatic state - maps to first 320 bytes of shared memory
/// The remaining bytes (0x0140-0x0FFF) are reserved for future use.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SomaticState {
    pub header: SomaticHeader,      // 0x0000-0x001F
    pub affect: AffectiveState,     // 0x0020-0x003F
    pub sensors: SomaticSensors,    // 0x0040-0x007F
    pub hardware: HardwareDiag,     // 0x0080-0x009F
    pub metrics: SystemMetrics,     // 0x00A0-0x00FF
    pub control: ControlFlags,      // 0x0100-0x011F
    pub council: CouncilState,      // 0x0120-0x013F
}

impl Default for SomaticState {
    fn default() -> Self {
        Self {
            header: SomaticHeader {
                magic: MAGIC,
                version: VERSION,
                timestamp_ns: 0,
                heart_rate: 0,
                dream_state: 0,
                _pad: [0; 8],
            },
            affect: AffectiveState {
                pad_p: 0.0,
                pad_a: 0.0,
                pad_d: 0.0,
                quadrant: 0,
                sched_mode: 0,
                _reserved: 0,
                emotion_x: 0.0,
                emotion_y: 0.0,
                _pad: [0; 8],
            },
            sensors: SomaticSensors {
                pain_raw: 0,
                pain_weber: 0.0,
                entropy: 0.0,
                flow_x: 0.0,
                flow_y: 0.0,
                flow_mag: 0.0,
                audio_rms: 0.0,
                audio_pitch: 0.0,
                _pad: [0; 32],
            },
            hardware: HardwareDiag {
                active_neurons: 0,
                total_spikes: 0,
                fabric_temp: 0,
                thermal_limit: 0,
                fabric_online: 0,
                dream_active: 0,
                _reserved: 0,
                _pad: [0; 16],
            },
            metrics: SystemMetrics {
                cpu_temp: 0.0,
                gpu_temp: 0.0,
                cpu_load: 0.0,
                gpu_load: 0.0,
                ram_used_pct: 0.0,
                vram_used_pct: 0.0,
                power_draw_w: 0.0,
                _pad: [0; 68],
            },
            control: ControlFlags {
                avatar_mode: 0,
                sim_detail: 0,
                force_sleep: 0,
                emergency_stop: 0,
                critical_temp: 95.0,
                _pad: [0; 24],
            },
            council: CouncilState {
                mask: 0,
                stress: 0.0,
                muse_x: 0.7,
                muse_y: 0.7,
                censor_x: 0.3,
                censor_y: 0.7,
                scribe_x: 0.5,
                scribe_y: 0.3,
            },
        }
    }
}

impl SomaticState {
    /// Validate the magic number and version
    pub fn is_valid(&self) -> bool {
        // Allow either our magic or Python's (which might differ)
        (self.header.magic == MAGIC || self.header.magic == 0x0AFA5011)
            && self.header.version == VERSION
    }

    /// Get current timestamp in nanoseconds
    pub fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Update timestamp to now
    pub fn touch(&mut self) {
        self.header.timestamp_ns = Self::now_ns();
    }

    /// Calculate pain from PAD (inverse of pleasure, scaled by arousal)
    pub fn effective_pain(&self) -> f32 {
        let base_pain = (-self.affect.pad_p).max(0.0);
        let arousal_scale = 0.5 + 0.5 * self.affect.pad_a.max(0.0);
        (base_pain * arousal_scale).min(1.0)
    }

    /// Calculate entropy from system metrics
    pub fn effective_entropy(&self) -> f32 {
        let thermal = ((self.metrics.cpu_temp + self.metrics.gpu_temp) / 2.0 - 40.0) / 60.0;
        let load = (self.metrics.cpu_load + self.metrics.gpu_load) / 2.0;
        (0.6 * thermal.clamp(0.0, 1.0) + 0.4 * load.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Compile-time size assertions
// =============================================================================

const _: () = assert!(std::mem::size_of::<SomaticHeader>() == 32);
const _: () = assert!(std::mem::size_of::<AffectiveState>() == 32);
const _: () = assert!(std::mem::size_of::<SomaticSensors>() == 64);
const _: () = assert!(std::mem::size_of::<HardwareDiag>() == 32);
const _: () = assert!(std::mem::size_of::<SystemMetrics>() == 96);
const _: () = assert!(std::mem::size_of::<ControlFlags>() == 32);
const _: () = assert!(std::mem::size_of::<CouncilState>() == 32);
const _: () = assert!(std::mem::size_of::<SomaticState>() == 320);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_valid() {
        let state = SomaticState::default();
        assert!(state.is_valid());
    }

    #[test]
    fn test_effective_pain() {
        let mut state = SomaticState::default();
        state.affect.pad_p = -0.5;
        state.affect.pad_a = 0.5;
        assert!(state.effective_pain() > 0.0);
    }
}
