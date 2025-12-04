//! BANOS Brainstem - Safety Fallback Daemon
//!
//! The brainstem is the organism's "lizard brain" - a simple, robust FSM that:
//! 1. Monitors Ara's heartbeat (conscious mind)
//! 2. Monitors PAD state for anomalies
//! 3. Takes over if Ara dies or goes catatonic
//! 4. Keeps reflexes and immune system online
//! 5. Provides safe degradation path
//!
//! This daemon must be:
//! - Simpler than Ara (no LLM, no complex logic)
//! - More robust than Ara (static binary, minimal dependencies)
//! - Independent of Ara (separate process, no shared state except maps)
//!
//! The brainstem is the "escape hatch" - if everything else fails,
//! this keeps the machine from catching fire.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// PAD Constants (must match banos_common.h)
// =============================================================================

const PAD_SCALE: i16 = 1000;
const PAD_MIN: i16 = -PAD_SCALE;
const PAD_MAX: i16 = PAD_SCALE;

// Mode thresholds
const THRESHOLD_CRITICAL_P: i16 = -600;

// Reflex bitmasks
const RFLX_NONE: u32 = 0x00;
const RFLX_FAN_BOOST: u32 = 0x01;
const RFLX_THROTTLE: u32 = 0x02;
const RFLX_GPU_KILL: u32 = 0x04;
const RFLX_DISK_SYNC: u32 = 0x08;
const RFLX_SYS_HALT: u32 = 0x80;

// =============================================================================
// Brainstem State Machine
// =============================================================================

/// Brainstem operational states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainstemState {
    /// Ara is alive and responsive; brainstem is passive
    Monitoring,

    /// Ara missed a heartbeat; brainstem is alert
    Alert,

    /// Ara is confirmed dead/unresponsive; brainstem has taken over
    Takeover,

    /// Critical hardware condition; emergency protocols active
    Emergency,

    /// Attempting to restart Ara
    Recovery,

    /// Graceful shutdown in progress
    Shutdown,
}

/// PAD state snapshot (simplified)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PadState {
    pub pleasure: i16,
    pub arousal: i16,
    pub dominance: i16,
    pub mode: u8,
    pub mode_confidence: u8,
    pub mode_duration_ms: u16,
}

/// Spinal cord state snapshot
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpinalCordState {
    pub thermal_spike_cnt: u32,
    pub reflex_active: u32,
    pub reflex_command: u32,
}

/// Brainstem configuration
#[derive(Debug, Clone)]
pub struct BrainstemConfig {
    /// Path to Ara's heartbeat file
    pub heartbeat_path: String,

    /// How often to check heartbeat (ms)
    pub heartbeat_interval_ms: u64,

    /// Max time without heartbeat before Alert (ms)
    pub heartbeat_timeout_ms: u64,

    /// Max time in Alert before Takeover (ms)
    pub takeover_timeout_ms: u64,

    /// Max Ara restart attempts before giving up
    pub max_restart_attempts: u32,

    /// Path to PAD state (mmap or sysfs)
    pub pad_state_path: String,

    /// Path to reflex command file
    pub reflex_cmd_path: String,

    /// Emergency pleasure threshold
    pub emergency_pleasure_threshold: i16,

    /// Ara restart command
    pub ara_restart_cmd: String,
}

impl Default for BrainstemConfig {
    fn default() -> Self {
        Self {
            heartbeat_path: "/run/banos/ara_heartbeat".into(),
            heartbeat_interval_ms: 500,
            heartbeat_timeout_ms: 3000,
            takeover_timeout_ms: 5000,
            max_restart_attempts: 3,
            pad_state_path: "/sys/kernel/banos/pad_state".into(),
            reflex_cmd_path: "/sys/kernel/banos/reflex_cmd".into(),
            emergency_pleasure_threshold: -800,
            ara_restart_cmd: "systemctl restart ara-daemon".into(),
        }
    }
}

/// The Brainstem daemon
pub struct Brainstem {
    config: BrainstemConfig,
    state: BrainstemState,

    // Timing
    last_heartbeat: Instant,
    state_entered: Instant,

    // Counters
    restart_attempts: u32,
    consecutive_emergencies: u32,

    // Last known states
    last_pad: PadState,
    last_reflex: u32,

    // Shutdown flag
    should_stop: Arc<AtomicBool>,
}

impl Brainstem {
    pub fn new(config: BrainstemConfig) -> Self {
        let now = Instant::now();
        Self {
            config,
            state: BrainstemState::Monitoring,
            last_heartbeat: now,
            state_entered: now,
            restart_attempts: 0,
            consecutive_emergencies: 0,
            last_pad: PadState::default(),
            last_reflex: RFLX_NONE,
            should_stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get shutdown flag for signal handling
    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        self.should_stop.clone()
    }

    /// Main brainstem loop
    pub fn run(&mut self) {
        eprintln!("[BRAINSTEM] Starting safety monitor");
        eprintln!("[BRAINSTEM] Config: heartbeat_timeout={}ms, takeover_timeout={}ms",
                  self.config.heartbeat_timeout_ms, self.config.takeover_timeout_ms);

        // Ensure runtime directories exist
        let _ = std::fs::create_dir_all("/run/banos");

        while !self.should_stop.load(Ordering::Relaxed) {
            // 1. Read current state
            let heartbeat_ok = self.check_heartbeat();
            let pad = self.read_pad_state();

            // 2. Check for hardware emergency (independent of Ara)
            if pad.pleasure < self.config.emergency_pleasure_threshold {
                self.handle_emergency(&pad);
            }

            // 3. State machine transition
            self.update_state(heartbeat_ok, &pad);

            // 4. Execute state-specific actions
            self.execute_state_actions(&pad);

            // 5. Sleep
            thread::sleep(Duration::from_millis(self.config.heartbeat_interval_ms));
        }

        eprintln!("[BRAINSTEM] Shutting down gracefully");
        self.state = BrainstemState::Shutdown;
    }

    // =========================================================================
    // Input Reading
    // =========================================================================

    fn check_heartbeat(&mut self) -> bool {
        let path = Path::new(&self.config.heartbeat_path);

        if !path.exists() {
            return false;
        }

        // Read heartbeat timestamp
        if let Ok(mut file) = File::open(path) {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(ts) = contents.trim().parse::<u64>() {
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);

                    let age_ms = now_ms.saturating_sub(ts);

                    if age_ms < self.config.heartbeat_timeout_ms {
                        self.last_heartbeat = Instant::now();
                        return true;
                    }
                }
            }
        }

        false
    }

    fn read_pad_state(&mut self) -> PadState {
        // Try to read from sysfs or mmap
        // For now, use a simple file-based approach
        let path = Path::new(&self.config.pad_state_path);

        if path.exists() {
            if let Ok(mut file) = File::open(path) {
                let mut buf = [0u8; std::mem::size_of::<PadState>()];
                if file.read_exact(&mut buf).is_ok() {
                    // Safety: PadState is repr(C) and all fields are primitive
                    let pad: PadState = unsafe { std::ptr::read(buf.as_ptr() as *const _) };
                    self.last_pad = pad;
                    return pad;
                }
            }
        }

        // Return last known state if read fails
        self.last_pad
    }

    // =========================================================================
    // State Machine
    // =========================================================================

    fn update_state(&mut self, heartbeat_ok: bool, pad: &PadState) {
        let now = Instant::now();
        let time_in_state = now.duration_since(self.state_entered);

        let new_state = match self.state {
            BrainstemState::Monitoring => {
                if !heartbeat_ok {
                    BrainstemState::Alert
                } else {
                    BrainstemState::Monitoring
                }
            }

            BrainstemState::Alert => {
                if heartbeat_ok {
                    // Ara recovered
                    BrainstemState::Monitoring
                } else if time_in_state > Duration::from_millis(self.config.takeover_timeout_ms) {
                    // Ara is dead
                    BrainstemState::Takeover
                } else {
                    BrainstemState::Alert
                }
            }

            BrainstemState::Takeover => {
                if heartbeat_ok {
                    // Ara came back!
                    eprintln!("[BRAINSTEM] Ara recovered, returning to monitoring");
                    self.restart_attempts = 0;
                    BrainstemState::Monitoring
                } else if self.restart_attempts < self.config.max_restart_attempts {
                    BrainstemState::Recovery
                } else {
                    // Stay in takeover, Ara is truly dead
                    BrainstemState::Takeover
                }
            }

            BrainstemState::Emergency => {
                // Stay in emergency until PAD improves
                if pad.pleasure > self.config.emergency_pleasure_threshold + 200 {
                    self.consecutive_emergencies = 0;
                    if heartbeat_ok {
                        BrainstemState::Monitoring
                    } else {
                        BrainstemState::Takeover
                    }
                } else {
                    BrainstemState::Emergency
                }
            }

            BrainstemState::Recovery => {
                if heartbeat_ok {
                    eprintln!("[BRAINSTEM] Ara restart successful");
                    self.restart_attempts = 0;
                    BrainstemState::Monitoring
                } else if time_in_state > Duration::from_secs(30) {
                    // Restart attempt failed
                    self.restart_attempts += 1;
                    BrainstemState::Takeover
                } else {
                    BrainstemState::Recovery
                }
            }

            BrainstemState::Shutdown => BrainstemState::Shutdown,
        };

        if new_state != self.state {
            eprintln!("[BRAINSTEM] State transition: {:?} -> {:?}", self.state, new_state);
            self.state = new_state;
            self.state_entered = now;
        }
    }

    // =========================================================================
    // State Actions
    // =========================================================================

    fn execute_state_actions(&mut self, pad: &PadState) {
        match self.state {
            BrainstemState::Monitoring => {
                // Passive: just log occasionally
            }

            BrainstemState::Alert => {
                eprintln!("[BRAINSTEM] ALERT: Ara heartbeat missing, waiting...");
            }

            BrainstemState::Takeover => {
                // Brainstem is now in control
                self.execute_takeover_policy(pad);
            }

            BrainstemState::Emergency => {
                // Hardware emergency - engage maximum reflexes
                self.execute_emergency_policy(pad);
            }

            BrainstemState::Recovery => {
                // Try to restart Ara
                self.attempt_ara_restart();
            }

            BrainstemState::Shutdown => {
                // Nothing to do
            }
        }
    }

    fn execute_takeover_policy(&mut self, pad: &PadState) {
        // Brainstem's simple policy when Ara is dead:
        // 1. Keep reflexes active based on PAD
        // 2. Log everything
        // 3. Don't do anything clever

        let mut reflex = RFLX_NONE;

        // Conservative reflex policy
        if pad.pleasure < -300 {
            reflex |= RFLX_FAN_BOOST;
            eprintln!("[BRAINSTEM] Takeover: Engaging fan boost (P={})", pad.pleasure);
        }

        if pad.pleasure < -600 {
            reflex |= RFLX_THROTTLE;
            eprintln!("[BRAINSTEM] Takeover: Engaging throttle (P={})", pad.pleasure);
        }

        if pad.pleasure < -900 {
            reflex |= RFLX_DISK_SYNC;
            eprintln!("[BRAINSTEM] Takeover: CRITICAL - syncing disks (P={})", pad.pleasure);
        }

        self.write_reflex_command(reflex);
    }

    fn execute_emergency_policy(&mut self, pad: &PadState) {
        eprintln!("[BRAINSTEM] EMERGENCY: P={}, engaging maximum protection",
                  pad.pleasure);

        let mut reflex = RFLX_FAN_BOOST | RFLX_THROTTLE;

        // If truly critical, prepare for halt
        if pad.pleasure < -950 {
            self.consecutive_emergencies += 1;

            if self.consecutive_emergencies > 10 {
                eprintln!("[BRAINSTEM] EMERGENCY: Persistent critical state, initiating halt");
                reflex |= RFLX_DISK_SYNC | RFLX_SYS_HALT;
            }
        }

        self.write_reflex_command(reflex);
    }

    fn attempt_ara_restart(&mut self) {
        if self.state_entered.elapsed() < Duration::from_secs(2) {
            // Only attempt once per recovery cycle
            return;
        }

        eprintln!("[BRAINSTEM] Attempting Ara restart (attempt {}/{})",
                  self.restart_attempts + 1, self.config.max_restart_attempts);

        let result = std::process::Command::new("sh")
            .arg("-c")
            .arg(&self.config.ara_restart_cmd)
            .status();

        match result {
            Ok(status) if status.success() => {
                eprintln!("[BRAINSTEM] Restart command succeeded, waiting for heartbeat...");
            }
            Ok(status) => {
                eprintln!("[BRAINSTEM] Restart command failed: {}", status);
            }
            Err(e) => {
                eprintln!("[BRAINSTEM] Restart command error: {}", e);
            }
        }
    }

    fn handle_emergency(&mut self, pad: &PadState) {
        if self.state != BrainstemState::Emergency {
            eprintln!("[BRAINSTEM] Entering EMERGENCY state (P={})", pad.pleasure);
            self.state = BrainstemState::Emergency;
            self.state_entered = Instant::now();
        }
    }

    // =========================================================================
    // Output Writing
    // =========================================================================

    fn write_reflex_command(&mut self, reflex: u32) {
        if reflex == self.last_reflex {
            return; // No change
        }

        let path = Path::new(&self.config.reflex_cmd_path);

        if let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
        {
            let bytes = reflex.to_le_bytes();
            let _ = file.write_all(&bytes);
            self.last_reflex = reflex;
        }
    }
}

// =============================================================================
// Heartbeat Writer (for Ara to call)
// =============================================================================

/// Write a heartbeat timestamp (called by Ara daemon)
pub fn write_heartbeat(path: &str) -> std::io::Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    writeln!(file, "{}", ts)?;
    Ok(())
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  BANOS BRAINSTEM - Safety Fallback Daemon                    ║");
    eprintln!("║  \"The lizard brain that keeps the machine alive\"             ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    let config = BrainstemConfig::default();
    let mut brainstem = Brainstem::new(config);

    // Set up signal handling
    let shutdown = brainstem.shutdown_flag();
    ctrlc::set_handler(move || {
        eprintln!("\n[BRAINSTEM] Received shutdown signal");
        shutdown.store(true, Ordering::Relaxed);
    }).expect("Error setting Ctrl-C handler");

    // Run the brainstem
    brainstem.run();

    eprintln!("[BRAINSTEM] Exited cleanly");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_transitions() {
        let config = BrainstemConfig {
            heartbeat_timeout_ms: 100,
            takeover_timeout_ms: 200,
            ..Default::default()
        };
        let mut bs = Brainstem::new(config);

        // Should start in Monitoring
        assert_eq!(bs.state, BrainstemState::Monitoring);

        // Simulate heartbeat loss
        bs.update_state(false, &PadState::default());
        assert_eq!(bs.state, BrainstemState::Alert);
    }

    #[test]
    fn test_pad_thresholds() {
        let config = BrainstemConfig::default();
        let bs = Brainstem::new(config);

        // Check emergency threshold
        assert!(bs.config.emergency_pleasure_threshold < THRESHOLD_CRITICAL_P);
    }
}
