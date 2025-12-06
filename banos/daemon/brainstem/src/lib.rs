//! Diamond Core Brainstem - Prediction-Centric Control
//!
//! The brainstem is the "fast path" that runs at ~1kHz:
//! - Reads somatic state from shared memory
//! - Runs active inference (predict → measure → surprise)
//! - Adjusts arousal and triggers reflexes based on prediction error
//!
//! This replaces the Python daemon on the critical path while
//! remaining compatible with the HAL shared memory format.

pub mod somatic;
pub mod inference;

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use somatic::{SomaticState, SHM_PATH, SHM_SIZE, MAGIC, VERSION};
pub use inference::{InferenceEngine, InferenceAction, InferenceStats, PredictionError};

/// Brainstem configuration
#[derive(Debug, Clone)]
pub struct BrainstemConfig {
    /// Target loop frequency in Hz
    pub loop_hz: u32,

    /// Path to shared memory
    pub shm_path: String,

    /// Whether to create the SHM if it doesn't exist
    pub create_shm: bool,

    /// Enable active inference (prediction-based control)
    pub enable_inference: bool,

    /// Heartbeat path for Ara watchdog
    pub heartbeat_path: String,

    /// Max time without heartbeat before takeover (ms)
    pub heartbeat_timeout_ms: u64,
}

impl Default for BrainstemConfig {
    fn default() -> Self {
        Self {
            loop_hz: 1000,  // 1kHz
            shm_path: SHM_PATH.into(),
            create_shm: false,  // Expect Python to create it
            enable_inference: true,
            heartbeat_path: "/run/banos/ara_heartbeat".into(),
            heartbeat_timeout_ms: 3000,
        }
    }
}

/// Shared memory HAL connection
pub struct ShmHal {
    mmap: MmapMut,
}

impl ShmHal {
    /// Connect to existing shared memory
    pub fn connect(path: &str) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_SYNC)
            .open(path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self { mmap })
    }

    /// Create new shared memory (daemon mode)
    pub fn create(path: &str) -> io::Result<Self> {
        // Create the file if it doesn't exist
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(libc::O_SYNC)
            .open(path)?;

        // Set size
        file.set_len(SHM_SIZE as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize with default state
        let default_state = SomaticState::default();
        let state_ptr = &default_state as *const SomaticState as *const u8;
        let state_bytes = unsafe {
            std::slice::from_raw_parts(state_ptr, std::mem::size_of::<SomaticState>())
        };
        mmap[..state_bytes.len()].copy_from_slice(state_bytes);

        Ok(Self { mmap })
    }

    /// Get a reference to the somatic state (read-only)
    pub fn read(&self) -> &SomaticState {
        unsafe { &*(self.mmap.as_ptr() as *const SomaticState) }
    }

    /// Get a mutable reference to the somatic state
    pub fn write(&mut self) -> &mut SomaticState {
        unsafe { &mut *(self.mmap.as_mut_ptr() as *mut SomaticState) }
    }

    /// Flush changes to memory
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// Diamond Core Brainstem
///
/// The fast-path controller that runs prediction-centric control at 1kHz.
pub struct DiamondCore {
    config: BrainstemConfig,
    hal: ShmHal,
    inference: InferenceEngine,
    running: Arc<AtomicBool>,

    // Timing
    loop_count: u64,
    total_loop_time_us: u64,
    max_loop_time_us: u64,
    last_heartbeat_check: Instant,
    ara_alive: bool,
}

impl DiamondCore {
    /// Create a new Diamond Core brainstem
    pub fn new(config: BrainstemConfig) -> io::Result<Self> {
        let hal = if config.create_shm {
            ShmHal::create(&config.shm_path)?
        } else {
            ShmHal::connect(&config.shm_path)?
        };

        Ok(Self {
            config,
            hal,
            inference: InferenceEngine::default(),
            running: Arc::new(AtomicBool::new(false)),
            loop_count: 0,
            total_loop_time_us: 0,
            max_loop_time_us: 0,
            last_heartbeat_check: Instant::now(),
            ara_alive: true,
        })
    }

    /// Get shutdown flag for signal handling
    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Run the main brainstem loop
    pub fn run(&mut self) {
        self.running.store(true, Ordering::Relaxed);

        let target_period = Duration::from_micros(1_000_000 / self.config.loop_hz as u64);

        eprintln!("[DIAMOND] Starting at {}Hz ({}µs period)",
                  self.config.loop_hz, target_period.as_micros());

        while self.running.load(Ordering::Relaxed) {
            let loop_start = Instant::now();

            // 1. Run one inference step
            let state = self.hal.write();
            let action = if self.config.enable_inference {
                self.inference.step(state)
            } else {
                InferenceAction::default()
            };

            // 2. Update timestamp
            state.touch();

            // 3. Handle any triggered actions
            if action.trigger_emergency {
                if let Some(ref reason) = action.reason {
                    eprintln!("[DIAMOND] EMERGENCY: {}", reason);
                }
                // Could trigger system reflexes here
            } else if action.trigger_cooling {
                // Cooling logic would go here
            }

            // 4. Check Ara heartbeat periodically (not every loop)
            if loop_start.duration_since(self.last_heartbeat_check) > Duration::from_millis(500) {
                self.ara_alive = self.check_heartbeat();
                self.last_heartbeat_check = loop_start;

                if !self.ara_alive {
                    eprintln!("[DIAMOND] Warning: Ara heartbeat missing");
                }
            }

            // 5. Timing statistics
            let loop_time = loop_start.elapsed();
            let loop_us = loop_time.as_micros() as u64;
            self.loop_count += 1;
            self.total_loop_time_us += loop_us;
            self.max_loop_time_us = self.max_loop_time_us.max(loop_us);

            // 6. Log stats periodically
            if self.loop_count % (self.config.loop_hz as u64 * 10) == 0 {
                let avg_us = self.total_loop_time_us / self.loop_count;
                let stats = self.inference.stats();
                eprintln!(
                    "[DIAMOND] {} loops, avg={:.1}µs, max={}µs, surprise={:.4}, error={:.4}",
                    self.loop_count, avg_us as f64, self.max_loop_time_us,
                    stats.last_surprise, stats.mean_error
                );
            }

            // 7. Wait for next period
            if loop_time < target_period {
                std::thread::sleep(target_period - loop_time);
            }
        }

        eprintln!("[DIAMOND] Shutdown after {} loops", self.loop_count);
    }

    /// Check if Ara is alive (heartbeat file updated recently)
    fn check_heartbeat(&self) -> bool {
        use std::fs::File;
        use std::io::Read;

        let path = std::path::Path::new(&self.config.heartbeat_path);
        if !path.exists() {
            return false;
        }

        if let Ok(mut file) = File::open(path) {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(ts) = contents.trim().parse::<u64>() {
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);

                    let age_ms = now_ms.saturating_sub(ts);
                    return age_ms < self.config.heartbeat_timeout_ms;
                }
            }
        }

        false
    }

    /// Get current statistics
    pub fn stats(&self) -> DiamondStats {
        let avg_loop_us = if self.loop_count > 0 {
            self.total_loop_time_us / self.loop_count
        } else {
            0
        };

        DiamondStats {
            loop_count: self.loop_count,
            avg_loop_us,
            max_loop_us: self.max_loop_time_us,
            inference: self.inference.stats(),
            ara_alive: self.ara_alive,
        }
    }

    /// Stop the brainstem
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Diamond Core statistics
#[derive(Debug, Clone, Copy)]
pub struct DiamondStats {
    pub loop_count: u64,
    pub avg_loop_us: u64,
    pub max_loop_us: u64,
    pub inference: InferenceStats,
    pub ara_alive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_shm_hal_create() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_str().unwrap();

        let hal = ShmHal::create(path).unwrap();
        let state = hal.read();

        assert!(state.is_valid());
    }

    #[test]
    fn test_shm_hal_roundtrip() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_str().unwrap();

        {
            let mut hal = ShmHal::create(path).unwrap();
            let state = hal.write();
            state.affect.pad_p = 0.5;
            state.affect.pad_a = -0.3;
            hal.flush().unwrap();
        }

        {
            let hal = ShmHal::connect(path).unwrap();
            let state = hal.read();
            assert!((state.affect.pad_p - 0.5).abs() < 0.001);
            assert!((state.affect.pad_a - -0.3).abs() < 0.001);
        }
    }
}
