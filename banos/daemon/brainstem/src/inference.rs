//! Active Inference - Prediction-Centric Control
//!
//! The Diamond Core treats prediction error as the primary internal scalar.
//! Large error → raise arousal, trigger corrective actions.
//! Small error → system is "in sync", can relax.
//!
//! This implements a simple but effective active inference loop:
//! 1. Predict next state from current observations
//! 2. Measure actual state
//! 3. Compute surprise (prediction error)
//! 4. Update arousal and trigger actions based on surprise

use crate::somatic::SomaticState;

/// Prediction model - starts simple (EMA), can evolve to SNN later
pub struct Predictor {
    // Exponential moving average coefficients
    alpha_pain: f32,
    alpha_entropy: f32,

    // Last predictions
    predicted_pain: f32,
    predicted_entropy: f32,

    // Prediction error history for smoothing
    error_history: [f32; 8],
    error_idx: usize,

    // Calibration
    prediction_count: u64,
    total_error: f64,
}

impl Default for Predictor {
    fn default() -> Self {
        Self::new(0.3, 0.2) // Reasonable defaults
    }
}

impl Predictor {
    /// Create a new predictor with given smoothing coefficients
    pub fn new(alpha_pain: f32, alpha_entropy: f32) -> Self {
        Self {
            alpha_pain,
            alpha_entropy,
            predicted_pain: 0.0,
            predicted_entropy: 0.0,
            error_history: [0.0; 8],
            error_idx: 0,
            prediction_count: 0,
            total_error: 0.0,
        }
    }

    /// Update predictions and compute error
    pub fn step(&mut self, state: &SomaticState) -> PredictionError {
        // Get actual values
        let actual_pain = state.effective_pain();
        let actual_entropy = state.effective_entropy();

        // Compute errors
        let pain_error = (self.predicted_pain - actual_pain).abs();
        let entropy_error = (self.predicted_entropy - actual_entropy).abs();

        // Combined surprise (weighted average)
        let surprise = 0.6 * pain_error + 0.4 * entropy_error;

        // Update error history for smoothing
        self.error_history[self.error_idx] = surprise;
        self.error_idx = (self.error_idx + 1) % self.error_history.len();

        // Smoothed surprise (average of history)
        let smoothed_surprise: f32 = self.error_history.iter().sum::<f32>()
            / self.error_history.len() as f32;

        // Update statistics
        self.prediction_count += 1;
        self.total_error += surprise as f64;

        // Update predictions for next step (simple EMA)
        self.predicted_pain = self.alpha_pain * actual_pain
            + (1.0 - self.alpha_pain) * self.predicted_pain;
        self.predicted_entropy = self.alpha_entropy * actual_entropy
            + (1.0 - self.alpha_entropy) * self.predicted_entropy;

        PredictionError {
            pain_error,
            entropy_error,
            surprise,
            smoothed_surprise,
            predicted_pain: self.predicted_pain,
            predicted_entropy: self.predicted_entropy,
        }
    }

    /// Get average prediction error over lifetime
    pub fn mean_error(&self) -> f32 {
        if self.prediction_count == 0 {
            0.0
        } else {
            (self.total_error / self.prediction_count as f64) as f32
        }
    }

    /// Reset the predictor
    pub fn reset(&mut self) {
        self.predicted_pain = 0.0;
        self.predicted_entropy = 0.0;
        self.error_history = [0.0; 8];
        self.error_idx = 0;
        self.prediction_count = 0;
        self.total_error = 0.0;
    }
}

/// Result of a prediction step
#[derive(Debug, Clone, Copy)]
pub struct PredictionError {
    pub pain_error: f32,
    pub entropy_error: f32,
    pub surprise: f32,
    pub smoothed_surprise: f32,
    pub predicted_pain: f32,
    pub predicted_entropy: f32,
}

/// Active Inference Policy
///
/// Maps prediction error into actions (arousal changes, reflex triggers, etc.)
pub struct InferencePolicy {
    /// Surprise threshold for raising arousal
    pub surprise_threshold_low: f32,
    pub surprise_threshold_high: f32,

    /// Arousal adjustment rates
    pub arousal_increase_rate: f32,
    pub arousal_decrease_rate: f32,

    /// Thermal response thresholds
    pub thermal_warning_threshold: f32,
    pub thermal_critical_threshold: f32,
}

impl Default for InferencePolicy {
    fn default() -> Self {
        Self {
            surprise_threshold_low: 0.1,
            surprise_threshold_high: 0.5,
            arousal_increase_rate: 0.1,
            arousal_decrease_rate: 0.05,
            thermal_warning_threshold: 80.0,
            thermal_critical_threshold: 90.0,
        }
    }
}

/// Actions to take based on inference
#[derive(Debug, Clone, Default)]
pub struct InferenceAction {
    /// Change in arousal (positive = increase)
    pub arousal_delta: f32,

    /// Cooling action needed
    pub trigger_cooling: bool,

    /// Emergency action needed
    pub trigger_emergency: bool,

    /// Description for logging
    pub reason: Option<String>,
}

impl InferencePolicy {
    /// Apply policy to current state and prediction error
    pub fn evaluate(
        &self,
        state: &SomaticState,
        error: &PredictionError,
    ) -> InferenceAction {
        let mut action = InferenceAction::default();

        // 1. Arousal adjustment based on surprise
        if error.smoothed_surprise > self.surprise_threshold_high {
            // High surprise → increase arousal
            action.arousal_delta = self.arousal_increase_rate;
            action.reason = Some(format!(
                "High surprise ({:.3}), increasing arousal",
                error.smoothed_surprise
            ));
        } else if error.smoothed_surprise < self.surprise_threshold_low {
            // Low surprise → system is in sync, can relax
            action.arousal_delta = -self.arousal_decrease_rate;
        }

        // 2. Thermal response
        let max_temp = state.metrics.cpu_temp.max(state.metrics.gpu_temp);

        if max_temp > self.thermal_critical_threshold {
            action.trigger_emergency = true;
            action.arousal_delta = action.arousal_delta.max(0.2);
            action.reason = Some(format!(
                "CRITICAL: Temp {:.1}°C exceeds {:.1}°C",
                max_temp, self.thermal_critical_threshold
            ));
        } else if max_temp > self.thermal_warning_threshold {
            action.trigger_cooling = true;
            action.reason = Some(format!(
                "Warning: Temp {:.1}°C exceeds {:.1}°C",
                max_temp, self.thermal_warning_threshold
            ));
        }

        // 3. Pain response
        if state.sensors.pain_weber > 0.7 {
            action.arousal_delta = action.arousal_delta.max(0.15);
            action.trigger_cooling = true;
        }

        // 4. Emergency stop flag
        if state.control.emergency_stop != 0 {
            action.trigger_emergency = true;
            action.reason = Some("Emergency stop flag set".into());
        }

        action
    }

    /// Apply action to state (modifies arousal)
    pub fn apply_to_state(&self, state: &mut SomaticState, action: &InferenceAction) {
        let new_arousal = (state.affect.pad_a + action.arousal_delta).clamp(-1.0, 1.0);
        state.affect.pad_a = new_arousal;
    }
}

/// Complete Active Inference Engine
pub struct InferenceEngine {
    pub predictor: Predictor,
    pub policy: InferencePolicy,
    pub last_error: Option<PredictionError>,
    pub last_action: Option<InferenceAction>,
    pub step_count: u64,
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self {
            predictor: Predictor::default(),
            policy: InferencePolicy::default(),
            last_error: None,
            last_action: None,
            step_count: 0,
        }
    }
}

impl InferenceEngine {
    /// Run one inference step
    pub fn step(&mut self, state: &mut SomaticState) -> InferenceAction {
        // 1. Predict and measure error
        let error = self.predictor.step(state);
        self.last_error = Some(error);

        // 2. Evaluate policy
        let action = self.policy.evaluate(state, &error);

        // 3. Apply action to state
        self.policy.apply_to_state(state, &action);

        self.last_action = Some(action.clone());
        self.step_count += 1;

        action
    }

    /// Get statistics
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            step_count: self.step_count,
            mean_error: self.predictor.mean_error(),
            last_surprise: self.last_error.map(|e| e.smoothed_surprise).unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InferenceStats {
    pub step_count: u64,
    pub mean_error: f32,
    pub last_surprise: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_step() {
        let mut pred = Predictor::default();
        let mut state = SomaticState::default();

        // First step should have some error
        let err = pred.step(&state);
        assert!(err.surprise >= 0.0);

        // After many steps with stable state, error should decrease
        for _ in 0..100 {
            pred.step(&state);
        }

        let err2 = pred.step(&state);
        assert!(err2.surprise < err.surprise || err.surprise < 0.01);
    }

    #[test]
    fn test_policy_thermal() {
        let policy = InferencePolicy::default();
        let mut state = SomaticState::default();
        state.metrics.cpu_temp = 95.0;

        let error = PredictionError {
            pain_error: 0.0,
            entropy_error: 0.0,
            surprise: 0.0,
            smoothed_surprise: 0.0,
            predicted_pain: 0.0,
            predicted_entropy: 0.0,
        };

        let action = policy.evaluate(&state, &error);
        assert!(action.trigger_emergency);
    }
}
