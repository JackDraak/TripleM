use crate::audio::mixer::MoodWeights;
use crate::audio::utils::{cosine_interpolate, clamp};

/// Manages smooth transitions between different mood states
#[derive(Debug)]
pub struct TransitionManager {
    current_weights: MoodWeights,
    target_weights: MoodWeights,
    transition_progress: f32,     // 0.0 to 1.0
    transition_duration: f32,     // In seconds
    transition_speed: f32,        // Progress increment per sample
    is_transitioning: bool,
    sample_rate: f32,
}

impl TransitionManager {
    /// Create a new transition manager
    pub fn new(sample_rate: f32, transition_duration: f32) -> Self {
        Self {
            current_weights: MoodWeights::default(),
            target_weights: MoodWeights::default(),
            transition_progress: 0.0,
            transition_duration,
            transition_speed: 0.0,
            is_transitioning: false,
            sample_rate,
        }
    }

    /// Start a transition to a new mood value
    pub fn start_transition(&mut self, target_mood: f32) {
        let target_mood = clamp(target_mood, 0.0, 1.0);
        let new_target_weights = MoodWeights::from_mood_value(target_mood);

        // Only start a new transition if the target is significantly different
        if self.weights_differ_significantly(&self.target_weights, &new_target_weights) {
            self.target_weights = new_target_weights;
            self.transition_progress = 0.0;
            self.transition_speed = 1.0 / (self.transition_duration * self.sample_rate);
            self.is_transitioning = true;
        }
    }

    /// Update the transition state (call once per audio sample)
    pub fn update(&mut self) {
        if !self.is_transitioning {
            return;
        }

        self.transition_progress += self.transition_speed;

        if self.transition_progress >= 1.0 {
            // Transition complete
            self.transition_progress = 1.0;
            self.current_weights = self.target_weights;
            self.is_transitioning = false;
        } else {
            // Interpolate between current and target weights
            self.current_weights = self.interpolate_weights(self.transition_progress);
        }
    }

    /// Get the current interpolated weights
    pub fn current_weights(&self) -> &MoodWeights {
        &self.current_weights
    }

    /// Check if currently transitioning
    pub fn is_transitioning(&self) -> bool {
        self.is_transitioning
    }

    /// Get transition progress (0.0 to 1.0)
    pub fn transition_progress(&self) -> f32 {
        self.transition_progress
    }

    /// Set the transition duration
    pub fn set_transition_duration(&mut self, duration: f32) {
        self.transition_duration = duration.max(0.1); // Minimum 100ms
        if self.is_transitioning {
            // Recalculate speed for current transition
            self.transition_speed = 1.0 / (self.transition_duration * self.sample_rate);
        }
    }

    /// Get the current transition duration
    pub fn transition_duration(&self) -> f32 {
        self.transition_duration
    }

    /// Force set the current weights without transition
    pub fn set_weights_immediate(&mut self, mood: f32) {
        let mood = clamp(mood, 0.0, 1.0);
        self.current_weights = MoodWeights::from_mood_value(mood);
        self.target_weights = self.current_weights;
        self.is_transitioning = false;
        self.transition_progress = 0.0;
    }

    /// Get the target weights
    pub fn target_weights(&self) -> &MoodWeights {
        &self.target_weights
    }

    /// Get estimated time remaining in transition (in seconds)
    pub fn time_remaining(&self) -> f32 {
        if !self.is_transitioning {
            return 0.0;
        }

        let remaining_progress = 1.0 - self.transition_progress;
        remaining_progress / self.transition_speed / self.sample_rate
    }

    /// Check if two weight sets differ significantly enough to warrant a transition
    fn weights_differ_significantly(&self, a: &MoodWeights, b: &MoodWeights) -> bool {
        const THRESHOLD: f32 = 0.05; // 5% difference threshold

        (a.environmental - b.environmental).abs() > THRESHOLD
            || (a.gentle_melodic - b.gentle_melodic).abs() > THRESHOLD
            || (a.active_ambient - b.active_ambient).abs() > THRESHOLD
            || (a.edm_style - b.edm_style).abs() > THRESHOLD
    }

    /// Interpolate between current and target weights using cosine interpolation
    fn interpolate_weights(&self, t: f32) -> MoodWeights {
        // Use cosine interpolation for smoother transitions
        let smooth_t = (1.0 - (t * std::f32::consts::PI).cos()) * 0.5;

        MoodWeights {
            environmental: cosine_interpolate(
                self.current_weights.environmental,
                self.target_weights.environmental,
                smooth_t,
            ),
            gentle_melodic: cosine_interpolate(
                self.current_weights.gentle_melodic,
                self.target_weights.gentle_melodic,
                smooth_t,
            ),
            active_ambient: cosine_interpolate(
                self.current_weights.active_ambient,
                self.target_weights.active_ambient,
                smooth_t,
            ),
            edm_style: cosine_interpolate(
                self.current_weights.edm_style,
                self.target_weights.edm_style,
                smooth_t,
            ),
        }
    }
}

/// Advanced transition manager with multiple curve types
#[derive(Debug)]
pub struct AdvancedTransitionManager {
    base_manager: TransitionManager,
    curve_type: TransitionCurve,
    ease_in_factor: f32,
    ease_out_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionCurve {
    Linear,
    Cosine,
    EaseIn,
    EaseOut,
    EaseInOut,
    Exponential,
}

impl AdvancedTransitionManager {
    pub fn new(sample_rate: f32, transition_duration: f32, curve_type: TransitionCurve) -> Self {
        Self {
            base_manager: TransitionManager::new(sample_rate, transition_duration),
            curve_type,
            ease_in_factor: 2.0,
            ease_out_factor: 2.0,
        }
    }

    pub fn start_transition(&mut self, target_mood: f32) {
        self.base_manager.start_transition(target_mood);
    }

    pub fn update(&mut self) {
        if !self.base_manager.is_transitioning {
            return;
        }

        self.base_manager.transition_progress += self.base_manager.transition_speed;

        if self.base_manager.transition_progress >= 1.0 {
            self.base_manager.transition_progress = 1.0;
            self.base_manager.current_weights = self.base_manager.target_weights;
            self.base_manager.is_transitioning = false;
        } else {
            let curved_progress = self.apply_transition_curve(self.base_manager.transition_progress);
            self.base_manager.current_weights = self.interpolate_with_curve(curved_progress);
        }
    }

    pub fn current_weights(&self) -> &MoodWeights {
        &self.base_manager.current_weights
    }

    pub fn is_transitioning(&self) -> bool {
        self.base_manager.is_transitioning
    }

    pub fn set_curve_type(&mut self, curve_type: TransitionCurve) {
        self.curve_type = curve_type;
    }

    pub fn set_ease_factors(&mut self, ease_in: f32, ease_out: f32) {
        self.ease_in_factor = ease_in.max(0.1);
        self.ease_out_factor = ease_out.max(0.1);
    }

    fn apply_transition_curve(&self, t: f32) -> f32 {
        let t = clamp(t, 0.0, 1.0);

        match self.curve_type {
            TransitionCurve::Linear => t,
            TransitionCurve::Cosine => (1.0 - (t * std::f32::consts::PI).cos()) * 0.5,
            TransitionCurve::EaseIn => t.powf(self.ease_in_factor),
            TransitionCurve::EaseOut => 1.0 - (1.0 - t).powf(self.ease_out_factor),
            TransitionCurve::EaseInOut => {
                if t < 0.5 {
                    0.5 * (2.0 * t).powf(self.ease_in_factor)
                } else {
                    1.0 - 0.5 * (2.0 * (1.0 - t)).powf(self.ease_out_factor)
                }
            }
            TransitionCurve::Exponential => {
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else {
                    2.0_f32.powf(10.0 * (t - 1.0))
                }
            }
        }
    }

    fn interpolate_with_curve(&self, t: f32) -> MoodWeights {
        MoodWeights {
            environmental: self.base_manager.current_weights.environmental
                + (self.base_manager.target_weights.environmental
                    - self.base_manager.current_weights.environmental)
                    * t,
            gentle_melodic: self.base_manager.current_weights.gentle_melodic
                + (self.base_manager.target_weights.gentle_melodic
                    - self.base_manager.current_weights.gentle_melodic)
                    * t,
            active_ambient: self.base_manager.current_weights.active_ambient
                + (self.base_manager.target_weights.active_ambient
                    - self.base_manager.current_weights.active_ambient)
                    * t,
            edm_style: self.base_manager.current_weights.edm_style
                + (self.base_manager.target_weights.edm_style - self.base_manager.current_weights.edm_style)
                    * t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_manager_creation() {
        let manager = TransitionManager::new(44100.0, 2.0);
        assert!(!manager.is_transitioning());
        assert_eq!(manager.transition_duration(), 2.0);
    }

    #[test]
    fn test_start_transition() {
        let mut manager = TransitionManager::new(44100.0, 1.0);

        manager.start_transition(1.0);
        assert!(manager.is_transitioning());
        assert_eq!(manager.transition_progress(), 0.0);
    }

    #[test]
    fn test_transition_update() {
        let mut manager = TransitionManager::new(44100.0, 1.0);
        manager.start_transition(1.0);

        // Simulate some updates
        for _ in 0..100 {
            manager.update();
        }

        assert!(manager.transition_progress() > 0.0);
        assert!(manager.transition_progress() < 1.0);
    }

    #[test]
    fn test_transition_completion() {
        let mut manager = TransitionManager::new(44100.0, 0.1); // Short transition
        manager.start_transition(1.0);

        // Run enough updates to complete transition
        for _ in 0..(44100.0 * 0.1) as usize + 10 {
            manager.update();
        }

        assert!(!manager.is_transitioning());
        assert_eq!(manager.transition_progress(), 1.0);
    }

    #[test]
    fn test_immediate_weight_setting() {
        let mut manager = TransitionManager::new(44100.0, 2.0);
        manager.set_weights_immediate(0.8);

        assert!(!manager.is_transitioning());
        let weights = manager.current_weights();
        assert!(weights.edm_style > 0.5); // Should be in EDM range
    }

    #[test]
    fn test_transition_duration_change() {
        let mut manager = TransitionManager::new(44100.0, 2.0);
        manager.set_transition_duration(1.0);
        assert_eq!(manager.transition_duration(), 1.0);
    }

    #[test]
    fn test_time_remaining() {
        let mut manager = TransitionManager::new(44100.0, 1.0);
        manager.start_transition(1.0);

        let initial_time = manager.time_remaining();
        assert!(initial_time > 0.9 && initial_time <= 1.0);

        // Update partway through
        for _ in 0..22050 { // Half a second at 44.1kHz
            manager.update();
        }

        let remaining = manager.time_remaining();
        assert!(remaining < initial_time);
        assert!(remaining > 0.4 && remaining < 0.6);
    }

    #[test]
    fn test_advanced_transition_curves() {
        let mut manager = AdvancedTransitionManager::new(44100.0, 1.0, TransitionCurve::EaseInOut);
        manager.start_transition(1.0);

        assert!(manager.is_transitioning());

        // Test curve type change
        manager.set_curve_type(TransitionCurve::Exponential);
        manager.update();

        // Should still be transitioning
        assert!(manager.is_transitioning());
    }

    #[test]
    fn test_weights_differ_significantly() {
        let manager = TransitionManager::new(44100.0, 1.0);

        let weights1 = MoodWeights::from_mood_value(0.0);
        let weights2 = MoodWeights::from_mood_value(1.0);
        let weights3 = MoodWeights::from_mood_value(0.01); // Very close to weights1

        assert!(manager.weights_differ_significantly(&weights1, &weights2));
        assert!(!manager.weights_differ_significantly(&weights1, &weights3));
    }
}