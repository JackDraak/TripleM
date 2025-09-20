//! Natural variation system for organic parameter evolution
//!
//! This module implements pink noise-driven parameter drift to prevent
//! the "loop detection" issue mentioned in the research notes. It provides
//! organic, slowly-evolving changes to musical parameters.

use crate::audio::utils;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Natural variation system for preventing static, looped-sounding audio
#[derive(Debug, Clone)]
pub struct NaturalVariation {
    // Pink noise states for different parameter types
    timing_state: [f32; 7],
    pitch_state: [f32; 7],
    amplitude_state: [f32; 7],
    timbre_state: [f32; 7],

    // Accumulators for smooth parameter drift
    timing_accumulator: f32,
    pitch_accumulator: f32,
    amplitude_accumulator: f32,
    timbre_accumulator: f32,

    // Rate controls for different types of variation
    timing_rate: f32,
    pitch_rate: f32,
    amplitude_rate: f32,
    timbre_rate: f32,

    // Random number generator for white noise input
    rng: StdRng,

    // Configuration
    intensity: f32,  // 0.0 to 1.0 - how much variation to apply
    time_scale: f32, // How slowly parameters evolve
}

impl NaturalVariation {
    /// Create a new natural variation system
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            timing_state: [0.0; 7],
            pitch_state: [0.0; 7],
            amplitude_state: [0.0; 7],
            timbre_state: [0.0; 7],

            timing_accumulator: 0.0,
            pitch_accumulator: 0.0,
            amplitude_accumulator: 0.0,
            timbre_accumulator: 0.0,

            // Different rates for different parameter types
            timing_rate: 0.0001,   // Very slow timing variations
            pitch_rate: 0.0002,    // Slow pitch drift
            amplitude_rate: 0.0005, // Moderate amplitude changes
            timbre_rate: 0.0003,   // Slow timbre evolution

            rng,
            intensity: 0.5,        // Moderate intensity by default
            time_scale: 1.0,       // Normal time scale
        }
    }

    /// Set the overall intensity of variations (0.0 to 1.0)
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// Set the time scale for parameter evolution (higher = faster changes)
    pub fn set_time_scale(&mut self, scale: f32) {
        self.time_scale = scale.max(0.1);
    }

    /// Update the variation system (call once per audio sample or batch)
    pub fn update(&mut self) {
        let white_noise = utils::white_noise(&mut self.rng);

        // Generate pink noise for each parameter type
        let timing_pink = utils::pink_noise_simple(white_noise, &mut self.timing_state);
        let pitch_pink = utils::pink_noise_simple(white_noise, &mut self.pitch_state);
        let amplitude_pink = utils::pink_noise_simple(white_noise, &mut self.amplitude_state);
        let timbre_pink = utils::pink_noise_simple(white_noise, &mut self.timbre_state);

        // Accumulate variations with different rates and time scaling
        let time_factor = self.time_scale * self.intensity;

        self.timing_accumulator += timing_pink * self.timing_rate * time_factor;
        self.pitch_accumulator += pitch_pink * self.pitch_rate * time_factor;
        self.amplitude_accumulator += amplitude_pink * self.amplitude_rate * time_factor;
        self.timbre_accumulator += timbre_pink * self.timbre_rate * time_factor;

        // Keep accumulators in reasonable bounds
        self.timing_accumulator = self.timing_accumulator.clamp(-1.0, 1.0);
        self.pitch_accumulator = self.pitch_accumulator.clamp(-1.0, 1.0);
        self.amplitude_accumulator = self.amplitude_accumulator.clamp(-1.0, 1.0);
        self.timbre_accumulator = self.timbre_accumulator.clamp(-1.0, 1.0);
    }

    /// Get timing variation (-1.0 to 1.0)
    /// Use this to slightly randomize note durations, beat timing, etc.
    pub fn get_timing_variation(&self) -> f32 {
        self.timing_accumulator
    }

    /// Get pitch variation (-1.0 to 1.0)
    /// Use this for subtle pitch drift, detuning, vibrato effects
    pub fn get_pitch_variation(&self) -> f32 {
        self.pitch_accumulator
    }

    /// Get amplitude variation (-1.0 to 1.0)
    /// Use this for volume fluctuations, tremolo effects
    pub fn get_amplitude_variation(&self) -> f32 {
        self.amplitude_accumulator
    }

    /// Get timbre variation (-1.0 to 1.0)
    /// Use this for filter cutoff drift, synthesis parameter changes
    pub fn get_timbre_variation(&self) -> f32 {
        self.timbre_accumulator
    }

    /// Get a scaled timing variation for duration multipliers (0.8 to 1.2 typical)
    pub fn get_duration_multiplier(&self, scale: f32) -> f32 {
        1.0 + self.timing_accumulator * scale
    }

    /// Get a scaled pitch variation in cents (-scale to +scale)
    pub fn get_pitch_drift_cents(&self, max_cents: f32) -> f32 {
        self.pitch_accumulator * max_cents
    }

    /// Get a scaled pitch variation as frequency multiplier
    pub fn get_pitch_drift_multiplier(&self, max_cents: f32) -> f32 {
        let cents = self.get_pitch_drift_cents(max_cents);
        2.0_f32.powf(cents / 1200.0)
    }

    /// Get a scaled amplitude variation (0.0 to 2.0 typical)
    pub fn get_amplitude_multiplier(&self, scale: f32) -> f32 {
        (1.0 + self.amplitude_accumulator * scale).max(0.0)
    }

    /// Get a scaled timbre variation for filter cutoff, etc.
    pub fn get_timbre_drift(&self, center: f32, scale: f32) -> f32 {
        center * (1.0 + self.timbre_accumulator * scale)
    }

    /// Reset all variation accumulators to zero
    pub fn reset(&mut self) {
        self.timing_accumulator = 0.0;
        self.pitch_accumulator = 0.0;
        self.amplitude_accumulator = 0.0;
        self.timbre_accumulator = 0.0;

        // Reset pink noise states
        self.timing_state = [0.0; 7];
        self.pitch_state = [0.0; 7];
        self.amplitude_state = [0.0; 7];
        self.timbre_state = [0.0; 7];
    }

    /// Create a variation system optimized for gentle, slow changes (spa music)
    pub fn gentle() -> Self {
        let mut variation = Self::new(None);
        variation.set_intensity(0.3);
        variation.set_time_scale(0.5);
        variation.timing_rate = 0.00005;
        variation.pitch_rate = 0.0001;
        variation.amplitude_rate = 0.0002;
        variation.timbre_rate = 0.00015;
        variation
    }

    /// Create a variation system optimized for active, productivity-focused music
    pub fn active() -> Self {
        let mut variation = Self::new(None);
        variation.set_intensity(0.4);
        variation.set_time_scale(0.8);
        variation.timing_rate = 0.0001;
        variation.pitch_rate = 0.0002;
        variation.amplitude_rate = 0.0003;
        variation.timbre_rate = 0.0004;
        variation
    }

    /// Create a variation system optimized for high-energy EDM
    pub fn energetic() -> Self {
        let mut variation = Self::new(None);
        variation.set_intensity(0.6);
        variation.set_time_scale(1.2);
        variation.timing_rate = 0.0002;
        variation.pitch_rate = 0.0003;
        variation.amplitude_rate = 0.0006;
        variation.timbre_rate = 0.0005;
        variation
    }
}

/// Micro-timing system for adding subtle human-like timing imperfections
#[derive(Debug, Clone)]
pub struct MicroTiming {
    variation: NaturalVariation,
    timing_jitter: f32,  // Maximum timing offset in seconds
}

impl MicroTiming {
    /// Create a new micro-timing system
    pub fn new(max_jitter_ms: f32) -> Self {
        Self {
            variation: NaturalVariation::new(None),
            timing_jitter: max_jitter_ms / 1000.0, // Convert to seconds
        }
    }

    /// Update the micro-timing system
    pub fn update(&mut self) {
        self.variation.update();
    }

    /// Get a timing offset for the current sample
    pub fn get_timing_offset(&self) -> f32 {
        self.variation.get_timing_variation() * self.timing_jitter
    }

    /// Apply micro-timing to a beat duration
    pub fn apply_to_duration(&self, duration: f32) -> f32 {
        duration + self.get_timing_offset()
    }
}

/// Dynamic range controller that prevents monotonous audio levels
#[derive(Debug, Clone)]
pub struct DynamicRange {
    variation: NaturalVariation,
    base_gain: f32,
    range_db: f32,  // Total dynamic range in dB
}

impl DynamicRange {
    /// Create a new dynamic range controller
    pub fn new(base_gain: f32, range_db: f32) -> Self {
        Self {
            variation: NaturalVariation::new(None),
            base_gain,
            range_db,
        }
    }

    /// Update the dynamic range controller
    pub fn update(&mut self) {
        self.variation.update();
    }

    /// Get the current gain multiplier
    pub fn get_gain(&self) -> f32 {
        let variation_db = self.variation.get_amplitude_variation() * self.range_db * 0.5;
        self.base_gain * utils::db_to_linear(variation_db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_variation_creation() {
        let variation = NaturalVariation::new(Some(42));
        assert_eq!(variation.intensity, 0.5);
        assert_eq!(variation.time_scale, 1.0);
    }

    #[test]
    fn test_variation_updates() {
        let mut variation = NaturalVariation::new(Some(42));

        // Initially should be zero
        assert_eq!(variation.get_timing_variation(), 0.0);

        // After updates, should have some variation
        for _ in 0..1000 {
            variation.update();
        }

        // Should have accumulated some variation
        assert_ne!(variation.get_timing_variation(), 0.0);
        assert!(variation.get_timing_variation().abs() <= 1.0);
    }

    #[test]
    fn test_pitch_drift_multiplier() {
        let mut variation = NaturalVariation::new(Some(42));
        variation.pitch_accumulator = 0.5; // Set directly for testing

        let multiplier = variation.get_pitch_drift_multiplier(100.0); // 100 cents = 1 semitone
        assert!(multiplier > 1.0); // Should be sharp
        assert!(multiplier < 1.1); // But not too much
    }

    #[test]
    fn test_preset_variations() {
        let gentle = NaturalVariation::gentle();
        assert_eq!(gentle.intensity, 0.3);

        let active = NaturalVariation::active();
        assert_eq!(active.intensity, 0.4);

        let energetic = NaturalVariation::energetic();
        assert_eq!(energetic.intensity, 0.6);
    }

    #[test]
    fn test_micro_timing() {
        let mut micro = MicroTiming::new(10.0); // 10ms max jitter

        for _ in 0..100 {
            micro.update();
        }

        let offset = micro.get_timing_offset();
        assert!(offset.abs() <= 0.01); // Should be within ±10ms
    }

    #[test]
    fn test_dynamic_range() {
        let mut range = DynamicRange::new(1.0, 6.0); // ±3dB range

        for _ in 0..100 {
            range.update();
        }

        let gain = range.get_gain();
        assert!(gain > 0.0);
        assert!(gain < 2.0); // Should be reasonable
    }
}