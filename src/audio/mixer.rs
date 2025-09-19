use crate::audio::utils::{cosine_interpolate, soft_clip, clamp};

/// Weights for blending different mood generators
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoodWeights {
    pub environmental: f32,
    pub gentle_melodic: f32,
    pub active_ambient: f32,
    pub edm_style: f32,
}

impl MoodWeights {
    /// Create weights for a specific mood value (0.0 to 1.0)
    pub fn from_mood_value(mood: f32) -> Self {
        let mood = clamp(mood, 0.0, 1.0);

        // Define overlapping regions for smooth transitions
        match mood {
            m if m <= 0.25 => Self::environmental_region(m),
            m if m <= 0.5 => Self::gentle_melodic_region(m),
            m if m <= 0.75 => Self::active_ambient_region(m),
            m => Self::edm_style_region(m),
        }
    }

    /// Environmental sound region (0.0 - 0.25)
    fn environmental_region(mood: f32) -> Self {
        let t = mood / 0.25; // Normalize to 0-1
        let environmental = 1.0;
        let gentle_melodic = (t * 2.0).min(1.0) * 0.3; // Small blend at end

        Self {
            environmental,
            gentle_melodic,
            active_ambient: 0.0,
            edm_style: 0.0,
        }
    }

    /// Gentle melodic region (0.25 - 0.5)
    fn gentle_melodic_region(mood: f32) -> Self {
        let t = (mood - 0.25) / 0.25; // Normalize to 0-1
        let environmental = (1.0 - t * 2.0).max(0.0) * 0.3;
        let gentle_melodic = 1.0;
        let active_ambient = (t * 2.0).min(1.0) * 0.3;

        Self {
            environmental,
            gentle_melodic,
            active_ambient,
            edm_style: 0.0,
        }
    }

    /// Active ambient region (0.5 - 0.75)
    fn active_ambient_region(mood: f32) -> Self {
        let t = (mood - 0.5) / 0.25; // Normalize to 0-1
        let gentle_melodic = (1.0 - t * 2.0).max(0.0) * 0.3;
        let active_ambient = 1.0;
        let edm_style = (t * 2.0).min(1.0) * 0.3;

        Self {
            environmental: 0.0,
            gentle_melodic,
            active_ambient,
            edm_style,
        }
    }

    /// EDM style region (0.75 - 1.0)
    fn edm_style_region(mood: f32) -> Self {
        let t = (mood - 0.75) / 0.25; // Normalize to 0-1
        let active_ambient = (1.0 - t * 2.0).max(0.0) * 0.3;
        let edm_style = 1.0;

        Self {
            environmental: 0.0,
            gentle_melodic: 0.0,
            active_ambient,
            edm_style,
        }
    }

    /// Normalize weights so they sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.environmental + self.gentle_melodic + self.active_ambient + self.edm_style;
        if sum > 0.0 {
            self.environmental /= sum;
            self.gentle_melodic /= sum;
            self.active_ambient /= sum;
            self.edm_style /= sum;
        }
    }

    /// Get normalized weights without modifying self
    pub fn normalized(&self) -> Self {
        let mut weights = *self;
        weights.normalize();
        weights
    }

    /// Interpolate between two sets of weights
    pub fn interpolate(&self, other: &Self, t: f32) -> Self {
        let t = clamp(t, 0.0, 1.0);
        Self {
            environmental: cosine_interpolate(self.environmental, other.environmental, t),
            gentle_melodic: cosine_interpolate(self.gentle_melodic, other.gentle_melodic, t),
            active_ambient: cosine_interpolate(self.active_ambient, other.active_ambient, t),
            edm_style: cosine_interpolate(self.edm_style, other.edm_style, t),
        }
    }

    /// Get the dominant mood type
    pub fn dominant_mood(&self) -> MoodType {
        let weights = [
            (MoodType::Environmental, self.environmental),
            (MoodType::GentleMelodic, self.gentle_melodic),
            (MoodType::ActiveAmbient, self.active_ambient),
            (MoodType::EdmStyle, self.edm_style),
        ];

        weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mood_type, _)| *mood_type)
            .unwrap_or(MoodType::Environmental)
    }
}

impl Default for MoodWeights {
    fn default() -> Self {
        Self {
            environmental: 1.0,
            gentle_melodic: 0.0,
            active_ambient: 0.0,
            edm_style: 0.0,
        }
    }
}

/// Enumeration of mood types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MoodType {
    Environmental,
    GentleMelodic,
    ActiveAmbient,
    EdmStyle,
}

/// Output mixer that blends audio from multiple generators
#[derive(Debug)]
pub struct OutputMixer {
    current_weights: MoodWeights,
    master_volume: f32,
    enable_limiter: bool,
    limiter_threshold: f32,

    // Simple peak limiter state
    limiter_envelope: f32,
    limiter_attack: f32,
    limiter_release: f32,
}

impl OutputMixer {
    /// Create a new output mixer
    pub fn new(sample_rate: f32, enable_limiter: bool) -> Self {
        let attack_time = 0.001; // 1ms attack
        let release_time = 0.1;  // 100ms release

        Self {
            current_weights: MoodWeights::default(),
            master_volume: 1.0,
            enable_limiter,
            limiter_threshold: 0.95,
            limiter_envelope: 0.0,
            limiter_attack: (-1.0 / (attack_time * sample_rate)).exp(),
            limiter_release: (-1.0 / (release_time * sample_rate)).exp(),
        }
    }

    /// Set the master volume (0.0 to 1.0)
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = clamp(volume, 0.0, 1.0);
    }

    /// Get the current master volume
    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }

    /// Set the current mood weights
    pub fn set_weights(&mut self, weights: MoodWeights) {
        self.current_weights = weights.normalized();
    }

    /// Get the current mood weights
    pub fn weights(&self) -> &MoodWeights {
        &self.current_weights
    }

    /// Mix audio from all generators into a single output sample
    pub fn mix_sample(
        &mut self,
        environmental: f32,
        gentle_melodic: f32,
        active_ambient: f32,
        edm_style: f32,
    ) -> f32 {
        // Mix based on current weights
        let mixed = environmental * self.current_weights.environmental
            + gentle_melodic * self.current_weights.gentle_melodic
            + active_ambient * self.current_weights.active_ambient
            + edm_style * self.current_weights.edm_style;

        // Apply master volume
        let output = mixed * self.master_volume;

        // Apply limiting if enabled
        if self.enable_limiter {
            self.apply_limiter(output)
        } else {
            soft_clip(output, 0.95)
        }
    }

    /// Mix audio from all generators into a buffer
    pub fn mix_buffer(
        &mut self,
        environmental: &[f32],
        gentle_melodic: &[f32],
        active_ambient: &[f32],
        edm_style: &[f32],
        output: &mut [f32],
    ) {
        let len = output.len().min(environmental.len())
            .min(gentle_melodic.len())
            .min(active_ambient.len())
            .min(edm_style.len());

        for i in 0..len {
            output[i] = self.mix_sample(
                environmental[i],
                gentle_melodic[i],
                active_ambient[i],
                edm_style[i],
            );
        }
    }

    /// Apply peak limiting to prevent clipping
    fn apply_limiter(&mut self, input: f32) -> f32 {
        let input_level = input.abs();
        let target_gain = if input_level > self.limiter_threshold {
            self.limiter_threshold / input_level
        } else {
            1.0
        };

        // Smooth the gain changes
        let coefficient = if target_gain < self.limiter_envelope {
            self.limiter_attack
        } else {
            self.limiter_release
        };

        self.limiter_envelope = target_gain + (self.limiter_envelope - target_gain) * coefficient;

        input * self.limiter_envelope
    }

    /// Reset the limiter state
    pub fn reset_limiter(&mut self) {
        self.limiter_envelope = 0.0;
    }

    /// Enable or disable the limiter
    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.enable_limiter = enabled;
        if !enabled {
            self.reset_limiter();
        }
    }

    /// Set the limiter threshold
    pub fn set_limiter_threshold(&mut self, threshold: f32) {
        self.limiter_threshold = clamp(threshold, 0.1, 1.0);
    }
}

impl Default for OutputMixer {
    fn default() -> Self {
        Self::new(44100.0, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mood_weights_from_value() {
        // Test environmental range
        let weights = MoodWeights::from_mood_value(0.0);
        assert!(weights.environmental > 0.9);
        assert!(weights.gentle_melodic < 0.1);

        // Test gentle melodic range
        let weights = MoodWeights::from_mood_value(0.375);
        assert!(weights.gentle_melodic > 0.9);

        // Test active ambient range
        let weights = MoodWeights::from_mood_value(0.625);
        assert!(weights.active_ambient > 0.9);

        // Test EDM range
        let weights = MoodWeights::from_mood_value(1.0);
        assert!(weights.edm_style > 0.9);
        assert!(weights.active_ambient < 0.1);
    }

    #[test]
    fn test_mood_weights_normalization() {
        let mut weights = MoodWeights {
            environmental: 2.0,
            gentle_melodic: 2.0,
            active_ambient: 2.0,
            edm_style: 2.0,
        };

        weights.normalize();
        let sum = weights.environmental + weights.gentle_melodic + weights.active_ambient + weights.edm_style;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mood_weights_interpolation() {
        let weights1 = MoodWeights::from_mood_value(0.0);
        let weights2 = MoodWeights::from_mood_value(1.0);

        let interpolated = weights1.interpolate(&weights2, 0.5);
        assert!(interpolated.environmental > 0.0);
        assert!(interpolated.edm_style > 0.0);
    }

    #[test]
    fn test_dominant_mood() {
        let weights = MoodWeights {
            environmental: 0.1,
            gentle_melodic: 0.7,
            active_ambient: 0.1,
            edm_style: 0.1,
        };

        assert_eq!(weights.dominant_mood(), MoodType::GentleMelodic);
    }

    #[test]
    fn test_output_mixer() {
        let mut mixer = OutputMixer::new(44100.0, false);
        mixer.set_weights(MoodWeights::from_mood_value(0.5));

        let output = mixer.mix_sample(1.0, 1.0, 1.0, 1.0);
        assert!(output > 0.0);
        assert!(output <= 1.0);
    }

    #[test]
    fn test_limiter() {
        let mut mixer = OutputMixer::new(44100.0, true);
        mixer.set_weights(MoodWeights {
            environmental: 1.0,
            gentle_melodic: 0.0,
            active_ambient: 0.0,
            edm_style: 0.0,
        });

        // Test with a signal that would clip
        let output = mixer.mix_sample(2.0, 0.0, 0.0, 0.0);
        assert!(output < 1.0); // Should be limited
    }

    #[test]
    fn test_master_volume() {
        let mut mixer = OutputMixer::new(44100.0, false);
        mixer.set_master_volume(0.5);
        mixer.set_weights(MoodWeights {
            environmental: 1.0,
            gentle_melodic: 0.0,
            active_ambient: 0.0,
            edm_style: 0.0,
        });

        let output = mixer.mix_sample(1.0, 0.0, 0.0, 0.0);
        assert!((output - 0.5).abs() < 0.01); // Should be halved
    }
}