use crate::error::{MoodMusicError, Result};

/// Quality settings for synthesis algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynthesisQuality {
    /// Low quality for testing/development
    Low,
    /// Medium quality for general use
    Medium,
    /// High quality for production
    High,
    /// Ultra quality for audiophile applications
    Ultra,
}

impl Default for SynthesisQuality {
    fn default() -> Self {
        SynthesisQuality::Medium
    }
}

/// Configuration for the mood music module
#[derive(Debug, Clone)]
pub struct MoodConfig {
    /// Audio sample rate (typically 44100 or 48000)
    pub sample_rate: u32,

    /// Audio buffer size for processing
    pub buffer_size: usize,

    /// Duration for mood transitions in seconds
    pub transition_duration: f32,

    /// Pattern cycle lengths for each mood type [environmental, gentle, active, edm]
    pub pattern_cycle_lengths: [f32; 4],

    /// Synthesis quality setting
    pub synthesis_quality: SynthesisQuality,

    /// Maximum number of simultaneous voices for polyphonic generators
    pub max_voices: usize,

    /// Enable/disable audio output clipping protection
    pub enable_limiter: bool,

    /// Master volume (0.0 to 1.0)
    pub master_volume: f32,

    /// Enable/disable debug logging for audio thread
    pub enable_audio_debug: bool,
}

impl MoodConfig {
    /// Create a new configuration with default values and specified sample rate
    pub fn default_with_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            buffer_size: 512,
            transition_duration: 4.0,
            pattern_cycle_lengths: [180.0, 120.0, 240.0, 180.0], // 3min, 2min, 4min, 3min
            synthesis_quality: SynthesisQuality::Medium,
            max_voices: 16,
            enable_limiter: true,
            master_volume: 0.7,
            enable_audio_debug: false,
        }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate < 8000 || self.sample_rate > 192000 {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Invalid sample rate: {}. Must be between 8000 and 192000", self.sample_rate)
            ));
        }

        if self.buffer_size < 64 || self.buffer_size > 8192 {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Invalid buffer size: {}. Must be between 64 and 8192", self.buffer_size)
            ));
        }

        if self.transition_duration < 0.1 || self.transition_duration > 30.0 {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Invalid transition duration: {}. Must be between 0.1 and 30.0 seconds", self.transition_duration)
            ));
        }

        for (i, &cycle_length) in self.pattern_cycle_lengths.iter().enumerate() {
            if cycle_length < 10.0 || cycle_length > 600.0 {
                return Err(MoodMusicError::InvalidConfiguration(
                    format!("Invalid pattern cycle length for mood {}: {}. Must be between 10.0 and 600.0 seconds", i, cycle_length)
                ));
            }
        }

        if self.max_voices == 0 || self.max_voices > 128 {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Invalid max voices: {}. Must be between 1 and 128", self.max_voices)
            ));
        }

        if !(0.0..=1.0).contains(&self.master_volume) {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Invalid master volume: {}. Must be between 0.0 and 1.0", self.master_volume)
            ));
        }

        Ok(())
    }

    /// Get the buffer size in seconds
    pub fn buffer_duration(&self) -> f32 {
        self.buffer_size as f32 / self.sample_rate as f32
    }

    /// Get the number of samples per second
    pub fn samples_per_second(&self) -> u32 {
        self.sample_rate
    }

    /// Get the time between samples in seconds
    pub fn sample_duration(&self) -> f64 {
        1.0 / self.sample_rate as f64
    }

    /// Get transition duration in samples
    pub fn transition_samples(&self) -> usize {
        (self.transition_duration * self.sample_rate as f32) as usize
    }

    /// Get pattern cycle lengths in samples
    pub fn pattern_cycle_samples(&self) -> [usize; 4] {
        [
            (self.pattern_cycle_lengths[0] * self.sample_rate as f32) as usize,
            (self.pattern_cycle_lengths[1] * self.sample_rate as f32) as usize,
            (self.pattern_cycle_lengths[2] * self.sample_rate as f32) as usize,
            (self.pattern_cycle_lengths[3] * self.sample_rate as f32) as usize,
        ]
    }

    /// Create a configuration optimized for low latency
    pub fn low_latency(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            buffer_size: 128,
            transition_duration: 2.0,
            pattern_cycle_lengths: [120.0, 90.0, 180.0, 120.0],
            synthesis_quality: SynthesisQuality::Medium,
            max_voices: 8,
            enable_limiter: true,
            master_volume: 0.7,
            enable_audio_debug: false,
        }
    }

    /// Create a configuration optimized for high quality
    pub fn high_quality(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            buffer_size: 1024,
            transition_duration: 6.0,
            pattern_cycle_lengths: [240.0, 180.0, 300.0, 240.0],
            synthesis_quality: SynthesisQuality::Ultra,
            max_voices: 32,
            enable_limiter: true,
            master_volume: 0.7,
            enable_audio_debug: false,
        }
    }
}

impl Default for MoodConfig {
    fn default() -> Self {
        Self::default_with_sample_rate(44100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = MoodConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let mut config = MoodConfig::default();
        config.sample_rate = 1000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_buffer_size() {
        let mut config = MoodConfig::default();
        config.buffer_size = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_buffer_duration_calculation() {
        let config = MoodConfig {
            sample_rate: 44100,
            buffer_size: 441,
            ..Default::default()
        };
        assert!((config.buffer_duration() - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_low_latency_config() {
        let config = MoodConfig::low_latency(48000);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.buffer_size, 128);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_quality_config() {
        let config = MoodConfig::high_quality(96000);
        assert_eq!(config.sample_rate, 96000);
        assert_eq!(config.synthesis_quality, SynthesisQuality::Ultra);
        assert!(config.validate().is_ok());
    }
}