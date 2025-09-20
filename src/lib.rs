//! # Mood Music Module
//!
//! A procedural mood music module that generates continuous audio streams based on a single
//! 0-1 floating point input parameter. The module supports four distinct mood categories
//! that blend seamlessly based on the input value.
//!
//! ## Usage
//!
//! ```rust
//! use mood_music_module::MoodMusicModule;
//!
//! let mut module = MoodMusicModule::new(44100);
//! module.set_mood(0.5); // Set to active ambient mode
//! module.start();
//!
//! // In your audio callback:
//! let sample = module.get_next_sample();
//! ```

pub mod audio;
pub mod generators;
pub mod patterns;
pub mod config;
pub mod error;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

pub use audio::AudioFrame;
pub use config::MoodConfig;
pub use error::{MoodMusicError, Result};

/// Main interface for the mood music module
#[derive(Debug)]
pub struct MoodMusicModule {
    current_mood: Arc<AtomicU32>, // f32 as u32 bits for atomic access
    audio_pipeline: audio::AudioPipeline,
    is_running: AtomicBool,
    config: MoodConfig,
}

impl MoodMusicModule {
    /// Create a new mood music module with the specified sample rate
    pub fn new(sample_rate: u32) -> Result<Self> {
        let config = MoodConfig::default_with_sample_rate(sample_rate);
        let audio_pipeline = audio::AudioPipeline::new(&config)?;

        Ok(Self {
            current_mood: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            audio_pipeline,
            is_running: AtomicBool::new(false),
            config,
        })
    }

    /// Create a new mood music module with custom configuration
    pub fn with_config(config: MoodConfig) -> Result<Self> {
        let audio_pipeline = audio::AudioPipeline::new(&config)?;

        Ok(Self {
            current_mood: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            audio_pipeline,
            is_running: AtomicBool::new(false),
            config,
        })
    }

    /// Set the mood parameter (0.0 to 1.0)
    /// Values below 0.0 will stop the module
    pub fn set_mood(&mut self, mood: f32) {
        if mood < 0.0 {
            self.stop();
            return;
        }

        let clamped_mood = mood.clamp(0.0, 1.0);
        self.current_mood.store(clamped_mood.to_bits(), Ordering::Relaxed);
        self.audio_pipeline.set_target_mood(clamped_mood);
    }

    /// Get the current mood parameter
    pub fn get_mood(&self) -> f32 {
        f32::from_bits(self.current_mood.load(Ordering::Relaxed))
    }

    /// Set the master volume (0.0 to 1.0)
    pub fn set_volume(&mut self, volume: f32) {
        self.audio_pipeline.set_master_volume(volume.clamp(0.0, 1.0));
    }

    /// Get the current master volume
    pub fn get_volume(&self) -> f32 {
        self.audio_pipeline.master_volume()
    }

    /// Start the audio generation
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Relaxed);
        self.audio_pipeline.start();
    }

    /// Stop the audio generation
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        self.audio_pipeline.stop();
    }

    /// Check if the module is currently running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Get the next audio sample (for real-time playback)
    pub fn get_next_sample(&mut self) -> f32 {
        if !self.is_running() {
            return 0.0;
        }

        self.audio_pipeline.get_next_sample()
    }

    /// Fill an audio buffer (more efficient for batch processing)
    pub fn fill_buffer(&mut self, buffer: &mut [f32]) {
        if !self.is_running() {
            buffer.fill(0.0);
            return;
        }

        self.audio_pipeline.fill_buffer(buffer);
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the current configuration
    pub fn config(&self) -> &MoodConfig {
        &self.config
    }
}

impl Default for MoodMusicModule {
    fn default() -> Self {
        Self::new(44100).expect("Failed to create default MoodMusicModule")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let module = MoodMusicModule::new(44100);
        assert!(module.is_ok());
    }

    #[test]
    fn test_mood_setting() {
        let module = MoodMusicModule::new(44100).unwrap();

        module.set_mood(0.5);
        assert_eq!(module.get_mood(), 0.5);

        module.set_mood(1.5); // Should clamp to 1.0
        assert_eq!(module.get_mood(), 1.0);

        module.set_mood(-0.5); // Should stop the module
        assert!(!module.is_running());
    }

    #[test]
    fn test_start_stop() {
        let module = MoodMusicModule::new(44100).unwrap();

        assert!(!module.is_running());

        module.start();
        assert!(module.is_running());

        module.stop();
        assert!(!module.is_running());
    }
}
