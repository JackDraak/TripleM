use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{MoodGenerator, GeneratorState};

/// Active ambient music generator (0.5-0.75 mood range)
/// Generates productivity-focused ambient music with rhythmic elements
#[derive(Debug)]
pub struct ActiveAmbientGenerator {
    intensity: f32,
    sample_rate: f32,
}

impl ActiveAmbientGenerator {
    pub fn new(config: &MoodConfig) -> Result<Self> {
        Ok(Self {
            intensity: 0.0,
            sample_rate: config.sample_rate as f32,
        })
    }
}

impl MoodGenerator for ActiveAmbientGenerator {
    fn generate_sample(&mut self, _time: f64) -> f32 {
        // Placeholder implementation
        if self.intensity > 0.0 {
            // Simple sine wave with different frequency
            let freq = 330.0; // E4
            let phase = _time * freq * 2.0 * std::f64::consts::PI;
            (phase.sin() as f32) * 0.08 * self.intensity
        } else {
            0.0
        }
    }

    fn generate_batch(&mut self, output: &mut [f32], start_time: f64) {
        let sample_duration = 1.0 / self.sample_rate as f64;
        for (i, sample) in output.iter_mut().enumerate() {
            let time = start_time + i as f64 * sample_duration;
            *sample = self.generate_sample(time);
        }
    }

    fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    fn intensity(&self) -> f32 {
        self.intensity
    }

    fn reset(&mut self) {
        // Reset any internal state
    }

    fn update_focus_parameters(&mut self) {
        // Update parameters when this is the dominant mood
    }

    fn get_state(&self) -> GeneratorState {
        GeneratorState {
            name: "Active Ambient".to_string(),
            intensity: self.intensity,
            is_active: self.intensity > 0.0,
            current_pattern: "Placeholder".to_string(),
            pattern_progress: 0.0,
            cpu_usage: 0.15,
        }
    }
}