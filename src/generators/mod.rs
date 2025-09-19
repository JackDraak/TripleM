//! Audio generators for different mood categories

pub mod environmental;
pub mod gentle_melodic;
pub mod active_ambient;
pub mod edm_style;

use crate::config::MoodConfig;
use crate::error::Result;

pub use environmental::EnvironmentalGenerator;
pub use gentle_melodic::GentleMelodicGenerator;
pub use active_ambient::ActiveAmbientGenerator;
pub use edm_style::EdmStyleGenerator;

/// Base trait for all mood generators
pub trait MoodGenerator {
    /// Generate a single audio sample at the given time
    fn generate_sample(&mut self, time: f64) -> f32;

    /// Generate a batch of samples (more efficient)
    fn generate_batch(&mut self, output: &mut [f32], start_time: f64);

    /// Set the intensity level (0.0 to 1.0)
    fn set_intensity(&mut self, intensity: f32);

    /// Get the current intensity level
    fn intensity(&self) -> f32;

    /// Reset the generator to initial state
    fn reset(&mut self);

    /// Update focus parameters (called when this is the dominant mood)
    fn update_focus_parameters(&mut self);

    /// Get the generator's current state for diagnostics
    fn get_state(&self) -> GeneratorState;
}

/// Pool of all mood generators
#[derive(Debug)]
pub struct GeneratorPool {
    pub environmental: EnvironmentalGenerator,
    pub gentle_melodic: GentleMelodicGenerator,
    pub active_ambient: ActiveAmbientGenerator,
    pub edm_style: EdmStyleGenerator,
}

impl GeneratorPool {
    /// Create a new generator pool
    pub fn new(config: &MoodConfig) -> Result<Self> {
        Ok(Self {
            environmental: EnvironmentalGenerator::new(config)?,
            gentle_melodic: GentleMelodicGenerator::new(config)?,
            active_ambient: ActiveAmbientGenerator::new(config)?,
            edm_style: EdmStyleGenerator::new(config)?,
        })
    }

    /// Reset all generators
    pub fn reset(&mut self) {
        self.environmental.reset();
        self.gentle_melodic.reset();
        self.active_ambient.reset();
        self.edm_style.reset();
    }

    /// Get states of all generators
    pub fn get_states(&self) -> GeneratorStates {
        GeneratorStates {
            environmental: self.environmental.get_state(),
            gentle_melodic: self.gentle_melodic.get_state(),
            active_ambient: self.active_ambient.get_state(),
            edm_style: self.edm_style.get_state(),
        }
    }
}

/// State information for a single generator
#[derive(Debug, Clone)]
pub struct GeneratorState {
    pub name: String,
    pub intensity: f32,
    pub is_active: bool,
    pub current_pattern: String,
    pub pattern_progress: f32,
    pub cpu_usage: f32,
}

impl Default for GeneratorState {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            intensity: 0.0,
            is_active: false,
            current_pattern: "None".to_string(),
            pattern_progress: 0.0,
            cpu_usage: 0.0,
        }
    }
}

/// Combined states of all generators
#[derive(Debug, Clone)]
pub struct GeneratorStates {
    pub environmental: GeneratorState,
    pub gentle_melodic: GeneratorState,
    pub active_ambient: GeneratorState,
    pub edm_style: GeneratorState,
}

impl GeneratorStates {
    /// Format all states as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Generator States:\n\
             Environmental: {} (intensity: {:.2})\n\
             Gentle Melodic: {} (intensity: {:.2})\n\
             Active Ambient: {} (intensity: {:.2})\n\
             EDM Style: {} (intensity: {:.2})",
            if self.environmental.is_active { "Active" } else { "Inactive" },
            self.environmental.intensity,
            if self.gentle_melodic.is_active { "Active" } else { "Inactive" },
            self.gentle_melodic.intensity,
            if self.active_ambient.is_active { "Active" } else { "Inactive" },
            self.active_ambient.intensity,
            if self.edm_style.is_active { "Active" } else { "Inactive" },
            self.edm_style.intensity,
        )
    }
}