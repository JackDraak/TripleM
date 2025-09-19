use crate::audio::{OutputMixer, TransitionManager, AudioBuffer};
use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{GeneratorPool, MoodGenerator};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Main audio processing pipeline
#[derive(Debug)]
pub struct AudioPipeline {
    generators: GeneratorPool,
    mixer: OutputMixer,
    transition_manager: TransitionManager,
    config: MoodConfig,

    // Internal state
    is_running: Arc<AtomicBool>,
    current_time: f64,
    sample_count: u64,

    // Buffering
    output_buffer: AudioBuffer,

    // Performance monitoring
    last_cpu_load: f32,
    cpu_load_smoother: f32,
}

impl AudioPipeline {
    /// Create a new audio pipeline
    pub fn new(config: &MoodConfig) -> Result<Self> {
        config.validate()?;

        let generators = GeneratorPool::new(config)?;
        let mixer = OutputMixer::new(config.sample_rate as f32, config.enable_limiter);
        let transition_manager = TransitionManager::new(
            config.sample_rate as f32,
            config.transition_duration,
        );

        // Create output buffer (power of 2 for efficiency)
        let buffer_size = config.buffer_size.next_power_of_two();
        let output_buffer = AudioBuffer::new(buffer_size)?;

        Ok(Self {
            generators,
            mixer,
            transition_manager,
            config: config.clone(),
            is_running: Arc::new(AtomicBool::new(false)),
            current_time: 0.0,
            sample_count: 0,
            output_buffer,
            last_cpu_load: 0.0,
            cpu_load_smoother: 0.0,
        })
    }

    /// Start the audio pipeline
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Relaxed);
    }

    /// Stop the audio pipeline
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Check if the pipeline is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Set the target mood for transition
    pub fn set_target_mood(&mut self, mood: f32) {
        self.transition_manager.start_transition(mood);
    }

    /// Get the next audio sample
    pub fn get_next_sample(&mut self) -> f32 {
        if !self.is_running() {
            return 0.0;
        }

        let start_time = std::time::Instant::now();

        // Update transition state
        self.transition_manager.update();

        // Update mixer weights
        self.mixer.set_weights(*self.transition_manager.current_weights());

        // Generate samples from all generators
        let environmental = self.generators.environmental.generate_sample(self.current_time);
        let gentle_melodic = self.generators.gentle_melodic.generate_sample(self.current_time);
        let active_ambient = self.generators.active_ambient.generate_sample(self.current_time);
        let edm_style = self.generators.edm_style.generate_sample(self.current_time);

        // Mix the samples
        let output = self.mixer.mix_sample(environmental, gentle_melodic, active_ambient, edm_style);

        // Update timing
        self.advance_time();

        // Update CPU load measurement
        let processing_time = start_time.elapsed().as_secs_f32();
        let target_time = 1.0 / self.config.sample_rate as f32;
        let cpu_load = processing_time / target_time;
        self.update_cpu_load(cpu_load);

        output
    }

    /// Fill an audio buffer (more efficient for batch processing)
    pub fn fill_buffer(&mut self, buffer: &mut [f32]) {
        if !self.is_running() {
            buffer.fill(0.0);
            return;
        }

        let start_time = std::time::Instant::now();

        // Generate samples in batches for better performance
        let batch_size = 64.min(buffer.len());
        let mut env_batch = vec![0.0; batch_size];
        let mut gentle_batch = vec![0.0; batch_size];
        let mut active_batch = vec![0.0; batch_size];
        let mut edm_batch = vec![0.0; batch_size];

        for chunk in buffer.chunks_mut(batch_size) {
            let chunk_size = chunk.len();

            // Update transition for this batch
            for _ in 0..chunk_size {
                self.transition_manager.update();
            }

            // Update mixer weights
            self.mixer.set_weights(*self.transition_manager.current_weights());

            // Generate batch samples
            self.generators.environmental.generate_batch(&mut env_batch[..chunk_size], self.current_time);
            self.generators.gentle_melodic.generate_batch(&mut gentle_batch[..chunk_size], self.current_time);
            self.generators.active_ambient.generate_batch(&mut active_batch[..chunk_size], self.current_time);
            self.generators.edm_style.generate_batch(&mut edm_batch[..chunk_size], self.current_time);

            // Mix the batch
            self.mixer.mix_buffer(
                &env_batch[..chunk_size],
                &gentle_batch[..chunk_size],
                &active_batch[..chunk_size],
                &edm_batch[..chunk_size],
                chunk,
            );

            // Update timing for the batch
            for _ in 0..chunk_size {
                self.advance_time();
            }
        }

        // Update CPU load measurement
        let processing_time = start_time.elapsed().as_secs_f32();
        let target_time = buffer.len() as f32 / self.config.sample_rate as f32;
        let cpu_load = processing_time / target_time;
        self.update_cpu_load(cpu_load);
    }

    /// Get current CPU load (0.0 to 1.0+)
    pub fn cpu_load(&self) -> f32 {
        self.cpu_load_smoother
    }

    /// Get current time in seconds
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get current sample count
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get current transition progress
    pub fn transition_progress(&self) -> f32 {
        self.transition_manager.transition_progress()
    }

    /// Check if currently transitioning
    pub fn is_transitioning(&self) -> bool {
        self.transition_manager.is_transitioning()
    }

    /// Get the configuration
    pub fn config(&self) -> &MoodConfig {
        &self.config
    }

    /// Reset all generators and state
    pub fn reset(&mut self) {
        self.generators.reset();
        self.transition_manager.set_weights_immediate(0.0);
        self.mixer.reset_limiter();
        self.current_time = 0.0;
        self.sample_count = 0;
        self.output_buffer.clear();
    }

    /// Update master volume
    pub fn set_master_volume(&mut self, volume: f32) {
        self.mixer.set_master_volume(volume);
    }

    /// Get current master volume
    pub fn master_volume(&self) -> f32 {
        self.mixer.master_volume()
    }

    /// Get current mood weights
    pub fn current_weights(&self) -> &crate::audio::mixer::MoodWeights {
        self.transition_manager.current_weights()
    }

    /// Enable or disable the limiter
    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.mixer.set_limiter_enabled(enabled);
    }

    /// Update generator parameters based on current mood
    pub fn update_generator_parameters(&mut self) {
        let weights = self.transition_manager.current_weights();
        let dominant_mood = weights.dominant_mood();

        // Update each generator with intensity based on its weight
        self.generators.environmental.set_intensity(weights.environmental);
        self.generators.gentle_melodic.set_intensity(weights.gentle_melodic);
        self.generators.active_ambient.set_intensity(weights.active_ambient);
        self.generators.edm_style.set_intensity(weights.edm_style);

        // Additional per-generator updates based on dominant mood
        match dominant_mood {
            crate::audio::mixer::MoodType::Environmental => {
                // Focus on environmental parameters
                self.generators.environmental.update_focus_parameters();
            }
            crate::audio::mixer::MoodType::GentleMelodic => {
                // Focus on melodic parameters
                self.generators.gentle_melodic.update_focus_parameters();
            }
            crate::audio::mixer::MoodType::ActiveAmbient => {
                // Focus on rhythmic parameters
                self.generators.active_ambient.update_focus_parameters();
            }
            crate::audio::mixer::MoodType::EdmStyle => {
                // Focus on electronic parameters
                self.generators.edm_style.update_focus_parameters();
            }
        }
    }

    /// Get generator diagnostics
    pub fn get_diagnostics(&self) -> PipelineDiagnostics {
        PipelineDiagnostics {
            is_running: self.is_running(),
            current_time: self.current_time,
            sample_count: self.sample_count,
            cpu_load: self.cpu_load_smoother,
            is_transitioning: self.is_transitioning(),
            transition_progress: self.transition_progress(),
            current_weights: *self.transition_manager.current_weights(),
            buffer_utilization: self.output_buffer.utilization(),
            generator_states: self.generators.get_states(),
        }
    }

    /// Advance the internal time by one sample
    fn advance_time(&mut self) {
        self.sample_count += 1;
        self.current_time = self.sample_count as f64 / self.config.sample_rate as f64;
    }

    /// Update CPU load with exponential smoothing
    fn update_cpu_load(&mut self, current_load: f32) {
        const SMOOTHING_FACTOR: f32 = 0.95;
        self.last_cpu_load = current_load;
        self.cpu_load_smoother = self.cpu_load_smoother * SMOOTHING_FACTOR
            + current_load * (1.0 - SMOOTHING_FACTOR);
    }
}

/// Diagnostic information about the pipeline state
#[derive(Debug, Clone)]
pub struct PipelineDiagnostics {
    pub is_running: bool,
    pub current_time: f64,
    pub sample_count: u64,
    pub cpu_load: f32,
    pub is_transitioning: bool,
    pub transition_progress: f32,
    pub current_weights: crate::audio::mixer::MoodWeights,
    pub buffer_utilization: f32,
    pub generator_states: crate::generators::GeneratorStates,
}

impl PipelineDiagnostics {
    /// Format diagnostics as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Pipeline Status:\n\
             Running: {}\n\
             Time: {:.2}s (Sample: {})\n\
             CPU Load: {:.1}%\n\
             Transitioning: {} ({:.1}%)\n\
             Weights: E:{:.2} G:{:.2} A:{:.2} EDM:{:.2}\n\
             Buffer: {:.1}% full",
            self.is_running,
            self.current_time,
            self.sample_count,
            self.cpu_load * 100.0,
            self.is_transitioning,
            self.transition_progress * 100.0,
            self.current_weights.environmental,
            self.current_weights.gentle_melodic,
            self.current_weights.active_ambient,
            self.current_weights.edm_style,
            self.buffer_utilization * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = MoodConfig::default();
        let pipeline = AudioPipeline::new(&config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_start_stop() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        assert!(!pipeline.is_running());

        pipeline.start();
        assert!(pipeline.is_running());

        pipeline.stop();
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_sample_generation() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();
        pipeline.set_target_mood(0.5);

        let sample = pipeline.get_next_sample();
        assert!(sample.is_finite());
        assert!(sample >= -1.0 && sample <= 1.0);
    }

    #[test]
    fn test_buffer_fill() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();
        pipeline.set_target_mood(0.3);

        let mut buffer = vec![0.0; 512];
        pipeline.fill_buffer(&mut buffer);

        // Check that buffer was filled with valid audio data
        assert!(buffer.iter().any(|&x| x != 0.0));
        assert!(buffer.iter().all(|&x| x.is_finite() && x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_mood_transition() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();
        pipeline.set_target_mood(0.0);

        // Process some samples
        for _ in 0..100 {
            pipeline.get_next_sample();
        }

        // Start a transition
        pipeline.set_target_mood(1.0);
        assert!(pipeline.is_transitioning());

        let initial_progress = pipeline.transition_progress();

        // Process more samples
        for _ in 0..1000 {
            pipeline.get_next_sample();
        }

        let later_progress = pipeline.transition_progress();
        assert!(later_progress > initial_progress);
    }

    #[test]
    fn test_time_advancement() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();

        let initial_time = pipeline.current_time();
        let initial_count = pipeline.sample_count();

        // Generate some samples
        for _ in 0..1000 {
            pipeline.get_next_sample();
        }

        assert!(pipeline.current_time() > initial_time);
        assert_eq!(pipeline.sample_count(), initial_count + 1000);
    }

    #[test]
    fn test_cpu_load_measurement() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();

        // Generate samples to measure CPU load
        for _ in 0..100 {
            pipeline.get_next_sample();
        }

        let cpu_load = pipeline.cpu_load();
        assert!(cpu_load >= 0.0);
        // CPU load can be > 1.0 if processing is slow
    }

    #[test]
    fn test_pipeline_reset() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();
        pipeline.set_target_mood(0.8);

        // Generate some samples
        for _ in 0..500 {
            pipeline.get_next_sample();
        }

        assert!(pipeline.current_time() > 0.0);
        assert!(pipeline.sample_count() > 0);

        pipeline.reset();

        assert_eq!(pipeline.current_time(), 0.0);
        assert_eq!(pipeline.sample_count(), 0);
    }

    #[test]
    fn test_diagnostics() {
        let config = MoodConfig::default();
        let mut pipeline = AudioPipeline::new(&config).unwrap();

        pipeline.start();
        pipeline.set_target_mood(0.6);

        // Generate some samples
        for _ in 0..200 {
            pipeline.get_next_sample();
        }

        let diagnostics = pipeline.get_diagnostics();
        assert!(diagnostics.is_running);
        assert!(diagnostics.current_time > 0.0);
        assert!(diagnostics.sample_count > 0);

        let formatted = diagnostics.format();
        assert!(formatted.contains("Pipeline Status"));
        assert!(formatted.contains("Running: true"));
    }
}