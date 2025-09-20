//! Granular synthesis engine for evolving ambient textures
//!
//! This module implements granular synthesis, a technique for creating
//! complex, evolving textures by manipulating small audio grains.
//! Essential for creating sophisticated ambient soundscapes.

use crate::audio::{NaturalVariation, StateVariableFilter, FilterType};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Granular synthesis engine for creating evolving textures
#[derive(Debug, Clone)]
pub struct GranularEngine {
    sample_rate: f32,
    grains: Vec<Grain>,
    grain_spawn_timer: f32,
    grain_spawn_interval: f32,

    // Synthesis parameters
    base_frequency: f32,
    frequency_spread: f32,
    grain_duration_range: (f32, f32),
    grain_amplitude_range: (f32, f32),

    // Texture evolution
    variation: NaturalVariation,
    texture_filter: StateVariableFilter,
    density_evolution_timer: f32,
    density_evolution_period: f32,

    // Audio buffer for grain generation
    grain_buffer: GrainBuffer,

    // Random number generator
    rng: StdRng,

    // Output controls
    output_gain: f32,
    stereo_spread: f32,
}

/// Individual grain for granular synthesis
#[derive(Debug, Clone)]
struct Grain {
    // Oscillator state
    phase: f32,
    frequency: f32,
    amplitude: f32,

    // Envelope state
    envelope_phase: f32,
    envelope_duration: f32,
    envelope_shape: EnvelopeShape,

    // Spatial positioning
    pan_position: f32,

    // Grain lifecycle
    is_active: bool,
    age: f32,
}

/// Envelope shapes for grain amplitude control
#[derive(Debug, Clone, Copy)]
enum EnvelopeShape {
    Hann,       // Smooth, bell-shaped
    Gaussian,   // Narrow, focused
    Triangle,   // Sharp attack/decay
    Exponential, // Natural decay
}

/// Buffer for generating grain source material
#[derive(Debug, Clone)]
struct GrainBuffer {
    // Wavetable for grain generation
    wavetables: Vec<Wavetable>,
    current_table: usize,
    table_morph_factor: f32,

    // Buffer evolution
    buffer_evolution_timer: f32,
    buffer_evolution_period: f32,
}

/// Wavetable for grain source material
#[derive(Debug, Clone)]
struct Wavetable {
    samples: Vec<f32>,
    name: String,
}

impl GranularEngine {
    /// Create a new granular synthesis engine
    pub fn new(sample_rate: f32, base_frequency: f32) -> Self {
        let mut rng = StdRng::from_entropy();

        // Create natural variation for organic evolution
        let variation = NaturalVariation::new(Some(rng.gen()));

        // Initialize texture filter for overall character
        let texture_filter = StateVariableFilter::new(sample_rate, 1000.0, 0.8);

        // Create grain buffer with multiple wavetables
        let grain_buffer = GrainBuffer::new();

        Self {
            sample_rate,
            grains: Vec::with_capacity(64), // Support up to 64 simultaneous grains
            grain_spawn_timer: 0.0,
            grain_spawn_interval: 0.1, // Spawn every 100ms initially

            base_frequency,
            frequency_spread: 0.2, // ±20% frequency variation
            grain_duration_range: (0.05, 0.5), // 50ms to 500ms grain duration
            grain_amplitude_range: (0.1, 0.8),

            variation,
            texture_filter,
            density_evolution_timer: 0.0,
            density_evolution_period: 10.0, // 10-second evolution cycles

            grain_buffer,
            rng,

            output_gain: 0.3,
            stereo_spread: 0.6,
        }
    }

    /// Process a single sample of granular synthesis
    pub fn process(&mut self) -> (f32, f32) { // Returns (left, right)
        let dt = 1.0 / self.sample_rate;

        // Update natural variation for organic parameter evolution
        self.variation.update();

        // Update texture evolution
        self.update_texture_evolution(dt);

        // Spawn new grains based on density
        self.update_grain_spawning(dt);

        // Process all active grains
        let (left, right) = self.process_grains(dt);

        // Apply texture filtering
        let filtered_left = self.texture_filter.process_type(left, FilterType::Lowpass);
        let filtered_right = self.texture_filter.process_type(right, FilterType::Lowpass);

        // Apply output gain with natural variation
        let gain_variation = self.variation.get_amplitude_multiplier(0.1);
        let final_gain = self.output_gain * gain_variation;

        (filtered_left * final_gain, filtered_right * final_gain)
    }

    /// Update texture evolution parameters
    fn update_texture_evolution(&mut self, dt: f32) {
        self.density_evolution_timer += dt;

        if self.density_evolution_timer >= self.density_evolution_period {
            self.density_evolution_timer = 0.0;

            // Evolve synthesis parameters using natural variation
            let density_drift = self.variation.get_timing_variation();
            self.grain_spawn_interval = (0.05 + density_drift * 0.1).max(0.01);

            let frequency_drift = self.variation.get_pitch_variation();
            self.base_frequency *= 1.0 + frequency_drift * 0.1;

            // Update filter characteristics
            let timbre_drift = self.variation.get_timbre_drift(1000.0, 0.3);
            self.texture_filter.set_cutoff(timbre_drift);
        }
    }

    /// Update grain spawning based on density parameters
    fn update_grain_spawning(&mut self, dt: f32) {
        self.grain_spawn_timer += dt;

        // Apply natural variation to spawn timing
        let timing_variation = self.variation.get_duration_multiplier(0.2);
        let varied_interval = self.grain_spawn_interval * timing_variation;

        if self.grain_spawn_timer >= varied_interval {
            self.grain_spawn_timer = 0.0;
            self.spawn_grain();
        }
    }

    /// Spawn a new grain
    fn spawn_grain(&mut self) {
        // Remove inactive grains to make room
        self.grains.retain(|grain| grain.is_active);

        // Don't spawn if we're at capacity
        if self.grains.len() >= 64 {
            return;
        }

        // Generate grain parameters with variation
        let frequency_variation = 1.0 + (self.rng.gen::<f32>() - 0.5) * self.frequency_spread;
        let frequency = self.base_frequency * frequency_variation;

        let amplitude = self.rng.gen_range(
            self.grain_amplitude_range.0..=self.grain_amplitude_range.1
        );

        let duration = self.rng.gen_range(
            self.grain_duration_range.0..=self.grain_duration_range.1
        );

        let pan_position = (self.rng.gen::<f32>() - 0.5) * self.stereo_spread;

        let envelope_shape = match self.rng.gen_range(0..4) {
            0 => EnvelopeShape::Hann,
            1 => EnvelopeShape::Gaussian,
            2 => EnvelopeShape::Triangle,
            _ => EnvelopeShape::Exponential,
        };

        let grain = Grain {
            phase: self.rng.gen::<f32>(), // Random starting phase
            frequency,
            amplitude,
            envelope_phase: 0.0,
            envelope_duration: duration,
            envelope_shape,
            pan_position,
            is_active: true,
            age: 0.0,
        };

        self.grains.push(grain);
    }

    /// Process all active grains and mix their output
    fn process_grains(&mut self, dt: f32) -> (f32, f32) {
        let mut left_output = 0.0;
        let mut right_output = 0.0;

        for grain in &mut self.grains {
            if !grain.is_active {
                continue;
            }

            // Generate grain audio (inline to avoid borrowing issues)
            let grain_sample = {
                let sample = self.grain_buffer.get_sample(grain.phase);
                grain.phase += grain.frequency / self.sample_rate;
                if grain.phase >= 1.0 {
                    grain.phase -= 1.0;
                }
                sample * grain.amplitude
            };

            // Apply envelope (inline calculation)
            let envelope_value = {
                let phase = grain.envelope_phase.clamp(0.0, 1.0);
                match grain.envelope_shape {
                    EnvelopeShape::Hann => {
                        0.5 * (1.0 - (2.0 * std::f32::consts::PI * phase).cos())
                    },
                    EnvelopeShape::Gaussian => {
                        let x = (phase - 0.5) * 4.0;
                        (-x * x).exp()
                    },
                    EnvelopeShape::Triangle => {
                        if phase < 0.5 {
                            phase * 2.0
                        } else {
                            2.0 - phase * 2.0
                        }
                    },
                    EnvelopeShape::Exponential => {
                        if phase < 0.1 {
                            phase * 10.0
                        } else {
                            (-5.0 * (phase - 0.1)).exp()
                        }
                    },
                }
            };

            let shaped_sample = grain_sample * envelope_value;

            // Apply stereo panning (inline calculation)
            let (left_gain, right_gain) = {
                let pan = grain.pan_position.clamp(-1.0, 1.0);
                let left_gain = (1.0 - pan.max(0.0)).sqrt();
                let right_gain = (1.0 + pan.min(0.0)).sqrt();
                (left_gain, right_gain)
            };

            left_output += shaped_sample * left_gain;
            right_output += shaped_sample * right_gain;

            // Update grain state
            grain.age += dt;
            grain.envelope_phase += dt / grain.envelope_duration;

            // Deactivate completed grains
            if grain.envelope_phase >= 1.0 {
                grain.is_active = false;
            }
        }

        // Normalize by number of active grains to prevent clipping
        let active_grain_count = self.grains.iter().filter(|g| g.is_active).count();
        if active_grain_count > 0 {
            let normalize_factor = 1.0 / (active_grain_count as f32).sqrt();
            left_output *= normalize_factor;
            right_output *= normalize_factor;
        }

        (left_output, right_output)
    }


    /// Set the base frequency for grain generation
    pub fn set_base_frequency(&mut self, frequency: f32) {
        self.base_frequency = frequency.clamp(20.0, 2000.0);
    }

    /// Set the grain density (spawning rate)
    pub fn set_density(&mut self, density: f32) {
        // density: 0.0 = sparse, 1.0 = dense
        let density_clamped = density.clamp(0.0, 1.0);
        self.grain_spawn_interval = 0.5 - density_clamped * 0.45; // 0.05s to 0.5s
    }

    /// Set the frequency spread for grain variation
    pub fn set_frequency_spread(&mut self, spread: f32) {
        self.frequency_spread = spread.clamp(0.0, 1.0);
    }

    /// Set the output gain
    pub fn set_output_gain(&mut self, gain: f32) {
        self.output_gain = gain.clamp(0.0, 1.0);
    }

    /// Get the number of active grains
    pub fn active_grain_count(&self) -> usize {
        self.grains.iter().filter(|g| g.is_active).count()
    }

    /// Reset the granular engine
    pub fn reset(&mut self) {
        self.grains.clear();
        self.grain_spawn_timer = 0.0;
        self.density_evolution_timer = 0.0;
        self.texture_filter.reset();
        self.variation.reset();
        self.grain_buffer.reset();
    }
}

impl GrainBuffer {
    /// Create a new grain buffer with default wavetables
    fn new() -> Self {
        let wavetables = vec![
            Self::create_sine_wavetable(),
            Self::create_sawtooth_wavetable(),
            Self::create_triangle_wavetable(),
            Self::create_noise_wavetable(),
            Self::create_harmonic_wavetable(),
        ];

        Self {
            wavetables,
            current_table: 0,
            table_morph_factor: 0.0,
            buffer_evolution_timer: 0.0,
            buffer_evolution_period: 15.0, // 15-second wavetable evolution
        }
    }

    /// Get interpolated sample from current wavetable(s)
    fn get_sample(&self, phase: f32) -> f32 {
        let table_size = self.wavetables[0].samples.len();
        let index = (phase * table_size as f32) % table_size as f32;
        let index_floor = index.floor() as usize;
        let index_frac = index.fract();

        // Linear interpolation within wavetable
        let current_table = &self.wavetables[self.current_table];
        let sample1 = current_table.samples[index_floor];
        let sample2 = current_table.samples[(index_floor + 1) % table_size];
        let interpolated = sample1 + (sample2 - sample1) * index_frac;

        // TODO: Add wavetable morphing for evolving textures
        interpolated
    }

    /// Reset the grain buffer
    fn reset(&mut self) {
        self.current_table = 0;
        self.table_morph_factor = 0.0;
        self.buffer_evolution_timer = 0.0;
    }

    /// Create sine wavetable
    fn create_sine_wavetable() -> Wavetable {
        let size = 1024;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;
            samples.push((phase * 2.0 * std::f32::consts::PI).sin());
        }

        Wavetable {
            samples,
            name: "Sine".to_string(),
        }
    }

    /// Create sawtooth wavetable
    fn create_sawtooth_wavetable() -> Wavetable {
        let size = 1024;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;
            samples.push(phase * 2.0 - 1.0);
        }

        Wavetable {
            samples,
            name: "Sawtooth".to_string(),
        }
    }

    /// Create triangle wavetable
    fn create_triangle_wavetable() -> Wavetable {
        let size = 1024;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;
            let sample = if phase < 0.5 {
                phase * 4.0 - 1.0
            } else {
                3.0 - phase * 4.0
            };
            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Triangle".to_string(),
        }
    }

    /// Create noise wavetable
    fn create_noise_wavetable() -> Wavetable {
        let size = 1024;
        let mut samples = Vec::with_capacity(size);
        let mut rng = StdRng::from_entropy();

        for _ in 0..size {
            samples.push((rng.gen::<f32>() - 0.5) * 2.0);
        }

        Wavetable {
            samples,
            name: "Noise".to_string(),
        }
    }

    /// Create harmonic wavetable (complex overtones)
    fn create_harmonic_wavetable() -> Wavetable {
        let size = 1024;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32 * 2.0 * std::f32::consts::PI;

            // Additive synthesis with harmonics
            let mut sample = 0.0;
            sample += 1.0 * phase.sin();                    // Fundamental
            sample += 0.5 * (phase * 2.0).sin();           // 2nd harmonic
            sample += 0.3 * (phase * 3.0).sin();           // 3rd harmonic
            sample += 0.2 * (phase * 4.0).sin();           // 4th harmonic
            sample += 0.1 * (phase * 5.0).sin();           // 5th harmonic

            samples.push(sample * 0.3); // Normalize
        }

        Wavetable {
            samples,
            name: "Harmonic".to_string(),
        }
    }
}

/// Presets for different types of granular textures
pub mod presets {
    use super::*;

    /// Create a gentle ambient texture suitable for spa music
    pub fn gentle_ambient(sample_rate: f32) -> GranularEngine {
        let mut engine = GranularEngine::new(sample_rate, 220.0); // A3
        engine.set_density(0.3); // Sparse, contemplative
        engine.set_frequency_spread(0.1); // Subtle variation
        engine.set_output_gain(0.2); // Quiet background texture
        engine
    }

    /// Create an active texture for productivity music
    pub fn active_texture(sample_rate: f32) -> GranularEngine {
        let mut engine = GranularEngine::new(sample_rate, 440.0); // A4
        engine.set_density(0.6); // Moderate density
        engine.set_frequency_spread(0.3); // More variation
        engine.set_output_gain(0.25); // Noticeable but not overwhelming
        engine
    }

    /// Create an evolving texture for electronic music
    pub fn electronic_texture(sample_rate: f32) -> GranularEngine {
        let mut engine = GranularEngine::new(sample_rate, 880.0); // A5
        engine.set_density(0.8); // Dense, complex
        engine.set_frequency_spread(0.5); // Wide variation
        engine.set_output_gain(0.3); // Prominent texture
        engine
    }

    /// Create a sub-bass texture for deep ambient
    pub fn sub_bass_texture(sample_rate: f32) -> GranularEngine {
        let mut engine = GranularEngine::new(sample_rate, 55.0); // A1
        engine.set_density(0.2); // Very sparse
        engine.set_frequency_spread(0.05); // Minimal variation for stability
        engine.set_output_gain(0.4); // Strong presence in low end
        engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granular_engine_creation() {
        let engine = GranularEngine::new(44100.0, 440.0);
        assert_eq!(engine.sample_rate, 44100.0);
        assert_eq!(engine.base_frequency, 440.0);
        assert_eq!(engine.active_grain_count(), 0);
    }

    #[test]
    fn test_grain_spawning() {
        let mut engine = GranularEngine::new(44100.0, 440.0);
        engine.set_density(1.0); // Maximum density

        // Process enough samples to trigger grain spawning
        for _ in 0..10000 {
            engine.process();
        }

        assert!(engine.active_grain_count() > 0);
    }

    #[test]
    fn test_stereo_output() {
        let mut engine = GranularEngine::new(44100.0, 440.0);
        engine.spawn_grain(); // Force spawn a grain

        let (left, right) = engine.process();
        assert!(left.is_finite());
        assert!(right.is_finite());
    }

    #[test]
    fn test_parameter_clamping() {
        let mut engine = GranularEngine::new(44100.0, 440.0);

        engine.set_density(2.0); // Should clamp to 1.0
        engine.set_frequency_spread(-0.5); // Should clamp to 0.0
        engine.set_output_gain(1.5); // Should clamp to 1.0

        // Engine should still function normally after invalid inputs
        let (left, right) = engine.process();
        assert!(left.is_finite());
        assert!(right.is_finite());
    }

    #[test]
    fn test_preset_creation() {
        let sample_rate = 44100.0;

        let _gentle = presets::gentle_ambient(sample_rate);
        let _active = presets::active_texture(sample_rate);
        let _electronic = presets::electronic_texture(sample_rate);
        let _sub_bass = presets::sub_bass_texture(sample_rate);

        // Just test that they can be created without panicking
    }

    #[test]
    fn test_wavetable_generation() {
        let sine_table = GrainBuffer::create_sine_wavetable();
        assert_eq!(sine_table.samples.len(), 1024);
        assert_eq!(sine_table.name, "Sine");

        let noise_table = GrainBuffer::create_noise_wavetable();
        assert_eq!(noise_table.samples.len(), 1024);
        assert_eq!(noise_table.name, "Noise");
    }
}