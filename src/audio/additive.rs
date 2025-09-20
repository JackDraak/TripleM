//! Advanced additive synthesis engine for complex polyharmonic content
//!
//! This module provides sophisticated additive synthesis with support for
//! harmonic and inharmonic partials, dynamic spectral evolution, and
//! organic variation for creating rich, evolving timbres.

use crate::audio::{NaturalVariation, StateVariableFilter, FilterType};
use std::f32::consts::TAU;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Advanced additive synthesizer with dynamic harmonic control
#[derive(Debug, Clone)]
pub struct AdditiveSynthesizer {
    /// Collection of harmonic oscillators
    harmonics: Vec<HarmonicOscillator>,

    /// Base frequency for the harmonic series
    fundamental_frequency: f32,

    /// Overall amplitude
    amplitude: f32,

    /// Spectral envelope for shaping harmonic content
    spectral_envelope: SpectralEnvelope,

    /// Natural variation for organic evolution
    variation: NaturalVariation,

    /// Harmonic evolution control
    harmonic_evolution: HarmonicEvolution,

    /// Output filtering and processing
    output_processor: OutputProcessor,

    sample_rate: f32,
}

/// Individual harmonic oscillator with independent control
#[derive(Debug, Clone)]
pub struct HarmonicOscillator {
    /// Phase accumulator
    phase: f32,

    /// Frequency ratio relative to fundamental (1.0 = fundamental, 2.0 = octave, etc.)
    frequency_ratio: f32,

    /// Base amplitude (before envelope and variation)
    base_amplitude: f32,

    /// Current amplitude (after processing)
    current_amplitude: f32,

    /// Frequency detuning for organic character
    detuning: f32,

    /// Phase offset for stereo and ensemble effects
    phase_offset: f32,

    /// Individual envelope for this harmonic
    envelope: HarmonicEnvelope,
}

/// Spectral envelope for shaping overall harmonic content
#[derive(Debug, Clone)]
pub struct SpectralEnvelope {
    /// Spectral tilt (negative = darker, positive = brighter)
    tilt: f32,

    /// High-frequency rolloff
    rolloff_frequency: f32,
    rolloff_steepness: f32,

    /// Formant-like resonances
    formants: Vec<SpectralFormant>,

    /// Dynamic evolution rate
    evolution_rate: f32,
}

/// Formant-like resonance in the spectral envelope
#[derive(Debug, Clone)]
pub struct SpectralFormant {
    /// Center frequency as ratio of fundamental
    frequency_ratio: f32,

    /// Bandwidth (Q factor)
    q_factor: f32,

    /// Gain (dB)
    gain: f32,

    /// Natural drift amount
    drift_amount: f32,
}

/// Individual harmonic envelope control
#[derive(Debug, Clone)]
pub struct HarmonicEnvelope {
    /// Attack time in seconds
    attack: f32,

    /// Decay time in seconds
    decay: f32,

    /// Sustain level (0.0 to 1.0)
    sustain: f32,

    /// Release time in seconds
    release: f32,

    /// Current envelope phase
    phase: EnvelopePhase,

    /// Time in current phase
    phase_time: f32,

    /// Current envelope value
    value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvelopePhase {
    Off,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Controls harmonic evolution over time
#[derive(Debug, Clone)]
pub struct HarmonicEvolution {
    /// Evolution speed (how fast harmonics change)
    speed: f32,

    /// Evolution depth (how much harmonics change)
    depth: f32,

    /// Current evolution phase
    phase: f32,

    /// Evolution patterns for different harmonic types
    patterns: Vec<EvolutionPattern>,

    /// Current pattern index
    current_pattern: usize,
}

/// Pattern for harmonic evolution
#[derive(Debug, Clone)]
pub struct EvolutionPattern {
    name: String,

    /// How this pattern affects harmonic amplitudes over time
    amplitude_modulation: Vec<f32>,

    /// How this pattern affects harmonic frequencies over time
    frequency_modulation: Vec<f32>,

    /// Pattern length in seconds
    duration: f32,
}

/// Output processing for the additive synthesizer
#[derive(Debug, Clone)]
pub struct OutputProcessor {
    /// Main filter for tone shaping
    main_filter: StateVariableFilter,

    /// Stereo width control
    stereo_width: f32,

    /// Harmonic saturation for richness
    saturation: HarmonicSaturation,

    /// Final gain compensation
    gain_compensation: f32,
}

/// Harmonic saturation for adding warmth and richness
#[derive(Debug, Clone)]
pub struct HarmonicSaturation {
    /// Drive amount
    drive: f32,

    /// Saturation type
    saturation_type: SaturationType,

    /// Output gain compensation
    output_gain: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum SaturationType {
    Warm,      // Tube-like even harmonics
    Bright,    // Tape-like odd harmonics
    Rich,      // Complex harmonic generation
    Clean,     // Minimal saturation
}

impl AdditiveSynthesizer {
    /// Create a new additive synthesizer
    pub fn new(sample_rate: f32, num_harmonics: usize) -> Self {
        let variation = NaturalVariation::new(None);

        // Create harmonic oscillators
        let harmonics = (1..=num_harmonics)
            .map(|i| HarmonicOscillator::new(i as f32, 1.0 / i as f32))
            .collect();

        Self {
            harmonics,
            fundamental_frequency: 440.0,
            amplitude: 1.0,
            spectral_envelope: SpectralEnvelope::new(),
            variation,
            harmonic_evolution: HarmonicEvolution::new(),
            output_processor: OutputProcessor::new(sample_rate),
            sample_rate,
        }
    }

    /// Process the next sample
    pub fn process(&mut self) -> f32 {
        // Update natural variation
        self.variation.update();

        // Update harmonic evolution
        self.harmonic_evolution.update();

        // Generate harmonic content
        let mut output = 0.0;

        for (i, harmonic) in self.harmonics.iter_mut().enumerate() {
            // Calculate harmonic frequency with variation
            let frequency_variation = self.variation.get_pitch_variation();
            let evolved_ratio = self.harmonic_evolution.get_frequency_ratio(i);
            let harmonic_frequency = self.fundamental_frequency *
                evolved_ratio * (1.0 + frequency_variation * 0.01);

            // Calculate harmonic amplitude with spectral envelope and evolution
            let spectral_gain = self.spectral_envelope.get_gain_for_harmonic(i, harmonic_frequency);
            let evolved_amplitude = self.harmonic_evolution.get_amplitude_ratio(i);
            let final_amplitude = harmonic.base_amplitude * spectral_gain * evolved_amplitude;

            // Update and get harmonic sample
            let harmonic_sample = harmonic.process(harmonic_frequency, final_amplitude, self.sample_rate);
            output += harmonic_sample;
        }

        // Apply amplitude with natural variation
        let amplitude_variation = self.variation.get_amplitude_variation();
        output *= self.amplitude * (1.0 + amplitude_variation * 0.1);

        // Process through output chain
        self.output_processor.process(output)
    }

    /// Set the fundamental frequency
    pub fn set_frequency(&mut self, frequency: f32) {
        self.fundamental_frequency = frequency.clamp(20.0, 5000.0);
    }

    /// Set overall amplitude
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Set spectral tilt (-1.0 = dark, 1.0 = bright)
    pub fn set_spectral_tilt(&mut self, tilt: f32) {
        self.spectral_envelope.tilt = tilt.clamp(-1.0, 1.0);
    }

    /// Set harmonic evolution speed
    pub fn set_evolution_speed(&mut self, speed: f32) {
        self.harmonic_evolution.speed = speed.clamp(0.01, 10.0);
    }

    /// Set harmonic evolution depth
    pub fn set_evolution_depth(&mut self, depth: f32) {
        self.harmonic_evolution.depth = depth.clamp(0.0, 1.0);
    }

    /// Trigger note on
    pub fn note_on(&mut self) {
        for harmonic in &mut self.harmonics {
            harmonic.envelope.trigger();
        }
    }

    /// Trigger note off
    pub fn note_off(&mut self) {
        for harmonic in &mut self.harmonics {
            harmonic.envelope.release();
        }
    }

    /// Create synthesizer optimized for specific character
    pub fn for_character(sample_rate: f32, character: PolyharmonicCharacter) -> Self {
        let num_harmonics = match character {
            PolyharmonicCharacter::Gentle => 8,
            PolyharmonicCharacter::Rich => 16,
            PolyharmonicCharacter::Complex => 32,
            PolyharmonicCharacter::Crystalline => 24,
            PolyharmonicCharacter::Organic => 12,
        };

        let mut synth = Self::new(sample_rate, num_harmonics);

        match character {
            PolyharmonicCharacter::Gentle => {
                synth.set_spectral_tilt(-0.3);
                synth.set_evolution_speed(0.05);
                synth.set_evolution_depth(0.2);
            },
            PolyharmonicCharacter::Rich => {
                synth.set_spectral_tilt(0.1);
                synth.set_evolution_speed(0.1);
                synth.set_evolution_depth(0.4);
            },
            PolyharmonicCharacter::Complex => {
                synth.set_spectral_tilt(0.3);
                synth.set_evolution_speed(0.2);
                synth.set_evolution_depth(0.8);
            },
            PolyharmonicCharacter::Crystalline => {
                synth.set_spectral_tilt(0.6);
                synth.set_evolution_speed(0.15);
                synth.set_evolution_depth(0.3);
            },
            PolyharmonicCharacter::Organic => {
                synth.set_spectral_tilt(-0.1);
                synth.set_evolution_speed(0.03);
                synth.set_evolution_depth(0.6);
            },
        }

        synth
    }
}

/// Character types for polyharmonic synthesis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolyharmonicCharacter {
    Gentle,      // Soft, few harmonics
    Rich,        // Warm, balanced harmonics
    Complex,     // Many harmonics, evolving
    Crystalline, // Bright, clear harmonics
    Organic,     // Natural, irregular harmonics
}

impl HarmonicOscillator {
    fn new(frequency_ratio: f32, base_amplitude: f32) -> Self {
        Self {
            phase: 0.0,
            frequency_ratio,
            base_amplitude,
            current_amplitude: 0.0,
            detuning: 0.0,
            phase_offset: 0.0,
            envelope: HarmonicEnvelope::new(0.01, 0.1, 0.8, 0.2),
        }
    }

    fn process(&mut self, fundamental_freq: f32, target_amplitude: f32, sample_rate: f32) -> f32 {
        // Update envelope
        self.envelope.update(sample_rate);

        // Calculate frequency with detuning
        let frequency = fundamental_freq * self.frequency_ratio * (1.0 + self.detuning);

        // Generate sine wave
        let sample = (self.phase * TAU).sin();

        // Update phase
        self.phase += frequency / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Apply envelope and amplitude
        sample * target_amplitude * self.envelope.value
    }
}

impl HarmonicEnvelope {
    fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Self {
            attack,
            decay,
            sustain,
            release,
            phase: EnvelopePhase::Off,
            phase_time: 0.0,
            value: 0.0,
        }
    }

    fn trigger(&mut self) {
        self.phase = EnvelopePhase::Attack;
        self.phase_time = 0.0;
    }

    fn release(&mut self) {
        self.phase = EnvelopePhase::Release;
        self.phase_time = 0.0;
    }

    fn update(&mut self, sample_rate: f32) {
        let dt = 1.0 / sample_rate;
        self.phase_time += dt;

        match self.phase {
            EnvelopePhase::Off => {
                self.value = 0.0;
            },
            EnvelopePhase::Attack => {
                if self.attack > 0.0 {
                    self.value = (self.phase_time / self.attack).min(1.0);
                    if self.phase_time >= self.attack {
                        self.phase = EnvelopePhase::Decay;
                        self.phase_time = 0.0;
                    }
                } else {
                    self.value = 1.0;
                    self.phase = EnvelopePhase::Decay;
                }
            },
            EnvelopePhase::Decay => {
                if self.decay > 0.0 {
                    let decay_progress = (self.phase_time / self.decay).min(1.0);
                    self.value = 1.0 - decay_progress * (1.0 - self.sustain);
                    if self.phase_time >= self.decay {
                        self.phase = EnvelopePhase::Sustain;
                        self.phase_time = 0.0;
                    }
                } else {
                    self.value = self.sustain;
                    self.phase = EnvelopePhase::Sustain;
                }
            },
            EnvelopePhase::Sustain => {
                self.value = self.sustain;
            },
            EnvelopePhase::Release => {
                if self.release > 0.0 {
                    let release_progress = (self.phase_time / self.release).min(1.0);
                    self.value = self.sustain * (1.0 - release_progress);
                    if self.phase_time >= self.release {
                        self.phase = EnvelopePhase::Off;
                        self.phase_time = 0.0;
                    }
                } else {
                    self.value = 0.0;
                    self.phase = EnvelopePhase::Off;
                }
            },
        }
    }
}

impl SpectralEnvelope {
    fn new() -> Self {
        Self {
            tilt: 0.0,
            rolloff_frequency: 8000.0,
            rolloff_steepness: -12.0, // dB per octave
            formants: vec![
                SpectralFormant { frequency_ratio: 2.5, q_factor: 3.0, gain: 2.0, drift_amount: 0.1 },
                SpectralFormant { frequency_ratio: 6.2, q_factor: 4.0, gain: 1.0, drift_amount: 0.15 },
            ],
            evolution_rate: 0.1,
        }
    }

    fn get_gain_for_harmonic(&self, harmonic_index: usize, frequency: f32) -> f32 {
        let mut gain = 1.0;

        // Apply spectral tilt
        let harmonic_number = (harmonic_index + 1) as f32;
        let tilt_gain = (1.0 + self.tilt * 0.1).powf(harmonic_number.log2());
        gain *= tilt_gain;

        // Apply high-frequency rolloff
        if frequency > self.rolloff_frequency {
            let octaves_above = (frequency / self.rolloff_frequency).log2();
            let rolloff_gain = (self.rolloff_steepness * octaves_above / 20.0).exp2();
            gain *= rolloff_gain;
        }

        // Apply formant resonances
        for formant in &self.formants {
            let formant_freq = 440.0 * formant.frequency_ratio; // Base on A4
            let distance = (frequency - formant_freq).abs();
            let bandwidth = formant_freq / formant.q_factor;

            if distance < bandwidth * 2.0 {
                let resonance_gain = 1.0 + formant.gain *
                    (1.0 - (distance / bandwidth).min(1.0));
                gain *= resonance_gain;
            }
        }

        gain.max(0.0)
    }
}

impl HarmonicEvolution {
    fn new() -> Self {
        Self {
            speed: 0.1,
            depth: 0.5,
            phase: 0.0,
            patterns: vec![
                EvolutionPattern::gentle_waves(),
                EvolutionPattern::complex_evolution(),
                EvolutionPattern::crystalline_shifts(),
            ],
            current_pattern: 0,
        }
    }

    fn update(&mut self) {
        self.phase += self.speed * 0.01; // Slow evolution
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            // Occasionally switch patterns
            if rand::random::<f32>() < 0.1 {
                self.current_pattern = (self.current_pattern + 1) % self.patterns.len();
            }
        }
    }

    fn get_amplitude_ratio(&self, harmonic_index: usize) -> f32 {
        let pattern = &self.patterns[self.current_pattern];
        if let Some(&base_amp) = pattern.amplitude_modulation.get(harmonic_index) {
            let modulation = (self.phase * TAU * (harmonic_index + 1) as f32 * 0.3).sin();
            base_amp * (1.0 + modulation * self.depth * 0.2)
        } else {
            1.0 / (harmonic_index + 1) as f32 // Default harmonic series
        }
    }

    fn get_frequency_ratio(&self, harmonic_index: usize) -> f32 {
        let pattern = &self.patterns[self.current_pattern];
        if let Some(&base_freq) = pattern.frequency_modulation.get(harmonic_index) {
            let modulation = (self.phase * TAU * (harmonic_index + 1) as f32 * 0.1).sin();
            base_freq * (1.0 + modulation * self.depth * 0.02)
        } else {
            (harmonic_index + 1) as f32 // Default harmonic series
        }
    }
}

impl EvolutionPattern {
    fn gentle_waves() -> Self {
        Self {
            name: "Gentle Waves".to_string(),
            amplitude_modulation: vec![1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08],
            frequency_modulation: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            duration: 10.0,
        }
    }

    fn complex_evolution() -> Self {
        Self {
            name: "Complex Evolution".to_string(),
            amplitude_modulation: vec![1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08],
            frequency_modulation: vec![1.0, 2.01, 3.02, 4.03, 5.05, 6.07, 7.08, 8.1, 9.12, 10.15, 11.18, 12.2],
            duration: 20.0,
        }
    }

    fn crystalline_shifts() -> Self {
        Self {
            name: "Crystalline Shifts".to_string(),
            amplitude_modulation: vec![1.0, 0.5, 0.8, 0.3, 0.6, 0.2, 0.4, 0.15],
            frequency_modulation: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            duration: 15.0,
        }
    }
}

impl OutputProcessor {
    fn new(sample_rate: f32) -> Self {
        Self {
            main_filter: StateVariableFilter::new(sample_rate, 12000.0, 0.7),
            stereo_width: 0.5,
            saturation: HarmonicSaturation::new(SaturationType::Warm),
            gain_compensation: 0.3, // Compensate for multiple harmonics
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        // Apply harmonic saturation
        let saturated = self.saturation.process(input);

        // Apply main filter
        let filtered = self.main_filter.process_type(saturated, FilterType::Lowpass);

        // Apply gain compensation
        filtered * self.gain_compensation
    }
}

impl HarmonicSaturation {
    fn new(saturation_type: SaturationType) -> Self {
        Self {
            drive: 1.2,
            saturation_type,
            output_gain: 0.8,
        }
    }

    fn process(&self, input: f32) -> f32 {
        let driven = input * self.drive;

        let saturated = match self.saturation_type {
            SaturationType::Warm => {
                // Soft tube-like saturation
                if driven.abs() < 0.5 {
                    driven
                } else {
                    driven.signum() * (0.5 + (driven.abs() - 0.5).tanh() * 0.3)
                }
            },
            SaturationType::Bright => {
                // Tape-like saturation with slight asymmetry
                (driven * 1.2).tanh() * 0.9
            },
            SaturationType::Rich => {
                // Complex harmonic generation
                let soft_clip = driven.tanh();
                let harmonic = (driven * 3.0).sin() * 0.1;
                soft_clip + harmonic
            },
            SaturationType::Clean => {
                // Minimal saturation
                if driven.abs() < 0.9 {
                    driven
                } else {
                    driven.signum() * 0.9
                }
            },
        };

        saturated * self.output_gain
    }
}

/// Preset configurations for additive synthesis
pub mod presets {
    use super::*;

    /// Gentle polyharmonic pad for ambient textures
    pub fn gentle_polyharmonic_pad(sample_rate: f32) -> AdditiveSynthesizer {
        AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Gentle)
    }

    /// Rich harmonic bell for resonant tones
    pub fn rich_harmonic_bell(sample_rate: f32) -> AdditiveSynthesizer {
        AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Crystalline)
    }

    /// Complex evolving texture for active ambient
    pub fn complex_evolving_texture(sample_rate: f32) -> AdditiveSynthesizer {
        AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Complex)
    }

    /// Organic harmonic instrument for environmental sounds
    pub fn organic_harmonic_instrument(sample_rate: f32) -> AdditiveSynthesizer {
        AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Organic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_synthesizer_creation() {
        let synth = AdditiveSynthesizer::new(44100.0, 16);
        assert_eq!(synth.harmonics.len(), 16);
        assert_eq!(synth.sample_rate, 44100.0);
    }

    #[test]
    fn test_harmonic_oscillator() {
        let mut osc = HarmonicOscillator::new(2.0, 0.5);
        let sample = osc.process(440.0, 0.5, 44100.0);
        assert!(sample.is_finite());
    }

    #[test]
    fn test_harmonic_envelope() {
        let mut env = HarmonicEnvelope::new(0.1, 0.2, 0.7, 0.3);
        env.trigger();

        // Test attack phase
        for _ in 0..4410 { // 0.1 seconds at 44.1kHz
            env.update(44100.0);
        }
        assert!(env.value > 0.8); // Should be near peak
        assert_eq!(env.phase, EnvelopePhase::Decay);
    }

    #[test]
    fn test_spectral_envelope() {
        let envelope = SpectralEnvelope::new();
        let gain1 = envelope.get_gain_for_harmonic(0, 440.0);
        let gain2 = envelope.get_gain_for_harmonic(4, 2200.0);

        assert!(gain1 > 0.0);
        assert!(gain2 > 0.0);
    }

    #[test]
    fn test_character_presets() {
        let sample_rate = 44100.0;

        let gentle = AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Gentle);
        let complex = AdditiveSynthesizer::for_character(sample_rate, PolyharmonicCharacter::Complex);

        assert!(gentle.harmonics.len() < complex.harmonics.len());
    }

    #[test]
    fn test_synthesis_output() {
        let mut synth = AdditiveSynthesizer::new(44100.0, 8);
        synth.set_frequency(440.0);
        synth.note_on();

        for _ in 0..1000 {
            let sample = synth.process();
            assert!(sample.is_finite());
            assert!(sample.abs() <= 2.0); // Reasonable bounds
        }
    }

    #[test]
    fn test_harmonic_evolution() {
        let mut evolution = HarmonicEvolution::new();
        evolution.speed = 1.0;

        let initial_amp = evolution.get_amplitude_ratio(2);

        // Update evolution
        for _ in 0..1000 {
            evolution.update();
        }

        let evolved_amp = evolution.get_amplitude_ratio(2);

        // Should have evolved
        assert!((initial_amp - evolved_amp).abs() > 0.001);
    }
}