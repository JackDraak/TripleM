//! Advanced drum synthesis for electronic music generation
//!
//! This module provides sophisticated drum synthesis techniques including
//! multi-layered kicks, realistic hi-hats, snare synthesis, and dynamic
//! processing for professional-quality electronic percussion.

use crate::audio::{NaturalVariation, StateVariableFilter, FilterType, AdsrEnvelope};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::{PI, TAU};

/// Complete drum synthesizer for electronic music
#[derive(Debug, Clone)]
pub struct DrumSynthesizer {
    kick_synth: KickSynthesizer,
    snare_synth: SnareSynthesizer,
    hihat_synth: HiHatSynthesizer,
    percussion_synth: PercussionSynthesizer,

    // Global processing
    global_filter: StateVariableFilter,
    variation: NaturalVariation,

    // Mixing
    kick_level: f32,
    snare_level: f32,
    hihat_level: f32,
    percussion_level: f32,

    sample_rate: f32,
}

/// Advanced kick drum synthesizer with multiple layers
#[derive(Debug, Clone)]
pub struct KickSynthesizer {
    // Multiple synthesis layers
    sine_layer: KickSineLayer,
    click_layer: KickClickLayer,
    noise_layer: KickNoiseLayer,
    sub_layer: KickSubLayer,

    // Processing
    envelope: AdsrEnvelope,
    filter: StateVariableFilter,
    pitch_envelope: PitchEnvelope,

    // Parameters
    punch: f32,       // 0.0 = soft, 1.0 = punchy
    tightness: f32,   // 0.0 = loose, 1.0 = tight
    sub_weight: f32,  // Amount of sub-bass

    sample_rate: f32,
}

/// Snare drum synthesizer with noise and tonal components
#[derive(Debug, Clone)]
pub struct SnareSynthesizer {
    // Synthesis layers
    noise_source: NoiseSource,
    tonal_layer: TonalLayer,

    // Processing
    envelope: AdsrEnvelope,
    filter: StateVariableFilter,
    distortion: Distortion,

    // Parameters
    snappiness: f32,  // 0.0 = soft, 1.0 = snappy
    tone: f32,        // 0.0 = noisy, 1.0 = tonal
    bite: f32,        // 0.0 = smooth, 1.0 = harsh

    sample_rate: f32,
}

/// Hi-hat synthesizer with realistic metallic sound
#[derive(Debug, Clone)]
pub struct HiHatSynthesizer {
    // Multiple oscillators for complex timbre
    oscillators: Vec<HiHatOscillator>,

    // Processing
    envelope: AdsrEnvelope,
    filter: StateVariableFilter,

    // Parameters
    metallic: f32,    // 0.0 = soft, 1.0 = metallic
    openness: f32,    // 0.0 = closed, 1.0 = open
    brightness: f32,  // 0.0 = dark, 1.0 = bright

    sample_rate: f32,
}

/// General percussion synthesizer for additional elements
#[derive(Debug, Clone)]
pub struct PercussionSynthesizer {
    // Synthesis engines
    fm_engine: PercussionFM,
    noise_engine: PercussionNoise,

    // Processing
    envelope: AdsrEnvelope,
    filter: StateVariableFilter,

    sample_rate: f32,
}

// Supporting synthesis components

#[derive(Debug, Clone)]
struct KickSineLayer {
    phase: f32,
    frequency: f32,
}

#[derive(Debug, Clone)]
struct KickClickLayer {
    phase: f32,
    frequency: f32,
    click_envelope: f32,
}

#[derive(Debug, Clone)]
struct KickNoiseLayer {
    noise_state: [f32; 7], // Pink noise state
}

#[derive(Debug, Clone)]
struct KickSubLayer {
    phase: f32,
    frequency: f32,
}

#[derive(Debug, Clone)]
struct PitchEnvelope {
    start_frequency: f32,
    end_frequency: f32,
    decay_time: f32,
    current_time: f32,
}

#[derive(Debug, Clone)]
struct NoiseSource {
    pink_state: [f32; 7],
    rng: StdRng,
}

#[derive(Debug, Clone)]
struct TonalLayer {
    oscillators: Vec<TonalOscillator>,
}

#[derive(Debug, Clone)]
struct TonalOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
}

#[derive(Debug, Clone)]
struct Distortion {
    drive: f32,
    tone: f32,
}

#[derive(Debug, Clone)]
struct HiHatOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    detuning: f32,
}

#[derive(Debug, Clone)]
struct PercussionFM {
    carrier_phase: f32,
    modulator_phase: f32,
    carrier_freq: f32,
    modulator_freq: f32,
    modulation_index: f32,
}

#[derive(Debug, Clone)]
struct PercussionNoise {
    noise_state: [f32; 7],
    rng: StdRng,
}

impl DrumSynthesizer {
    /// Create a new drum synthesizer
    pub fn new(sample_rate: f32) -> Self {
        Self {
            kick_synth: KickSynthesizer::new(sample_rate),
            snare_synth: SnareSynthesizer::new(sample_rate),
            hihat_synth: HiHatSynthesizer::new(sample_rate),
            percussion_synth: PercussionSynthesizer::new(sample_rate),

            global_filter: StateVariableFilter::new(sample_rate, 8000.0, 0.7),
            variation: NaturalVariation::energetic(),

            kick_level: 0.8,
            snare_level: 0.6,
            hihat_level: 0.4,
            percussion_level: 0.3,

            sample_rate,
        }
    }

    /// Trigger a kick drum
    pub fn trigger_kick(&mut self, velocity: f32) {
        self.kick_synth.trigger(velocity);
    }

    /// Trigger a snare drum
    pub fn trigger_snare(&mut self, velocity: f32) {
        self.snare_synth.trigger(velocity);
    }

    /// Trigger a hi-hat (closed or open)
    pub fn trigger_hihat(&mut self, velocity: f32, open: bool) {
        self.hihat_synth.trigger(velocity, open);
    }

    /// Trigger percussion element
    pub fn trigger_percussion(&mut self, velocity: f32, element_type: PercussionType) {
        self.percussion_synth.trigger(velocity, element_type);
    }

    /// Process a single sample
    pub fn process(&mut self) -> f32 {
        // Update natural variation
        self.variation.update();

        // Generate drum samples
        let kick_sample = self.kick_synth.process() * self.kick_level;
        let snare_sample = self.snare_synth.process() * self.snare_level;
        let hihat_sample = self.hihat_synth.process() * self.hihat_level;
        let percussion_sample = self.percussion_synth.process() * self.percussion_level;

        // Mix all drums
        let mixed = kick_sample + snare_sample + hihat_sample + percussion_sample;

        // Apply global processing with natural variation
        let filter_drift = self.variation.get_timbre_drift(8000.0, 0.1);
        self.global_filter.set_cutoff(filter_drift);

        let filtered = self.global_filter.process_type(mixed, FilterType::Lowpass);

        // Apply gentle saturation
        self.soft_saturate(filtered)
    }

    /// Soft saturation for warmth
    fn soft_saturate(&self, input: f32) -> f32 {
        let drive = 1.2;
        let x = input * drive;
        let saturated = x / (1.0 + x.abs());
        saturated * 0.8
    }

    /// Set drum levels
    pub fn set_levels(&mut self, kick: f32, snare: f32, hihat: f32, percussion: f32) {
        self.kick_level = kick.clamp(0.0, 1.0);
        self.snare_level = snare.clamp(0.0, 1.0);
        self.hihat_level = hihat.clamp(0.0, 1.0);
        self.percussion_level = percussion.clamp(0.0, 1.0);
    }

    /// Configure for different EDM styles
    pub fn set_style(&mut self, style: EDMStyle) {
        match style {
            EDMStyle::House => {
                self.kick_synth.set_parameters(0.8, 0.7, 0.6); // Punchy, tight, moderate sub
                self.snare_synth.set_parameters(0.6, 0.3, 0.4); // Moderate snap, less tonal
                self.hihat_synth.set_parameters(0.5, 0.2, 0.6); // Moderate metallic, closed
            },
            EDMStyle::Techno => {
                self.kick_synth.set_parameters(0.9, 0.9, 0.8); // Very punchy, very tight, heavy sub
                self.snare_synth.set_parameters(0.8, 0.2, 0.6); // Snappy, noisy, biting
                self.hihat_synth.set_parameters(0.7, 0.3, 0.8); // Metallic, slightly open
            },
            EDMStyle::Dubstep => {
                self.kick_synth.set_parameters(0.7, 0.6, 0.9); // Moderate punch, heavy sub
                self.snare_synth.set_parameters(0.9, 0.4, 0.8); // Very snappy, biting
                self.hihat_synth.set_parameters(0.8, 0.1, 0.9); // Very metallic, closed, bright
            },
            EDMStyle::Ambient => {
                self.kick_synth.set_parameters(0.4, 0.5, 0.3); // Soft, loose, light sub
                self.snare_synth.set_parameters(0.3, 0.6, 0.2); // Soft, tonal, smooth
                self.hihat_synth.set_parameters(0.3, 0.6, 0.4); // Less metallic, open, moderate
            },
        }
    }
}

impl KickSynthesizer {
    fn new(sample_rate: f32) -> Self {
        let mut envelope = AdsrEnvelope::new(sample_rate);
        envelope.set_adsr(0.001, 0.1, 0.0, 0.15); // Fast attack, quick decay

        Self {
            sine_layer: KickSineLayer { phase: 0.0, frequency: 60.0 },
            click_layer: KickClickLayer { phase: 0.0, frequency: 2000.0, click_envelope: 0.0 },
            noise_layer: KickNoiseLayer { noise_state: [0.0; 7] },
            sub_layer: KickSubLayer { phase: 0.0, frequency: 30.0 },

            envelope,
            filter: StateVariableFilter::new(sample_rate, 200.0, 0.8),
            pitch_envelope: PitchEnvelope {
                start_frequency: 80.0,
                end_frequency: 50.0,
                decay_time: 0.05,
                current_time: 0.0,
            },

            punch: 0.7,
            tightness: 0.6,
            sub_weight: 0.5,

            sample_rate,
        }
    }

    fn trigger(&mut self, velocity: f32) {
        self.envelope.note_on();
        self.pitch_envelope.current_time = 0.0;
        self.click_layer.click_envelope = velocity;

        // Reset phases for consistency
        self.sine_layer.phase = 0.0;
        self.click_layer.phase = 0.0;
        self.sub_layer.phase = 0.0;
    }

    fn process(&mut self) -> f32 {
        let envelope_value = self.envelope.generate();
        if envelope_value <= 0.001 {
            return 0.0;
        }

        // Update pitch envelope
        self.pitch_envelope.current_time += 1.0 / self.sample_rate;
        let pitch_progress = (self.pitch_envelope.current_time / self.pitch_envelope.decay_time).min(1.0);
        let current_frequency = self.pitch_envelope.start_frequency *
            (1.0 - pitch_progress) + self.pitch_envelope.end_frequency * pitch_progress;

        self.sine_layer.frequency = current_frequency;

        // Generate layers
        let sine_sample = self.generate_sine_layer();
        let click_sample = self.generate_click_layer();
        let noise_sample = self.generate_noise_layer();
        let sub_sample = self.generate_sub_layer();

        // Mix layers
        let mixed = sine_sample * 0.6 +
                   click_sample * self.punch * 0.3 +
                   noise_sample * 0.2 +
                   sub_sample * self.sub_weight * 0.8;

        // Apply filter and envelope
        let filtered = self.filter.process_type(mixed, FilterType::Lowpass);
        filtered * envelope_value
    }

    fn generate_sine_layer(&mut self) -> f32 {
        let sample = (self.sine_layer.phase * TAU).sin();
        self.sine_layer.phase += self.sine_layer.frequency / self.sample_rate;
        if self.sine_layer.phase >= 1.0 {
            self.sine_layer.phase -= 1.0;
        }
        sample
    }

    fn generate_click_layer(&mut self) -> f32 {
        if self.click_layer.click_envelope <= 0.001 {
            return 0.0;
        }

        let sample = (self.click_layer.phase * TAU).sin() * self.click_layer.click_envelope;
        self.click_layer.phase += self.click_layer.frequency / self.sample_rate;
        if self.click_layer.phase >= 1.0 {
            self.click_layer.phase -= 1.0;
        }

        // Decay click envelope quickly
        self.click_layer.click_envelope *= 0.95;
        sample
    }

    fn generate_noise_layer(&mut self) -> f32 {
        let white_noise = (rand::random::<f32>() - 0.5) * 2.0;
        crate::audio::utils::pink_noise_simple(white_noise, &mut self.noise_layer.noise_state)
    }

    fn generate_sub_layer(&mut self) -> f32 {
        let sample = (self.sub_layer.phase * TAU).sin();
        self.sub_layer.phase += self.sub_layer.frequency / self.sample_rate;
        if self.sub_layer.phase >= 1.0 {
            self.sub_layer.phase -= 1.0;
        }
        sample
    }

    fn set_parameters(&mut self, punch: f32, tightness: f32, sub_weight: f32) {
        self.punch = punch.clamp(0.0, 1.0);
        self.tightness = tightness.clamp(0.0, 1.0);
        self.sub_weight = sub_weight.clamp(0.0, 1.0);

        // Adjust envelope based on tightness
        let decay_time = 0.05 + (1.0 - tightness) * 0.1;
        self.envelope.set_adsr(0.001, decay_time, 0.0, decay_time);
    }
}

impl SnareSynthesizer {
    fn new(sample_rate: f32) -> Self {
        let mut envelope = AdsrEnvelope::new(sample_rate);
        envelope.set_adsr(0.001, 0.05, 0.0, 0.1);

        Self {
            noise_source: NoiseSource::new(),
            tonal_layer: TonalLayer::new(),
            envelope,
            filter: StateVariableFilter::new(sample_rate, 4000.0, 1.2),
            distortion: Distortion { drive: 1.5, tone: 0.5 },
            snappiness: 0.6,
            tone: 0.3,
            bite: 0.4,
            sample_rate,
        }
    }

    fn trigger(&mut self, velocity: f32) {
        self.envelope.note_on();
        self.tonal_layer.trigger(velocity);
    }

    fn process(&mut self) -> f32 {
        let envelope_value = self.envelope.generate();
        if envelope_value <= 0.001 {
            return 0.0;
        }

        let noise_sample = self.noise_source.generate();
        let tonal_sample = self.tonal_layer.generate();

        // Mix noise and tonal components
        let mixed = noise_sample * (1.0 - self.tone) + tonal_sample * self.tone;

        // Apply distortion for bite
        let distorted = self.distortion.process(mixed, self.bite);

        // Apply filter and envelope
        let filtered = self.filter.process_type(distorted, FilterType::Highpass);
        filtered * envelope_value * (1.0 + self.snappiness)
    }

    fn set_parameters(&mut self, snappiness: f32, tone: f32, bite: f32) {
        self.snappiness = snappiness.clamp(0.0, 1.0);
        self.tone = tone.clamp(0.0, 1.0);
        self.bite = bite.clamp(0.0, 1.0);
    }
}

impl HiHatSynthesizer {
    fn new(sample_rate: f32) -> Self {
        let mut envelope = AdsrEnvelope::new(sample_rate);
        envelope.set_adsr(0.001, 0.02, 0.0, 0.05);

        // Create multiple oscillators for complex timbre
        let oscillators = vec![
            HiHatOscillator { phase: 0.0, frequency: 8000.0, amplitude: 0.6, detuning: 0.0 },
            HiHatOscillator { phase: 0.0, frequency: 10000.0, amplitude: 0.4, detuning: 0.02 },
            HiHatOscillator { phase: 0.0, frequency: 12000.0, amplitude: 0.3, detuning: -0.015 },
            HiHatOscillator { phase: 0.0, frequency: 15000.0, amplitude: 0.2, detuning: 0.03 },
        ];

        Self {
            oscillators,
            envelope,
            filter: StateVariableFilter::new(sample_rate, 12000.0, 2.0),
            metallic: 0.6,
            openness: 0.3,
            brightness: 0.7,
            sample_rate,
        }
    }

    fn trigger(&mut self, velocity: f32, open: bool) {
        self.envelope.note_on();

        // Adjust envelope based on openness
        let release_time = if open { 0.2 + self.openness * 0.3 } else { 0.05 };
        self.envelope.set_adsr(0.001, 0.02, 0.0, release_time);

        // Reset oscillator phases
        for osc in &mut self.oscillators {
            osc.phase = rand::random::<f32>();
        }
    }

    fn process(&mut self) -> f32 {
        let envelope_value = self.envelope.generate();
        if envelope_value <= 0.001 {
            return 0.0;
        }

        let mut output = 0.0;

        // Generate from all oscillators
        for osc in &mut self.oscillators {
            let detuned_freq = osc.frequency * (1.0 + osc.detuning * self.metallic);
            let sample = (osc.phase * TAU).sin() * osc.amplitude;

            osc.phase += detuned_freq / self.sample_rate;
            if osc.phase >= 1.0 {
                osc.phase -= 1.0;
            }

            output += sample;
        }

        // Add noise for realism
        let noise = (rand::random::<f32>() - 0.5) * 2.0 * 0.4;
        output += noise;

        // Apply filtering based on brightness
        let filter_freq = 8000.0 + self.brightness * 8000.0;
        self.filter.set_cutoff(filter_freq);
        let filtered = self.filter.process_type(output, FilterType::Highpass);

        filtered * envelope_value * 0.3
    }

    fn set_parameters(&mut self, metallic: f32, openness: f32, brightness: f32) {
        self.metallic = metallic.clamp(0.0, 1.0);
        self.openness = openness.clamp(0.0, 1.0);
        self.brightness = brightness.clamp(0.0, 1.0);
    }
}

impl PercussionSynthesizer {
    fn new(sample_rate: f32) -> Self {
        let mut envelope = AdsrEnvelope::new(sample_rate);
        envelope.set_adsr(0.001, 0.03, 0.0, 0.08);

        Self {
            fm_engine: PercussionFM::new(),
            noise_engine: PercussionNoise::new(),
            envelope,
            filter: StateVariableFilter::new(sample_rate, 6000.0, 1.0),
            sample_rate,
        }
    }

    fn trigger(&mut self, velocity: f32, element_type: PercussionType) {
        self.envelope.note_on();

        match element_type {
            PercussionType::Rim => {
                self.fm_engine.set_frequencies(2000.0, 400.0, 3.0);
            },
            PercussionType::Clap => {
                self.noise_engine.trigger();
            },
            PercussionType::Crash => {
                self.fm_engine.set_frequencies(8000.0, 200.0, 8.0);
                self.envelope.set_adsr(0.001, 0.1, 0.2, 1.0);
            },
            PercussionType::Ride => {
                self.fm_engine.set_frequencies(6000.0, 150.0, 2.0);
                self.envelope.set_adsr(0.001, 0.05, 0.3, 0.8);
            },
        }
    }

    fn process(&mut self) -> f32 {
        let envelope_value = self.envelope.generate();
        if envelope_value <= 0.001 {
            return 0.0;
        }

        let fm_sample = self.fm_engine.generate();
        let noise_sample = self.noise_engine.generate();

        let mixed = fm_sample * 0.7 + noise_sample * 0.3;
        let filtered = self.filter.process_type(mixed, FilterType::Bandpass);

        filtered * envelope_value * 0.4
    }
}

// Supporting implementations
impl NoiseSource {
    fn new() -> Self {
        Self {
            pink_state: [0.0; 7],
            rng: StdRng::from_entropy(),
        }
    }

    fn generate(&mut self) -> f32 {
        let white = (self.rng.gen::<f32>() - 0.5) * 2.0;
        crate::audio::utils::pink_noise_simple(white, &mut self.pink_state)
    }
}

impl TonalLayer {
    fn new() -> Self {
        Self {
            oscillators: vec![
                TonalOscillator { phase: 0.0, frequency: 200.0, amplitude: 0.8 },
                TonalOscillator { phase: 0.0, frequency: 300.0, amplitude: 0.4 },
                TonalOscillator { phase: 0.0, frequency: 500.0, amplitude: 0.2 },
            ],
        }
    }

    fn trigger(&mut self, velocity: f32) {
        for osc in &mut self.oscillators {
            osc.phase = rand::random::<f32>();
            osc.amplitude *= velocity;
        }
    }

    fn generate(&mut self) -> f32 {
        let mut output = 0.0;
        for osc in &mut self.oscillators {
            output += (osc.phase * TAU).sin() * osc.amplitude;
            osc.phase += osc.frequency / 44100.0; // Assuming sample rate
            if osc.phase >= 1.0 {
                osc.phase -= 1.0;
            }
        }
        output * 0.3
    }
}

impl Distortion {
    fn process(&self, input: f32, amount: f32) -> f32 {
        let driven = input * (1.0 + self.drive * amount);
        driven.tanh() * 0.7
    }
}

impl PercussionFM {
    fn new() -> Self {
        Self {
            carrier_phase: 0.0,
            modulator_phase: 0.0,
            carrier_freq: 1000.0,
            modulator_freq: 200.0,
            modulation_index: 2.0,
        }
    }

    fn set_frequencies(&mut self, carrier: f32, modulator: f32, mod_index: f32) {
        self.carrier_freq = carrier;
        self.modulator_freq = modulator;
        self.modulation_index = mod_index;
    }

    fn generate(&mut self) -> f32 {
        let modulator = (self.modulator_phase * TAU).sin();
        let modulated_freq = self.carrier_freq + modulator * self.modulation_index * self.carrier_freq;

        let carrier = (self.carrier_phase * TAU).sin();

        self.carrier_phase += modulated_freq / 44100.0;
        self.modulator_phase += self.modulator_freq / 44100.0;

        if self.carrier_phase >= 1.0 { self.carrier_phase -= 1.0; }
        if self.modulator_phase >= 1.0 { self.modulator_phase -= 1.0; }

        carrier * 0.5
    }
}

impl PercussionNoise {
    fn new() -> Self {
        Self {
            noise_state: [0.0; 7],
            rng: StdRng::from_entropy(),
        }
    }

    fn trigger(&mut self) {
        // Reset noise state for consistent trigger
        self.noise_state = [0.0; 7];
    }

    fn generate(&mut self) -> f32 {
        let white = (self.rng.gen::<f32>() - 0.5) * 2.0;
        crate::audio::utils::pink_noise_simple(white, &mut self.noise_state)
    }
}

/// EDM style presets
#[derive(Debug, Clone, Copy)]
pub enum EDMStyle {
    House,
    Techno,
    Dubstep,
    Ambient,
}

/// Percussion element types
#[derive(Debug, Clone, Copy)]
pub enum PercussionType {
    Rim,
    Clap,
    Crash,
    Ride,
}

/// Preset configurations for different EDM styles
pub mod presets {
    use super::*;

    /// Create a drum synthesizer optimized for house music
    pub fn house_drums(sample_rate: f32) -> DrumSynthesizer {
        let mut drums = DrumSynthesizer::new(sample_rate);
        drums.set_style(EDMStyle::House);
        drums.set_levels(0.9, 0.7, 0.5, 0.3);
        drums
    }

    /// Create a drum synthesizer optimized for techno
    pub fn techno_drums(sample_rate: f32) -> DrumSynthesizer {
        let mut drums = DrumSynthesizer::new(sample_rate);
        drums.set_style(EDMStyle::Techno);
        drums.set_levels(0.95, 0.8, 0.6, 0.4);
        drums
    }

    /// Create a drum synthesizer optimized for dubstep
    pub fn dubstep_drums(sample_rate: f32) -> DrumSynthesizer {
        let mut drums = DrumSynthesizer::new(sample_rate);
        drums.set_style(EDMStyle::Dubstep);
        drums.set_levels(0.8, 0.9, 0.4, 0.5);
        drums
    }

    /// Create a drum synthesizer optimized for ambient electronic
    pub fn ambient_drums(sample_rate: f32) -> DrumSynthesizer {
        let mut drums = DrumSynthesizer::new(sample_rate);
        drums.set_style(EDMStyle::Ambient);
        drums.set_levels(0.6, 0.5, 0.7, 0.6);
        drums
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drum_synthesizer_creation() {
        let drums = DrumSynthesizer::new(44100.0);
        assert_eq!(drums.sample_rate, 44100.0);
    }

    #[test]
    fn test_kick_trigger() {
        let mut kick = KickSynthesizer::new(44100.0);
        kick.trigger(0.8);

        // Should produce non-zero output after trigger
        let sample = kick.process();
        assert!(sample.abs() > 0.0);
    }

    #[test]
    fn test_drum_processing() {
        let mut drums = DrumSynthesizer::new(44100.0);
        drums.trigger_kick(0.8);
        drums.trigger_snare(0.7);
        drums.trigger_hihat(0.5, false);

        // Process several samples
        for _ in 0..1000 {
            let sample = drums.process();
            assert!(sample.is_finite());
        }
    }

    #[test]
    fn test_style_configuration() {
        let mut drums = DrumSynthesizer::new(44100.0);

        drums.set_style(EDMStyle::House);
        drums.set_style(EDMStyle::Techno);
        drums.set_style(EDMStyle::Dubstep);
        drums.set_style(EDMStyle::Ambient);

        // Should not panic and should accept all styles
    }

    #[test]
    fn test_preset_creation() {
        let sample_rate = 44100.0;

        let _house = presets::house_drums(sample_rate);
        let _techno = presets::techno_drums(sample_rate);
        let _dubstep = presets::dubstep_drums(sample_rate);
        let _ambient = presets::ambient_drums(sample_rate);

        // Just test that they can be created without panicking
    }

    #[test]
    fn test_percussion_types() {
        let mut drums = DrumSynthesizer::new(44100.0);

        drums.trigger_percussion(0.7, PercussionType::Rim);
        drums.trigger_percussion(0.8, PercussionType::Clap);
        drums.trigger_percussion(0.6, PercussionType::Crash);
        drums.trigger_percussion(0.5, PercussionType::Ride);

        // Process samples to ensure no crashes
        for _ in 0..100 {
            let sample = drums.process();
            assert!(sample.is_finite());
        }
    }
}