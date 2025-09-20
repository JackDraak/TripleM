//! Advanced wavetable synthesis with morphing and spectral processing
//!
//! This module provides sophisticated wavetable synthesis with organic shapes,
//! real-time morphing, formant processing, and spectral manipulation for
//! creating deeply expressive and moody timbres.

use crate::audio::{NaturalVariation, StateVariableFilter, FilterType};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::{PI, TAU};

/// Advanced wavetable synthesizer with morphing capabilities
#[derive(Debug, Clone)]
pub struct MorphingWavetableSynth {
    // Wavetable system
    wavetable_bank: WavetableBank,
    current_table_a: usize,
    current_table_b: usize,
    morph_position: f32,  // 0.0 = table_a, 1.0 = table_b

    // Oscillator state
    phase: f32,
    frequency: f32,
    amplitude: f32,

    // Morphing control
    morph_lfo: MorphLFO,
    variation: NaturalVariation,

    // Spectral processing
    formant_processor: FormantProcessor,
    harmonic_enhancer: HarmonicEnhancer,

    // Output processing
    output_filter: StateVariableFilter,

    sample_rate: f32,
}

/// Bank of organic wavetables for different moods and timbres
#[derive(Debug, Clone)]
pub struct WavetableBank {
    tables: Vec<Wavetable>,
}

/// Individual wavetable with metadata
#[derive(Debug, Clone)]
pub struct Wavetable {
    samples: Vec<f32>,
    name: String,
    character: WaveCharacter,
    harmonic_content: f32,    // 0.0 = simple, 1.0 = complex
    brightness: f32,          // 0.0 = dark, 1.0 = bright
    organic_factor: f32,      // 0.0 = geometric, 1.0 = organic
}

/// Character classification for wavetables
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveCharacter {
    Organic,      // Natural, flowing shapes
    Harmonic,     // Rich harmonic content
    Percussive,   // Sharp, transient-like
    Ethereal,     // Soft, atmospheric
    Growling,     // Aggressive, distorted
    Crystalline,  // Clear, bell-like
    Vocal,        // Formant-rich, speech-like
    Synthetic,    // Obviously artificial but interesting
}

/// LFO for controlling wavetable morphing
#[derive(Debug, Clone)]
pub struct MorphLFO {
    phase: f32,
    frequency: f32,
    depth: f32,
    waveform: LFOWaveform,
    sample_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum LFOWaveform {
    Sine,
    Triangle,
    Sawtooth,
    Square,
    Random,
    Perlin,  // Smooth noise
}

/// Formant processor for vocal-like characteristics
#[derive(Debug, Clone)]
pub struct FormantProcessor {
    formants: Vec<FormantFilter>,
    intensity: f32,
}

#[derive(Debug, Clone)]
pub struct FormantFilter {
    center_frequency: f32,
    bandwidth: f32,
    gain: f32,
    filter: StateVariableFilter,
}

/// Harmonic enhancer for rich polyharmonic content
#[derive(Debug, Clone)]
pub struct HarmonicEnhancer {
    harmonics: Vec<HarmonicComponent>,
    enhancement_amount: f32,
    spectral_tilt: f32,  // -1.0 = dark, 1.0 = bright
}

#[derive(Debug, Clone)]
pub struct HarmonicComponent {
    harmonic_number: usize,
    amplitude: f32,
    phase_offset: f32,
    frequency_drift: f32,  // Slight detuning for organic feel
}

impl MorphingWavetableSynth {
    /// Create a new morphing wavetable synthesizer
    pub fn new(sample_rate: f32, character: WaveCharacter) -> Self {
        let wavetable_bank = WavetableBank::new();
        let variation = NaturalVariation::new(None);

        // Choose initial wavetables based on character
        let (table_a, table_b) = wavetable_bank.get_character_pair(character);

        Self {
            wavetable_bank,
            current_table_a: table_a,
            current_table_b: table_b,
            morph_position: 0.0,

            phase: 0.0,
            frequency: 440.0,
            amplitude: 1.0,

            morph_lfo: MorphLFO::new(sample_rate, 0.1, LFOWaveform::Sine),
            variation,

            formant_processor: FormantProcessor::new(sample_rate, character),
            harmonic_enhancer: HarmonicEnhancer::new(character),

            output_filter: StateVariableFilter::new(sample_rate, 8000.0, 0.7),

            sample_rate,
        }
    }

    /// Generate the next sample
    pub fn process(&mut self) -> f32 {
        // Update natural variation
        self.variation.update();

        // Update morphing
        self.update_morphing();

        // Generate base waveform through wavetable lookup
        let base_sample = self.generate_wavetable_sample();

        // Apply harmonic enhancement
        let enhanced_sample = self.harmonic_enhancer.process(base_sample, self.frequency);

        // Apply formant processing for vocal characteristics
        let formant_sample = self.formant_processor.process(enhanced_sample);

        // Apply natural variation to filter
        let filter_drift = self.variation.get_timbre_drift(8000.0, 0.3);
        self.output_filter.set_cutoff(filter_drift);
        let filtered_sample = self.output_filter.process_type(formant_sample, FilterType::Lowpass);

        // Update oscillator phase
        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        filtered_sample * self.amplitude
    }

    /// Update wavetable morphing
    fn update_morphing(&mut self) {
        // Update morph LFO
        let lfo_value = self.morph_lfo.process();

        // Apply natural variation to morphing
        let morph_variation = self.variation.get_timbre_variation();

        // Combine LFO and variation for organic morphing
        self.morph_position = (lfo_value + morph_variation * 0.3).clamp(0.0, 1.0);

        // Occasionally switch wavetables for longer-term evolution
        if self.variation.get_timing_variation() > 0.95 {
            self.evolve_wavetable_selection();
        }
    }

    /// Generate sample through wavetable lookup with morphing
    fn generate_wavetable_sample(&self) -> f32 {
        let table_size = self.wavetable_bank.table_size();
        let lookup_index = self.phase * table_size as f32;
        let index_floor = lookup_index.floor() as usize;
        let index_frac = lookup_index.fract();

        // Get samples from both wavetables
        let sample_a = self.wavetable_bank.get_interpolated_sample(
            self.current_table_a,
            index_floor,
            index_frac
        );
        let sample_b = self.wavetable_bank.get_interpolated_sample(
            self.current_table_b,
            index_floor,
            index_frac
        );

        // Morph between the two samples
        sample_a * (1.0 - self.morph_position) + sample_b * self.morph_position
    }

    /// Evolve wavetable selection for long-term variation
    fn evolve_wavetable_selection(&mut self) {
        // Choose new wavetables based on current character and some randomness
        let current_character = self.wavetable_bank.get_character(self.current_table_a);
        let (new_a, new_b) = self.wavetable_bank.get_evolution_pair(current_character);

        self.current_table_a = new_a;
        self.current_table_b = new_b;
    }

    /// Set synthesizer parameters
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.clamp(20.0, 20000.0);
    }

    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    pub fn set_morph_rate(&mut self, rate: f32) {
        self.morph_lfo.set_frequency(rate.clamp(0.01, 10.0));
    }

    pub fn set_formant_intensity(&mut self, intensity: f32) {
        self.formant_processor.set_intensity(intensity.clamp(0.0, 1.0));
    }

    pub fn set_harmonic_enhancement(&mut self, amount: f32) {
        self.harmonic_enhancer.set_enhancement(amount.clamp(0.0, 1.0));
    }

    /// Create synthesizer optimized for specific mood
    pub fn for_mood(sample_rate: f32, mood: f32) -> Self {
        let character = match mood {
            m if m < 0.25 => WaveCharacter::Organic,     // Environmental
            m if m < 0.5 => WaveCharacter::Ethereal,     // Gentle melodic
            m if m < 0.75 => WaveCharacter::Harmonic,    // Active ambient
            _ => WaveCharacter::Synthetic,               // EDM
        };

        Self::new(sample_rate, character)
    }
}

impl WavetableBank {
    /// Create a bank of organic wavetables
    pub fn new() -> Self {
        let mut tables = Vec::new();

        // Organic wavetables based on natural phenomena
        tables.push(Self::create_ocean_wave());
        tables.push(Self::create_mountain_profile());
        tables.push(Self::create_wind_pattern());
        tables.push(Self::create_crystal_structure());
        tables.push(Self::create_vocal_formant());

        // Harmonic wavetables
        tables.push(Self::create_rich_harmonic());
        tables.push(Self::create_bell_spectrum());
        tables.push(Self::create_string_harmonics());

        // Ethereal wavetables
        tables.push(Self::create_ethereal_pad());
        tables.push(Self::create_floating_atmosphere());

        // Synthetic wavetables
        tables.push(Self::create_digital_pulse());
        tables.push(Self::create_modulated_complex());

        Self { tables }
    }

    /// Create ocean wave-inspired wavetable
    fn create_ocean_wave() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Base wave with multiple harmonics for complexity
            let mut sample = 0.0;
            sample += (phase * TAU).sin() * 0.6;                    // Fundamental
            sample += (phase * TAU * 2.0).sin() * 0.2;             // 2nd harmonic
            sample += (phase * TAU * 3.0).sin() * 0.1;             // 3rd harmonic

            // Add organic irregularities
            let organic_noise = ((phase * TAU * 17.3).sin() * 0.05) +
                               ((phase * TAU * 23.7).sin() * 0.03);
            sample += organic_noise;

            // Apply gentle wave shaping for ocean-like character
            sample = sample.tanh() * 0.8;

            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Ocean Wave".to_string(),
            character: WaveCharacter::Organic,
            harmonic_content: 0.4,
            brightness: 0.3,
            organic_factor: 0.9,
        }
    }

    /// Create mountain profile-inspired wavetable
    fn create_mountain_profile() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Create jagged mountain-like profile
            let mut sample = 0.0;

            // Multiple peaks at different scales
            sample += (phase * TAU * 1.0).sin() * 0.5;             // Main ridge
            sample += (phase * TAU * 3.7).sin() * 0.3;             // Medium peaks
            sample += (phase * TAU * 8.3).sin() * 0.2;             // Small peaks
            sample += (phase * TAU * 15.1).sin() * 0.1;            // Rocky details

            // Add some randomness for natural irregularity
            let random_factor = ((phase * TAU * 31.4).sin() *
                               (phase * TAU * 47.2).cos()) * 0.05;
            sample += random_factor;

            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Mountain Profile".to_string(),
            character: WaveCharacter::Organic,
            harmonic_content: 0.7,
            brightness: 0.6,
            organic_factor: 0.8,
        }
    }

    /// Create wind pattern-inspired wavetable
    fn create_wind_pattern() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Smooth, flowing wind-like pattern
            let mut sample = 0.0;

            // Base flow with harmonics
            sample += (phase * TAU).sin() * 0.7;
            sample += (phase * TAU * 1.618).sin() * 0.3;           // Golden ratio for organic feel
            sample += (phase * TAU * 2.414).sin() * 0.2;           // √2 ratio

            // Add turbulence
            let turbulence = ((phase * TAU * 7.1).sin() *
                            (phase * TAU * 11.3).sin()) * 0.15;
            sample += turbulence;

            // Smooth the overall shape
            sample = sample * (1.0 + (phase * TAU * 0.5).cos() * 0.1);

            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Wind Pattern".to_string(),
            character: WaveCharacter::Ethereal,
            harmonic_content: 0.5,
            brightness: 0.4,
            organic_factor: 0.9,
        }
    }

    /// Create crystal structure-inspired wavetable
    fn create_crystal_structure() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Crystalline structure with mathematical ratios
            let mut sample = 0.0;

            // Perfect harmonic ratios for crystal-like clarity
            sample += (phase * TAU * 1.0).sin() * 0.8;             // Fundamental
            sample += (phase * TAU * 2.0).sin() * 0.4;             // Octave
            sample += (phase * TAU * 3.0).sin() * 0.3;             // Perfect fifth
            sample += (phase * TAU * 4.0).sin() * 0.2;             // Second octave
            sample += (phase * TAU * 5.0).sin() * 0.15;            // Major third
            sample += (phase * TAU * 6.0).sin() * 0.1;             // Perfect fifth + octave

            // Add some crystalline sparkle
            let sparkle = ((phase * TAU * 12.0).sin() *
                          (phase * TAU * 19.0).sin()) * 0.05;
            sample += sparkle;

            samples.push(sample * 0.7);
        }

        Wavetable {
            samples,
            name: "Crystal Structure".to_string(),
            character: WaveCharacter::Crystalline,
            harmonic_content: 0.9,
            brightness: 0.8,
            organic_factor: 0.2,
        }
    }

    /// Create vocal formant-inspired wavetable
    fn create_vocal_formant() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Vocal-like formant structure (approximating "ah" vowel)
            let mut sample = 0.0;

            // Fundamental
            sample += (phase * TAU).sin() * 0.6;

            // Formant peaks at specific frequencies
            sample += (phase * TAU * 2.5).sin() * 0.3;             // First formant
            sample += (phase * TAU * 6.2).sin() * 0.2;             // Second formant
            sample += (phase * TAU * 11.8).sin() * 0.1;            // Third formant

            // Add some vocal breathiness
            let breathiness = ((phase * TAU * 29.7).sin() *
                             (phase * TAU * 41.3).sin()) * 0.08;
            sample += breathiness;

            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Vocal Formant".to_string(),
            character: WaveCharacter::Vocal,
            harmonic_content: 0.6,
            brightness: 0.5,
            organic_factor: 0.8,
        }
    }

    /// Create rich harmonic wavetable
    fn create_rich_harmonic() -> Wavetable {
        let size = 2048;
        let mut samples = Vec::with_capacity(size);

        for i in 0..size {
            let phase = i as f32 / size as f32;

            // Rich harmonic series
            let mut sample = 0.0;

            for harmonic in 1..=16 {
                let amplitude = 1.0 / harmonic as f32;  // Natural harmonic decay
                sample += (phase * TAU * harmonic as f32).sin() * amplitude * 0.3;
            }

            samples.push(sample);
        }

        Wavetable {
            samples,
            name: "Rich Harmonic".to_string(),
            character: WaveCharacter::Harmonic,
            harmonic_content: 1.0,
            brightness: 0.7,
            organic_factor: 0.3,
        }
    }

    /// Additional wavetable creation methods...
    fn create_bell_spectrum() -> Wavetable {
        // Implementation for bell-like spectrum
        let size = 2048;
        let mut samples = vec![0.0; size];

        // Bell-like inharmonic partials
        let partials = [1.0, 2.76, 5.4, 8.93, 13.34, 18.64];
        for (i, &partial) in partials.iter().enumerate() {
            let amplitude = 1.0 / (i + 1) as f32;
            for j in 0..size {
                let phase = j as f32 / size as f32;
                samples[j] += (phase * TAU * partial).sin() * amplitude * 0.2;
            }
        }

        Wavetable {
            samples,
            name: "Bell Spectrum".to_string(),
            character: WaveCharacter::Crystalline,
            harmonic_content: 0.8,
            brightness: 0.9,
            organic_factor: 0.1,
        }
    }

    // Additional creation methods would follow similar patterns...
    fn create_string_harmonics() -> Wavetable { Self::create_rich_harmonic() }
    fn create_ethereal_pad() -> Wavetable { Self::create_wind_pattern() }
    fn create_floating_atmosphere() -> Wavetable { Self::create_vocal_formant() }
    fn create_digital_pulse() -> Wavetable { Self::create_crystal_structure() }
    fn create_modulated_complex() -> Wavetable { Self::create_mountain_profile() }

    /// Get wavetable pair for specific character
    pub fn get_character_pair(&self, character: WaveCharacter) -> (usize, usize) {
        let character_tables: Vec<usize> = self.tables
            .iter()
            .enumerate()
            .filter(|(_, table)| table.character == character)
            .map(|(i, _)| i)
            .collect();

        if character_tables.len() >= 2 {
            (character_tables[0], character_tables[1])
        } else {
            (0, 1.min(self.tables.len() - 1))
        }
    }

    /// Get evolution pair (similar character with some variation)
    pub fn get_evolution_pair(&self, character: WaveCharacter) -> (usize, usize) {
        // Implementation would choose tables that are similar but not identical
        self.get_character_pair(character)
    }

    pub fn get_character(&self, index: usize) -> WaveCharacter {
        self.tables.get(index).map(|t| t.character).unwrap_or(WaveCharacter::Organic)
    }

    pub fn table_size(&self) -> usize {
        self.tables.first().map(|t| t.samples.len()).unwrap_or(2048)
    }

    pub fn get_interpolated_sample(&self, table_index: usize, index: usize, frac: f32) -> f32 {
        if let Some(table) = self.tables.get(table_index) {
            let sample1 = table.samples[index % table.samples.len()];
            let sample2 = table.samples[(index + 1) % table.samples.len()];
            sample1 + (sample2 - sample1) * frac
        } else {
            0.0
        }
    }
}

impl MorphLFO {
    fn new(sample_rate: f32, frequency: f32, waveform: LFOWaveform) -> Self {
        Self {
            phase: 0.0,
            frequency,
            depth: 1.0,
            waveform,
            sample_rate,
        }
    }

    fn process(&mut self) -> f32 {
        let output = match self.waveform {
            LFOWaveform::Sine => (self.phase * TAU).sin(),
            LFOWaveform::Triangle => {
                if self.phase < 0.5 {
                    self.phase * 4.0 - 1.0
                } else {
                    3.0 - self.phase * 4.0
                }
            },
            LFOWaveform::Sawtooth => self.phase * 2.0 - 1.0,
            LFOWaveform::Square => if self.phase < 0.5 { -1.0 } else { 1.0 },
            LFOWaveform::Random => (rand::random::<f32>() - 0.5) * 2.0,
            LFOWaveform::Perlin => {
                // Simplified Perlin-like noise
                let scaled_phase = self.phase * 8.0;
                ((scaled_phase.sin() + (scaled_phase * 2.0).sin() * 0.5 +
                  (scaled_phase * 4.0).sin() * 0.25) / 1.75).clamp(-1.0, 1.0)
            },
        };

        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        output * self.depth
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.clamp(0.001, 20.0);
    }
}

impl FormantProcessor {
    fn new(sample_rate: f32, character: WaveCharacter) -> Self {
        let formants = match character {
            WaveCharacter::Vocal => vec![
                FormantFilter::new(sample_rate, 730.0, 100.0, 1.0),   // A vowel formants
                FormantFilter::new(sample_rate, 1090.0, 80.0, 0.8),
                FormantFilter::new(sample_rate, 2440.0, 120.0, 0.6),
            ],
            WaveCharacter::Ethereal => vec![
                FormantFilter::new(sample_rate, 400.0, 200.0, 0.6),   // Softer formants
                FormantFilter::new(sample_rate, 1200.0, 300.0, 0.4),
            ],
            _ => vec![
                FormantFilter::new(sample_rate, 800.0, 150.0, 0.7),   // Generic formant
            ],
        };

        Self {
            formants,
            intensity: 0.5,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let mut output = input;

        for formant in &mut self.formants {
            output = formant.process(output);
        }

        // Blend between original and formant-processed signal
        input * (1.0 - self.intensity) + output * self.intensity
    }

    fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }
}

impl FormantFilter {
    fn new(sample_rate: f32, center_freq: f32, bandwidth: f32, gain: f32) -> Self {
        let q = center_freq / bandwidth;
        let filter = StateVariableFilter::new(sample_rate, center_freq, q);

        Self {
            center_frequency: center_freq,
            bandwidth,
            gain,
            filter,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let filtered = self.filter.process_type(input, FilterType::Bandpass);
        filtered * self.gain
    }
}

impl HarmonicEnhancer {
    fn new(character: WaveCharacter) -> Self {
        let harmonics = match character {
            WaveCharacter::Harmonic => {
                // Rich harmonic series
                (1..=8).map(|n| HarmonicComponent {
                    harmonic_number: n,
                    amplitude: 1.0 / n as f32,
                    phase_offset: 0.0,
                    frequency_drift: (n as f32 * 0.001).min(0.01),
                }).collect()
            },
            WaveCharacter::Crystalline => {
                // Perfect harmonic ratios
                vec![
                    HarmonicComponent { harmonic_number: 2, amplitude: 0.4, phase_offset: 0.0, frequency_drift: 0.0 },
                    HarmonicComponent { harmonic_number: 3, amplitude: 0.3, phase_offset: 0.0, frequency_drift: 0.0 },
                    HarmonicComponent { harmonic_number: 4, amplitude: 0.2, phase_offset: 0.0, frequency_drift: 0.0 },
                ]
            },
            _ => {
                // Subtle harmonics
                vec![
                    HarmonicComponent { harmonic_number: 2, amplitude: 0.2, phase_offset: 0.0, frequency_drift: 0.005 },
                    HarmonicComponent { harmonic_number: 3, amplitude: 0.1, phase_offset: 0.0, frequency_drift: 0.003 },
                ]
            },
        };

        Self {
            harmonics,
            enhancement_amount: 0.5,
            spectral_tilt: 0.0,
        }
    }

    fn process(&self, input: f32, fundamental_freq: f32) -> f32 {
        let mut output = input;

        // Add harmonics
        for harmonic in &self.harmonics {
            let harmonic_freq = fundamental_freq * harmonic.harmonic_number as f32;
            let detuned_freq = harmonic_freq * (1.0 + harmonic.frequency_drift);

            // Generate harmonic (simplified - in real implementation would need phase tracking)
            let harmonic_sample = (detuned_freq * TAU / 44100.0).sin() * harmonic.amplitude;
            output += harmonic_sample * self.enhancement_amount;
        }

        // Apply spectral tilt
        if self.spectral_tilt != 0.0 {
            // Simplified spectral tilt - would use proper filtering in full implementation
            output *= 1.0 + self.spectral_tilt * 0.1;
        }

        output
    }

    fn set_enhancement(&mut self, amount: f32) {
        self.enhancement_amount = amount.clamp(0.0, 1.0);
    }
}

/// Preset configurations for different musical contexts
pub mod presets {
    use super::*;

    /// Warm, organic wavetable synth for gentle ambient music
    pub fn gentle_organic(sample_rate: f32) -> MorphingWavetableSynth {
        let mut synth = MorphingWavetableSynth::new(sample_rate, WaveCharacter::Organic);
        synth.set_morph_rate(0.05);  // Very slow morphing
        synth.set_formant_intensity(0.3);  // Subtle formants
        synth.set_harmonic_enhancement(0.4);  // Moderate harmonics
        synth
    }

    /// Rich harmonic synth for complex pads
    pub fn harmonic_pad(sample_rate: f32) -> MorphingWavetableSynth {
        let mut synth = MorphingWavetableSynth::new(sample_rate, WaveCharacter::Harmonic);
        synth.set_morph_rate(0.1);  // Slow morphing
        synth.set_formant_intensity(0.2);  // Light formants
        synth.set_harmonic_enhancement(0.8);  // Rich harmonics
        synth
    }

    /// Ethereal synth for atmospheric textures
    pub fn ethereal_atmosphere(sample_rate: f32) -> MorphingWavetableSynth {
        let mut synth = MorphingWavetableSynth::new(sample_rate, WaveCharacter::Ethereal);
        synth.set_morph_rate(0.03);  // Very slow morphing
        synth.set_formant_intensity(0.6);  // Strong formants
        synth.set_harmonic_enhancement(0.3);  // Subtle harmonics
        synth
    }

    /// Crystalline synth for bell-like tones
    pub fn crystalline_bells(sample_rate: f32) -> MorphingWavetableSynth {
        let mut synth = MorphingWavetableSynth::new(sample_rate, WaveCharacter::Crystalline);
        synth.set_morph_rate(0.2);  // Moderate morphing
        synth.set_formant_intensity(0.1);  // Minimal formants
        synth.set_harmonic_enhancement(0.9);  // Maximum harmonics
        synth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavetable_bank_creation() {
        let bank = WavetableBank::new();
        assert!(bank.tables.len() > 0);
        assert_eq!(bank.table_size(), 2048);
    }

    #[test]
    fn test_morphing_synth_creation() {
        let synth = MorphingWavetableSynth::new(44100.0, WaveCharacter::Organic);
        assert_eq!(synth.sample_rate, 44100.0);
    }

    #[test]
    fn test_wavetable_sample_generation() {
        let mut synth = MorphingWavetableSynth::new(44100.0, WaveCharacter::Harmonic);
        synth.set_frequency(440.0);

        for _ in 0..1000 {
            let sample = synth.process();
            assert!(sample.is_finite());
            assert!(sample.abs() <= 2.0);  // Reasonable bounds
        }
    }

    #[test]
    fn test_formant_processor() {
        let mut processor = FormantProcessor::new(44100.0, WaveCharacter::Vocal);
        processor.set_intensity(0.5);

        let input = 0.5;
        let output = processor.process(input);
        assert!(output.is_finite());
    }

    #[test]
    fn test_harmonic_enhancer() {
        let enhancer = HarmonicEnhancer::new(WaveCharacter::Harmonic);
        let output = enhancer.process(0.5, 440.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_preset_creation() {
        let sample_rate = 44100.0;

        let _gentle = presets::gentle_organic(sample_rate);
        let _harmonic = presets::harmonic_pad(sample_rate);
        let _ethereal = presets::ethereal_atmosphere(sample_rate);
        let _crystalline = presets::crystalline_bells(sample_rate);

        // Just test that they can be created without panicking
    }
}