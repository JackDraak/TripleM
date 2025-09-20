//! Unified synthesizer interface integrating wavetable and additive synthesis
//!
//! This module provides a unified synthesizer that seamlessly integrates
//! wavetable and additive synthesis engines with the unified pattern generators,
//! enabling continuous parameter morphing across the entire input range (0.0-1.0).

use crate::patterns::{
    UnifiedRhythmGenerator, UnifiedMelodyGenerator, UnifiedHarmonyGenerator,
    RhythmPattern, MelodyPattern, HarmonyPattern,
};
use crate::audio::{
    MorphingWavetableSynth, AdditiveSynthesizer, WaveCharacter,
    StereoFrame, NaturalVariation, CrossfadeManager, CrossfadeParameter, CrossfadePriority,
};
use crate::error::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Unified synthesizer integrating all pattern generators and synthesis engines
#[derive(Debug, Clone)]
pub struct UnifiedSynthesizer {
    /// Pattern generators
    rhythm_generator: UnifiedRhythmGenerator,
    melody_generator: UnifiedMelodyGenerator,
    harmony_generator: UnifiedHarmonyGenerator,

    /// Synthesis engines
    wavetable_synth: MorphingWavetableSynth,
    additive_synth: AdditiveSynthesizer,

    /// Synthesis balance and morphing
    synthesis_morph: SynthesisMorpher,

    /// Voice management
    voice_manager: VoiceManager,

    /// Output processing
    output_processor: UnifiedOutputProcessor,

    /// Crossfade manager for seamless parameter transitions
    crossfade_manager: CrossfadeManager,

    /// Current input value controlling everything
    input_value: f32,

    /// Natural variation for organic feel
    variation: NaturalVariation,

    /// Sample counter for phase calculation
    sample_counter: usize,

    /// Sample rate
    sample_rate: f32,

    /// Random number generator
    rng: StdRng,
}

/// Manages morphing between wavetable and additive synthesis
#[derive(Debug, Clone)]
pub struct SynthesisMorpher {
    /// Current morph position (0.0 = pure wavetable, 1.0 = pure additive)
    morph_position: f32,

    /// Morph curve for different input ranges
    morph_curve: MorphCurve,

    /// Cross-fade parameters
    crossfade_smoothing: f32,
}

/// Defines how synthesis morphing responds to input value
#[derive(Debug, Clone)]
pub struct MorphCurve {
    /// Control points defining morph behavior across input range
    control_points: Vec<MorphPoint>,

    /// Interpolation type between points
    interpolation: MorphInterpolation,
}

#[derive(Debug, Clone)]
pub struct MorphPoint {
    /// Input value (0.0-1.0)
    input: f32,

    /// Wavetable weight at this point
    wavetable_weight: f32,

    /// Additive weight at this point
    additive_weight: f32,

    /// Wavetable character preference
    wavetable_character: WaveCharacter,

    /// Additive harmonic complexity
    additive_complexity: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum MorphInterpolation {
    Linear,
    Smooth,
    Exponential,
}

/// Manages voice allocation for polyphonic synthesis
#[derive(Debug, Clone)]
pub struct VoiceManager {
    /// Active synthesis voices
    voices: Vec<SynthVoice>,

    /// Maximum polyphony
    max_voices: usize,

    /// Voice allocation strategy
    allocation_strategy: VoiceAllocation,

    /// Voice stealing parameters
    stealing_params: VoiceStealingParams,
}

/// Individual synthesis voice
#[derive(Debug, Clone)]
pub struct SynthVoice {
    /// Unique voice ID
    id: usize,

    /// MIDI note number
    note: u8,

    /// Current frequency
    frequency: f32,

    /// Voice amplitude
    amplitude: f32,

    /// Voice age (for stealing decisions)
    age: f32,

    /// Synthesis parameters for this voice
    synth_params: VoiceSynthParams,

    /// Voice state
    state: VoiceState,
}

#[derive(Debug, Clone)]
pub struct VoiceSynthParams {
    /// Wavetable parameters
    wavetable_params: WavetableVoiceParams,

    /// Additive parameters
    additive_params: AdditiveVoiceParams,

    /// Morph position for this voice
    morph_position: f32,
}

#[derive(Debug, Clone)]
pub struct WavetableVoiceParams {
    /// Current wavetable character
    character: WaveCharacter,

    /// Morph position within wavetable bank
    table_morph: f32,

    /// Filter cutoff
    filter_cutoff: f32,

    /// Filter resonance
    filter_resonance: f32,
}

#[derive(Debug, Clone)]
pub struct AdditiveVoiceParams {
    /// Number of active harmonics
    harmonic_count: usize,

    /// Harmonic distribution
    harmonic_weights: Vec<f32>,

    /// Spectral tilt
    spectral_tilt: f32,

    /// Formant characteristics
    formant_emphasis: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum VoiceAllocation {
    RoundRobin,
    LeastRecent,
    LowestAmplitude,
    SmartStealing,
}

#[derive(Debug, Clone)]
pub struct VoiceStealingParams {
    /// Minimum voice age before stealing (in seconds)
    min_age_for_stealing: f32,

    /// Amplitude threshold for stealing
    amplitude_threshold: f32,

    /// Priority based on note range
    note_priority_curve: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoiceState {
    Attack,
    Sustain,
    Release,
    Free,
}

/// Unified output processor for final audio processing
#[derive(Debug, Clone)]
pub struct UnifiedOutputProcessor {
    /// Spatial processing for stereo field
    spatial_processor: SpatialProcessor,

    /// Dynamic range processing
    dynamics_processor: DynamicsProcessor,

    /// Final output filters
    output_filters: OutputFilterBank,

    /// Level control
    master_level: f32,
}

#[derive(Debug, Clone)]
pub struct SpatialProcessor {
    /// Stereo width control
    stereo_width: f32,

    /// Spatial movement parameters
    spatial_movement: SpatialMovement,

    /// Voice panning assignments
    voice_panning: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpatialMovement {
    /// Movement speed
    movement_speed: f32,

    /// Movement depth
    movement_depth: f32,

    /// Movement pattern
    movement_pattern: MovementPattern,
}

#[derive(Debug, Clone, Copy)]
pub enum MovementPattern {
    Static,
    Circular,
    PingPong,
    Random,
    RhythmSynced,
}

#[derive(Debug, Clone)]
pub struct DynamicsProcessor {
    /// Compression ratio
    compression_ratio: f32,

    /// Compression threshold
    threshold: f32,

    /// Limiter for peak control
    limiter_enabled: bool,

    /// Dynamic range expansion for quiet passages
    expansion_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct OutputFilterBank {
    /// High-pass filter for low-end cleanup
    highpass_cutoff: f32,

    /// Low-pass filter for high-end smoothing
    lowpass_cutoff: f32,

    /// Presence/air boost
    presence_boost: f32,

    /// Filter slopes
    filter_slopes: FilterSlopes,
}

#[derive(Debug, Clone)]
pub struct FilterSlopes {
    /// High-pass slope (db/octave)
    highpass_slope: f32,

    /// Low-pass slope (db/octave)
    lowpass_slope: f32,
}

impl UnifiedSynthesizer {
    /// Create a new unified synthesizer
    pub fn new(sample_rate: f32) -> Result<Self> {
        let rhythm_generator = UnifiedRhythmGenerator::new(sample_rate)?;
        let melody_generator = UnifiedMelodyGenerator::new(sample_rate)?;
        let harmony_generator = UnifiedHarmonyGenerator::new(sample_rate)?;

        let wavetable_synth = MorphingWavetableSynth::new(sample_rate, WaveCharacter::Organic);
        let additive_synth = AdditiveSynthesizer::new(sample_rate, 16); // 16 harmonics
        let crossfade_manager = CrossfadeManager::new(sample_rate)?;

        let variation = NaturalVariation::new(None);

        Ok(Self {
            rhythm_generator,
            melody_generator,
            harmony_generator,
            wavetable_synth,
            additive_synth,
            synthesis_morph: SynthesisMorpher::new(),
            voice_manager: VoiceManager::new(),
            output_processor: UnifiedOutputProcessor::new(),
            crossfade_manager,
            input_value: 0.0,
            variation,
            sample_counter: 0,
            sample_rate,
            rng: StdRng::from_entropy(),
        })
    }

    /// Set the input value and update all systems with crossfading
    pub fn set_input_value(&mut self, input: f32) {
        let clamped_input = input.clamp(0.0, 1.0);

        // Request crossfaded input value change
        if let Ok(Some(_crossfade_id)) = self.crossfade_manager.request_parameter_change(
            CrossfadeParameter::InputValue,
            clamped_input,
            CrossfadePriority::Normal,
        ) {
            // Crossfade was initiated
        } else {
            // Change was too small or rate-limited, apply immediately
            self.apply_input_value_immediate(clamped_input);
        }
    }

    /// Set input value immediately without crossfading (for initialization)
    pub fn set_input_value_immediate(&mut self, input: f32) {
        let clamped_input = input.clamp(0.0, 1.0);
        self.apply_input_value_immediate(clamped_input);
    }

    /// Apply input value changes to all systems
    fn apply_input_value_immediate(&mut self, input: f32) {
        self.input_value = input;

        // Update pattern generators
        self.rhythm_generator.set_input_value(input);
        self.melody_generator.set_input_value(input);
        self.harmony_generator.set_input_value(input);

        // Update synthesis morphing
        self.synthesis_morph.update_from_input(input);

        // Update output processing
        self.output_processor.update_from_input(input);
    }

    /// Generate the next audio frame
    pub fn process_sample(&mut self) -> StereoFrame {
        // Update natural variation and sample counter
        self.variation.update();
        self.sample_counter = self.sample_counter.wrapping_add(1);

        // Update crossfade manager
        let delta_time = 1.0 / self.sample_rate;
        self.crossfade_manager.update(delta_time);

        // Get crossfaded input value and apply if changed
        let crossfaded_input = self.crossfade_manager.get_parameter_value(CrossfadeParameter::InputValue);
        if (crossfaded_input - self.input_value).abs() > 0.001 {
            self.apply_input_value_immediate(crossfaded_input);
        }

        // Get current patterns from generators
        let rhythm_pattern = self.rhythm_generator.process_sample();
        let melody_pattern = self.melody_generator.process_sample(0.0); // beat_phase stub
        let harmony_pattern = self.harmony_generator.process_sample(0.0); // beat_phase stub

        // Update voice allocation based on patterns
        self.update_voice_allocation(&rhythm_pattern, &melody_pattern, &harmony_pattern);

        // Generate audio from active voices
        let mut output = StereoFrame::silence();

        // Collect voice data first to avoid borrow checker issues
        let mut voice_outputs = Vec::new();
        for voice in &self.voice_manager.voices {
            if voice.state != VoiceState::Free {
                voice_outputs.push((voice.frequency, voice.amplitude, voice.synth_params.clone(), voice.id));
            }
        }

        // Generate samples for each active voice
        for (frequency, amplitude, params, voice_id) in voice_outputs {
            let voice_output = self.generate_voice_sample_data(frequency, amplitude, &params, voice_id);
            output.left += voice_output.left;
            output.right += voice_output.right;
        }

        // Apply output processing
        self.output_processor.process(output)
    }


    /// Update voice allocation based on current patterns
    fn update_voice_allocation(
        &mut self,
        rhythm_pattern: &RhythmPattern,
        melody_pattern: &MelodyPattern,
        harmony_pattern: &HarmonyPattern,
    ) {
        // Trigger melody notes
        if !melody_pattern.is_rest {
            self.trigger_voice(
                melody_pattern.note,
                melody_pattern.velocity as f32 / 127.0, // Convert u8 to f32
                VoiceType::Melody,
            );
        }

        // Trigger harmony notes (simplified trigger logic)
        if rhythm_pattern.kick || rhythm_pattern.snare {
            for &note in &harmony_pattern.voicing {
                if note > 0 {
                    self.trigger_voice(note, 0.7, VoiceType::Harmony);
                }
            }
        }

        // Update voice ages and states
        self.voice_manager.update_voices(1.0 / self.sample_rate);
    }

    /// Trigger a new voice
    fn trigger_voice(&mut self, note: u8, velocity: f32, voice_type: VoiceType) {
        let voice_id = self.voice_manager.allocate_voice(note, velocity, voice_type);
        // Calculate synthesis parameters first
        let synth_params = self.calculate_voice_synth_params(voice_type);

        if let Some(voice) = self.voice_manager.get_voice_mut(voice_id) {
            voice.note = note;
            voice.frequency = midi_to_frequency(note);
            voice.amplitude = velocity;
            voice.age = 0.0;
            voice.state = VoiceState::Attack;
            voice.synth_params = synth_params;
        }
    }

    /// Calculate synthesis parameters for a voice
    fn calculate_voice_synth_params(&self, voice_type: VoiceType) -> VoiceSynthParams {
        let morph_info = self.synthesis_morph.get_current_morph_info();

        VoiceSynthParams {
            wavetable_params: WavetableVoiceParams {
                character: morph_info.wavetable_character,
                table_morph: morph_info.wavetable_morph,
                filter_cutoff: self.calculate_filter_cutoff(voice_type),
                filter_resonance: self.calculate_filter_resonance(voice_type),
            },
            additive_params: AdditiveVoiceParams {
                harmonic_count: self.calculate_harmonic_count(),
                harmonic_weights: self.calculate_harmonic_weights(),
                spectral_tilt: morph_info.spectral_tilt,
                formant_emphasis: self.calculate_formant_emphasis(voice_type),
            },
            morph_position: morph_info.morph_position,
        }
    }

    /// Generate audio sample for a single voice using data
    fn generate_voice_sample_data(
        &mut self,
        frequency: f32,
        amplitude: f32,
        params: &VoiceSynthParams,
        voice_id: usize
    ) -> StereoFrame {

        // Generate wavetable sample (stub implementation)
        let wavetable_sample = self.generate_wavetable_sample(
            frequency,
            amplitude,
            &params.wavetable_params,
        );

        // Generate additive sample (stub implementation)
        let additive_sample = self.generate_additive_sample(
            frequency,
            amplitude,
            &params.additive_params,
        );

        // Cross-fade between synthesis methods
        let morph_pos = params.morph_position;
        let wavetable_weight = 1.0 - morph_pos;
        let additive_weight = morph_pos;

        let mixed_sample = (wavetable_sample * wavetable_weight) + (additive_sample * additive_weight);

        // Apply simple envelope (stub implementation)
        let processed_sample = mixed_sample * amplitude;

        // Convert to stereo with voice panning
        let pan_position = self.voice_manager.get_voice_pan(voice_id);
        StereoFrame::new(
            processed_sample * (1.0 - pan_position).max(0.0),
            processed_sample * pan_position.max(0.0),
        )
    }


    /// Calculate attack envelope
    fn calculate_attack_envelope(&self, age: f32) -> f32 {
        let attack_time = self.calculate_attack_time();
        if age >= attack_time {
            1.0
        } else {
            (age / attack_time).powf(0.5) // Slight curve for natural feel
        }
    }

    /// Calculate release envelope
    fn calculate_release_envelope(&self, age: f32) -> f32 {
        let release_time = self.calculate_release_time();
        if age >= release_time {
            0.0
        } else {
            (1.0 - (age / release_time)).powf(2.0) // Exponential decay
        }
    }

    /// Calculate attack time based on input value
    fn calculate_attack_time(&self) -> f32 {
        match self.input_value {
            i if i <= 1.0 => 0.1 + (i * 0.4), // 0.1 to 0.5 seconds
            i if i <= 2.0 => {
                let t = i - 1.0;
                0.5 - (t * 0.3) // 0.5 to 0.2 seconds
            },
            _ => {
                let t = self.input_value - 2.0;
                0.2 - (t * 0.15) // 0.2 to 0.05 seconds
            }
        }
    }

    /// Calculate release time based on input value
    fn calculate_release_time(&self) -> f32 {
        match self.input_value {
            i if i <= 1.0 => 2.0 + (i * 3.0), // 2 to 5 seconds
            i if i <= 2.0 => {
                let t = i - 1.0;
                5.0 - (t * 2.0) // 5 to 3 seconds
            },
            _ => {
                let t = self.input_value - 2.0;
                3.0 - (t * 2.0) // 3 to 1 second
            }
        }
    }

    /// Calculate filter cutoff based on voice type and input
    fn calculate_filter_cutoff(&self, voice_type: VoiceType) -> f32 {
        let base_cutoff = match voice_type {
            VoiceType::Melody => 8000.0,
            VoiceType::Harmony => 4000.0,
            VoiceType::Bass => 2000.0,
        };

        // Modulate based on input value
        let input_factor = self.input_value;
        base_cutoff * (0.3 + input_factor * 0.7)
    }

    /// Calculate filter resonance
    fn calculate_filter_resonance(&self, _voice_type: VoiceType) -> f32 {
        0.1 + self.input_value * 0.4 // 0.1 to 0.5
    }

    /// Calculate number of harmonics for additive synthesis
    fn calculate_harmonic_count(&self) -> usize {
        match self.input_value {
            i if i <= 1.0 => 3 + (i * 5.0) as usize, // 3 to 8 harmonics
            i if i <= 2.0 => {
                let t = i - 1.0;
                8 + (t * 8.0) as usize // 8 to 16 harmonics
            },
            _ => {
                let t = self.input_value - 2.0;
                16 + (t * 16.0) as usize // 16 to 32 harmonics
            }
        }.min(64) // Cap at 64 harmonics
    }

    /// Calculate harmonic weights for additive synthesis
    fn calculate_harmonic_weights(&self) -> Vec<f32> {
        let harmonic_count = self.calculate_harmonic_count();
        let mut weights = Vec::with_capacity(harmonic_count);

        for i in 0..harmonic_count {
            let harmonic_num = i + 1;

            // Calculate weight based on harmonic number and input value
            let base_weight = 1.0 / (harmonic_num as f32).sqrt();

            // Modulate based on input value
            let input_factor = self.input_value;
            let weight = base_weight * (0.5 + input_factor * 0.5);

            weights.push(weight);
        }

        weights
    }

    /// Calculate formant emphasis based on voice type
    fn calculate_formant_emphasis(&self, voice_type: VoiceType) -> f32 {
        let base_emphasis = match voice_type {
            VoiceType::Melody => 0.7,
            VoiceType::Harmony => 0.4,
            VoiceType::Bass => 0.2,
        };

        base_emphasis * self.input_value
    }

    /// Generate wavetable sample (stub implementation)
    fn generate_wavetable_sample(
        &mut self,
        frequency: f32,
        amplitude: f32,
        _params: &WavetableVoiceParams,
    ) -> f32 {
        // Simple sine wave for now - would interface with actual wavetable synth
        use std::f32::consts::TAU;
        let phase = (frequency * TAU / self.sample_rate) * self.sample_counter as f32;
        (phase.sin() * amplitude).clamp(-1.0, 1.0)
    }

    /// Generate additive sample (stub implementation)
    fn generate_additive_sample(
        &mut self,
        frequency: f32,
        amplitude: f32,
        params: &AdditiveVoiceParams,
    ) -> f32 {
        // Simple additive synthesis - sum harmonics
        use std::f32::consts::TAU;
        let mut sample = 0.0;

        for (i, &weight) in params.harmonic_weights.iter().enumerate() {
            let harmonic_freq = frequency * (i + 1) as f32;
            let phase = (harmonic_freq * TAU / self.sample_rate) * self.sample_counter as f32;
            sample += (phase.sin() * weight).clamp(-1.0, 1.0);
        }

        (sample * amplitude).clamp(-1.0, 1.0)
    }

    /// Get current input value
    pub fn input_value(&self) -> f32 {
        self.input_value
    }

    /// Get crossfaded input value
    pub fn crossfaded_input_value(&self) -> f32 {
        self.crossfade_manager.get_parameter_value(CrossfadeParameter::InputValue)
    }

    /// Request parameter change with crossfading
    pub fn request_parameter_change(
        &mut self,
        parameter: CrossfadeParameter,
        value: f32,
        priority: CrossfadePriority,
    ) -> Result<Option<usize>> {
        self.crossfade_manager.request_parameter_change(parameter, value, priority)
    }

    /// Get crossfade statistics for monitoring
    pub fn get_crossfade_stats(&self) -> crate::audio::crossfade::CrossfadeStats {
        self.crossfade_manager.get_crossfade_stats()
    }

    /// Cancel crossfades for a specific parameter
    pub fn cancel_parameter_crossfades(&mut self, parameter: CrossfadeParameter) {
        self.crossfade_manager.cancel_parameter_crossfades(parameter);
    }

    /// Reset all systems
    pub fn reset(&mut self) {
        self.rhythm_generator.reset();
        self.melody_generator.reset();
        self.harmony_generator.reset();
        self.voice_manager.reset();
        self.output_processor.reset();
        self.sample_counter = 0;

        // Reset crossfade manager (would need to implement this method)
        // self.crossfade_manager.reset();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VoiceType {
    Melody,
    Harmony,
    Bass,
}

/// Current morph information for synthesis engines
#[derive(Debug, Clone)]
pub struct MorphInfo {
    /// Morph position (0.0 = wavetable, 1.0 = additive)
    pub morph_position: f32,

    /// Wavetable character
    pub wavetable_character: WaveCharacter,

    /// Wavetable morph amount
    pub wavetable_morph: f32,

    /// Additive complexity
    pub additive_complexity: f32,

    /// Spectral parameters
    pub spectral_params: SpectralParams,

    /// Spectral tilt
    pub spectral_tilt: f32,
}

#[derive(Debug, Clone)]
pub struct SpectralParams {
    /// High frequency content
    pub high_freq_content: f32,

    /// Harmonic distribution
    pub harmonic_distribution: HarmonicDistribution,

    /// Formant characteristics
    pub formant_characteristics: FormantCharacteristics,
}

#[derive(Debug, Clone, Copy)]
pub enum HarmonicDistribution {
    Natural,     // 1/f falloff
    Even,        // Even harmonics emphasized
    Odd,         // Odd harmonics emphasized
    Inharmonic,  // Non-integer ratios
}

#[derive(Debug, Clone)]
pub struct FormantCharacteristics {
    /// Formant frequency ratios
    pub formant_ratios: Vec<f32>,

    /// Formant bandwidths
    pub formant_bandwidths: Vec<f32>,

    /// Formant gains
    pub formant_gains: Vec<f32>,
}

// Helper function for MIDI note to frequency conversion
fn midi_to_frequency(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

// Implementation stubs for supporting structures
impl SynthesisMorpher {
    fn new() -> Self {
        Self {
            morph_position: 0.0,
            morph_curve: MorphCurve::default(),
            crossfade_smoothing: 0.95,
        }
    }

    fn update_from_input(&mut self, input: f32) {
        self.morph_position = self.morph_curve.calculate_morph_position(input);
    }

    fn get_current_morph_info(&self) -> MorphInfo {
        // Calculate current synthesis parameters based on morph position
        let wavetable_character = self.calculate_wavetable_character();
        let additive_complexity = self.morph_position;

        MorphInfo {
            morph_position: self.morph_position,
            wavetable_character,
            wavetable_morph: 1.0 - self.morph_position,
            additive_complexity,
            spectral_params: self.calculate_spectral_params(),
            spectral_tilt: self.calculate_spectral_tilt(),
        }
    }

    fn calculate_wavetable_character(&self) -> WaveCharacter {
        match self.morph_position {
            p if p < 0.2 => WaveCharacter::Organic,
            p if p < 0.4 => WaveCharacter::Ethereal,
            p if p < 0.6 => WaveCharacter::Harmonic,
            p if p < 0.8 => WaveCharacter::Crystalline,
            _ => WaveCharacter::Synthetic,
        }
    }

    fn calculate_spectral_params(&self) -> SpectralParams {
        SpectralParams {
            high_freq_content: self.morph_position,
            harmonic_distribution: if self.morph_position < 0.5 {
                HarmonicDistribution::Natural
            } else {
                HarmonicDistribution::Inharmonic
            },
            formant_characteristics: FormantCharacteristics {
                formant_ratios: vec![1.0, 2.5, 4.2, 6.8],
                formant_bandwidths: vec![0.1, 0.15, 0.2, 0.3],
                formant_gains: vec![0.8, 0.6, 0.4, 0.2],
            },
        }
    }

    fn calculate_spectral_tilt(&self) -> f32 {
        -6.0 + (self.morph_position * 12.0) // -6 dB/oct to +6 dB/oct
    }
}

impl MorphCurve {
    fn default() -> Self {
        Self {
            control_points: vec![
                MorphPoint {
                    input: 0.0,
                    wavetable_weight: 1.0,
                    additive_weight: 0.0,
                    wavetable_character: WaveCharacter::Organic,
                    additive_complexity: 0.1,
                },
                MorphPoint {
                    input: 1.5,
                    wavetable_weight: 0.7,
                    additive_weight: 0.3,
                    wavetable_character: WaveCharacter::Harmonic,
                    additive_complexity: 0.5,
                },
                MorphPoint {
                    input: 1.0,
                    wavetable_weight: 0.2,
                    additive_weight: 0.8,
                    wavetable_character: WaveCharacter::Synthetic,
                    additive_complexity: 1.0,
                },
            ],
            interpolation: MorphInterpolation::Smooth,
        }
    }

    fn calculate_morph_position(&self, input: f32) -> f32 {
        // Interpolate between control points
        if input <= self.control_points[0].input {
            return self.control_points[0].additive_weight;
        }

        for i in 0..self.control_points.len() - 1 {
            let p1 = &self.control_points[i];
            let p2 = &self.control_points[i + 1];

            if input >= p1.input && input <= p2.input {
                let t = (input - p1.input) / (p2.input - p1.input);
                return self.interpolate(p1.additive_weight, p2.additive_weight, t);
            }
        }

        self.control_points.last().unwrap().additive_weight
    }

    fn interpolate(&self, a: f32, b: f32, t: f32) -> f32 {
        match self.interpolation {
            MorphInterpolation::Linear => a + (b - a) * t,
            MorphInterpolation::Smooth => {
                let t_smooth = t * t * (3.0 - 2.0 * t);
                a + (b - a) * t_smooth
            },
            MorphInterpolation::Exponential => a + (b - a) * t * t,
        }
    }
}

impl VoiceManager {
    fn new() -> Self {
        Self {
            voices: Vec::new(),
            max_voices: 16, // Reasonable polyphony limit
            allocation_strategy: VoiceAllocation::SmartStealing,
            stealing_params: VoiceStealingParams::default(),
        }
    }

    fn allocate_voice(&mut self, note: u8, velocity: f32, voice_type: VoiceType) -> usize {
        // Try to find a free voice first
        for (i, voice) in self.voices.iter().enumerate() {
            if voice.state == VoiceState::Free {
                return i;
            }
        }

        // If no free voices, try to expand if under limit
        if self.voices.len() < self.max_voices {
            let voice_id = self.voices.len();
            self.voices.push(SynthVoice::new(voice_id, note));
            return voice_id;
        }

        // Need to steal a voice
        self.steal_voice(note, velocity, voice_type)
    }

    fn steal_voice(&mut self, _note: u8, _velocity: f32, _voice_type: VoiceType) -> usize {
        // Find the best voice to steal based on strategy
        match self.allocation_strategy {
            VoiceAllocation::LeastRecent => {
                let mut oldest_voice = 0;
                let mut oldest_age = 0.0;

                for (i, voice) in self.voices.iter().enumerate() {
                    if voice.age > oldest_age {
                        oldest_age = voice.age;
                        oldest_voice = i;
                    }
                }
                oldest_voice
            },
            VoiceAllocation::LowestAmplitude => {
                let mut quietest_voice = 0;
                let mut lowest_amplitude = f32::MAX;

                for (i, voice) in self.voices.iter().enumerate() {
                    if voice.amplitude < lowest_amplitude {
                        lowest_amplitude = voice.amplitude;
                        quietest_voice = i;
                    }
                }
                quietest_voice
            },
            _ => 0, // Default to first voice
        }
    }

    fn get_voice_mut(&mut self, voice_id: usize) -> Option<&mut SynthVoice> {
        self.voices.get_mut(voice_id)
    }

    fn update_voices(&mut self, delta_time: f32) {
        for voice in &mut self.voices {
            if voice.state != VoiceState::Free {
                voice.age += delta_time;

                // Update voice state transitions
                match voice.state {
                    VoiceState::Attack => {
                        if voice.age > 0.1 { // Simple attack time
                            voice.state = VoiceState::Sustain;
                        }
                    },
                    VoiceState::Release => {
                        if voice.age > 2.0 { // Simple release time
                            voice.state = VoiceState::Free;
                            voice.age = 0.0;
                        }
                    },
                    _ => {},
                }
            }
        }
    }

    fn get_voice_pan(&self, voice_id: usize) -> f32 {
        // Simple panning based on voice ID
        (voice_id as f32 / self.max_voices as f32).clamp(0.0, 1.0)
    }

    fn reset(&mut self) {
        for voice in &mut self.voices {
            voice.state = VoiceState::Free;
            voice.age = 0.0;
        }
    }
}

impl SynthVoice {
    fn new(id: usize, note: u8) -> Self {
        Self {
            id,
            note,
            frequency: midi_to_frequency(note),
            amplitude: 0.0,
            age: 0.0,
            synth_params: VoiceSynthParams::default(),
            state: VoiceState::Free,
        }
    }
}

impl VoiceSynthParams {
    fn default() -> Self {
        Self {
            wavetable_params: WavetableVoiceParams::default(),
            additive_params: AdditiveVoiceParams::default(),
            morph_position: 0.0,
        }
    }
}

impl WavetableVoiceParams {
    fn default() -> Self {
        Self {
            character: WaveCharacter::Organic,
            table_morph: 0.0,
            filter_cutoff: 4000.0,
            filter_resonance: 0.1,
        }
    }
}

impl AdditiveVoiceParams {
    fn default() -> Self {
        Self {
            harmonic_count: 8,
            harmonic_weights: vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
            spectral_tilt: -3.0,
            formant_emphasis: 0.5,
        }
    }
}

impl VoiceStealingParams {
    fn default() -> Self {
        Self {
            min_age_for_stealing: 0.1,
            amplitude_threshold: 0.1,
            note_priority_curve: vec![1.0; 128], // Equal priority for all notes
        }
    }
}

impl UnifiedOutputProcessor {
    fn new() -> Self {
        Self {
            spatial_processor: SpatialProcessor::new(),
            dynamics_processor: DynamicsProcessor::new(),
            output_filters: OutputFilterBank::new(),
            master_level: 0.8,
        }
    }

    fn update_from_input(&mut self, input: f32) {
        self.spatial_processor.update_from_input(input);
        self.dynamics_processor.update_from_input(input);
        self.output_filters.update_from_input(input);
    }

    fn process(&mut self, input: StereoFrame) -> StereoFrame {
        let mut output = input;

        // Apply spatial processing
        output = self.spatial_processor.process(output);

        // Apply dynamics processing
        output = self.dynamics_processor.process(output);

        // Apply output filtering
        output = self.output_filters.process(output);

        // Apply master level
        StereoFrame::new(
            output.left * self.master_level,
            output.right * self.master_level,
        )
    }

    fn reset(&mut self) {
        self.spatial_processor.reset();
        self.dynamics_processor.reset();
        self.output_filters.reset();
    }
}

impl SpatialProcessor {
    fn new() -> Self {
        Self {
            stereo_width: 1.0,
            spatial_movement: SpatialMovement::new(),
            voice_panning: Vec::new(),
        }
    }

    fn update_from_input(&mut self, input: f32) {
        self.stereo_width = 0.5 + input * 0.5; // 0.5 to 1.0
        self.spatial_movement.update_from_input(input);
    }

    fn process(&mut self, input: StereoFrame) -> StereoFrame {
        // Apply stereo width
        let mid = (input.left + input.right) * 0.5;
        let side = (input.left - input.right) * 0.5 * self.stereo_width;

        StereoFrame::new(
            mid + side,
            mid - side,
        )
    }

    fn reset(&mut self) {
        self.spatial_movement.reset();
    }
}

impl SpatialMovement {
    fn new() -> Self {
        Self {
            movement_speed: 0.1,
            movement_depth: 0.2,
            movement_pattern: MovementPattern::Static,
        }
    }

    fn update_from_input(&mut self, input: f32) {
        self.movement_speed = 0.05 + input * 0.15;
        self.movement_depth = 0.1 + input * 0.3;

        self.movement_pattern = match input {
            i if i < 1.0 => MovementPattern::Static,
            i if i < 2.0 => MovementPattern::Circular,
            _ => MovementPattern::RhythmSynced,
        };
    }

    fn reset(&mut self) {
        // Reset movement state
    }
}

impl DynamicsProcessor {
    fn new() -> Self {
        Self {
            compression_ratio: 2.0,
            threshold: -12.0,
            limiter_enabled: true,
            expansion_ratio: 1.0,
        }
    }

    fn update_from_input(&mut self, input: f32) {
        // More compression at higher input values
        self.compression_ratio = 1.0 + input * 3.0; // 1.0 to 4.0
        self.threshold = -20.0 + input * 8.0; // -20 to -12 dB
    }

    fn process(&mut self, input: StereoFrame) -> StereoFrame {
        // Simple compression approximation
        let left_compressed = self.compress_sample(input.left);
        let right_compressed = self.compress_sample(input.right);

        StereoFrame::new(left_compressed, right_compressed)
    }

    fn compress_sample(&self, sample: f32) -> f32 {
        let threshold_linear = 10.0_f32.powf(self.threshold / 20.0);
        let abs_sample = sample.abs();

        if abs_sample > threshold_linear {
            let excess = abs_sample - threshold_linear;
            let compressed_excess = excess / self.compression_ratio;
            sample.signum() * (threshold_linear + compressed_excess)
        } else {
            sample
        }
    }

    fn reset(&mut self) {
        // Reset compressor state
    }
}

impl OutputFilterBank {
    fn new() -> Self {
        Self {
            highpass_cutoff: 20.0,
            lowpass_cutoff: 20000.0,
            presence_boost: 0.0,
            filter_slopes: FilterSlopes::new(),
        }
    }

    fn update_from_input(&mut self, input: f32) {
        // Adjust filtering based on input
        self.highpass_cutoff = 20.0 + input * 30.0; // 20 to 50 Hz
        self.lowpass_cutoff = 20000.0 - input * 5000.0; // 20kHz to 15kHz
        self.presence_boost = input * 3.0; // 0 to 3 dB
    }

    fn process(&mut self, input: StereoFrame) -> StereoFrame {
        // Simple filtering approximation - in real implementation would use proper filters
        input // For now, pass through
    }

    fn reset(&mut self) {
        // Reset filter states
    }
}

impl FilterSlopes {
    fn new() -> Self {
        Self {
            highpass_slope: 12.0, // dB/octave
            lowpass_slope: 24.0,  // dB/octave
        }
    }
}