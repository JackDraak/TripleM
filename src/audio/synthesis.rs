//! Advanced synthesis algorithms for rich timbres
//!
//! This module provides various synthesis techniques including
//! FM synthesis, additive synthesis, and voice management.

use std::f32::consts::{PI, TAU};

/// Basic oscillator types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OscillatorType {
    Sine,
    Sawtooth,
    Square,
    Triangle,
    Noise,
}

/// Basic oscillator implementation
#[derive(Debug, Clone)]
pub struct Oscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    osc_type: OscillatorType,
    sample_rate: f32,
}

impl Oscillator {
    /// Create a new oscillator
    pub fn new(sample_rate: f32, frequency: f32, osc_type: OscillatorType) -> Self {
        Self {
            phase: 0.0,
            frequency,
            amplitude: 1.0,
            osc_type,
            sample_rate,
        }
    }

    /// Set the frequency
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    /// Set the amplitude
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    /// Generate the next sample
    pub fn generate(&mut self) -> f32 {
        let phase_increment = self.frequency / self.sample_rate;
        self.phase += phase_increment;

        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        let sample = match self.osc_type {
            OscillatorType::Sine => (self.phase * TAU).sin(),
            OscillatorType::Sawtooth => self.phase * 2.0 - 1.0,
            OscillatorType::Square => if self.phase < 0.5 { -1.0 } else { 1.0 },
            OscillatorType::Triangle => {
                if self.phase < 0.5 {
                    self.phase * 4.0 - 1.0
                } else {
                    3.0 - self.phase * 4.0
                }
            },
            OscillatorType::Noise => (rand::random::<f32>() - 0.5) * 2.0,
        };

        sample * self.amplitude
    }

    /// Reset the oscillator phase
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// FM (Frequency Modulation) synthesis operator
#[derive(Debug, Clone)]
pub struct FmOperator {
    carrier: Oscillator,
    modulator: Oscillator,
    modulation_index: f32,
    feedback: f32,
    previous_output: f32,
}

impl FmOperator {
    /// Create a new FM operator
    pub fn new(sample_rate: f32, carrier_freq: f32, modulator_ratio: f32) -> Self {
        Self {
            carrier: Oscillator::new(sample_rate, carrier_freq, OscillatorType::Sine),
            modulator: Oscillator::new(sample_rate, carrier_freq * modulator_ratio, OscillatorType::Sine),
            modulation_index: 1.0,
            feedback: 0.0,
            previous_output: 0.0,
        }
    }

    /// Set the base frequency (updates both carrier and modulator)
    pub fn set_frequency(&mut self, frequency: f32) {
        let ratio = self.modulator.frequency / self.carrier.frequency;
        self.carrier.set_frequency(frequency);
        self.modulator.set_frequency(frequency * ratio);
    }

    /// Set the modulation index (depth of FM)
    pub fn set_modulation_index(&mut self, index: f32) {
        self.modulation_index = index.max(0.0);
    }

    /// Set the feedback amount
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 1.0);
    }

    /// Set the modulator frequency ratio
    pub fn set_modulator_ratio(&mut self, ratio: f32) {
        self.modulator.set_frequency(self.carrier.frequency * ratio);
    }

    /// Generate the next sample
    pub fn generate(&mut self) -> f32 {
        // Add feedback
        let feedback_mod = self.previous_output * self.feedback;

        // Generate modulator output
        let mod_output = self.modulator.generate() + feedback_mod;

        // Modulate the carrier frequency
        let modulated_freq = self.carrier.frequency + (mod_output * self.modulation_index * self.carrier.frequency);

        // Temporarily set carrier frequency
        let original_freq = self.carrier.frequency;
        self.carrier.set_frequency(modulated_freq);

        // Generate carrier output
        let output = self.carrier.generate();

        // Restore original frequency
        self.carrier.set_frequency(original_freq);

        // Store for feedback
        self.previous_output = output;

        output
    }

    /// Reset the operator
    pub fn reset(&mut self) {
        self.carrier.reset();
        self.modulator.reset();
        self.previous_output = 0.0;
    }
}

/// Multi-operator FM synthesizer
#[derive(Debug, Clone)]
pub struct FmSynth {
    operators: Vec<FmOperator>,
    algorithm: FmAlgorithm,
    sample_rate: f32,
}

/// FM synthesis algorithms (connections between operators)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FmAlgorithm {
    Stack,      // Op1 -> Op2 -> Op3 -> Op4 (serial)
    Parallel,   // All operators in parallel
    Mixed,      // Op1 -> Op2, Op3 -> Op4, then parallel
}

impl FmSynth {
    /// Create a new FM synthesizer
    pub fn new(sample_rate: f32, base_frequency: f32, algorithm: FmAlgorithm) -> Self {
        let operators = vec![
            FmOperator::new(sample_rate, base_frequency, 1.0),   // Carrier
            FmOperator::new(sample_rate, base_frequency, 2.0),   // 2:1 ratio
            FmOperator::new(sample_rate, base_frequency, 3.0),   // 3:1 ratio
            FmOperator::new(sample_rate, base_frequency, 4.0),   // 4:1 ratio
        ];

        Self {
            operators,
            algorithm,
            sample_rate,
        }
    }

    /// Set the base frequency for all operators
    pub fn set_frequency(&mut self, frequency: f32) {
        for op in &mut self.operators {
            op.set_frequency(frequency);
        }
    }

    /// Set modulation index for a specific operator
    pub fn set_modulation_index(&mut self, operator: usize, index: f32) {
        if let Some(op) = self.operators.get_mut(operator) {
            op.set_modulation_index(index);
        }
    }

    /// Generate the next sample
    pub fn generate(&mut self) -> f32 {
        match self.algorithm {
            FmAlgorithm::Stack => {
                // Serial connection: Op4 -> Op3 -> Op2 -> Op1 (carrier)
                let mut output = 0.0;
                if self.operators.len() >= 4 {
                    let op4_out = self.operators[3].generate();
                    let op3_out = self.operators[2].generate();
                    let op2_out = self.operators[1].generate();
                    output = self.operators[0].generate();

                    // Apply modulation in cascade
                    output += op2_out * 0.5;
                    output += op3_out * 0.3;
                    output += op4_out * 0.2;
                }
                output * 0.5
            },
            FmAlgorithm::Parallel => {
                // All operators in parallel
                let mut output = 0.0;
                for op in &mut self.operators {
                    output += op.generate();
                }
                output / self.operators.len() as f32
            },
            FmAlgorithm::Mixed => {
                // Two pairs: (Op1 + Op2) and (Op3 + Op4)
                let mut output = 0.0;
                if self.operators.len() >= 4 {
                    let pair1 = (self.operators[0].generate() + self.operators[1].generate()) * 0.5;
                    let pair2 = (self.operators[2].generate() + self.operators[3].generate()) * 0.5;
                    output = (pair1 + pair2) * 0.5;
                }
                output
            },
        }
    }

    /// Reset all operators
    pub fn reset(&mut self) {
        for op in &mut self.operators {
            op.reset();
        }
    }
}

/// ADSR (Attack, Decay, Sustain, Release) envelope generator
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    attack_time: f32,
    decay_time: f32,
    sustain_level: f32,
    release_time: f32,
    sample_rate: f32,

    // State
    stage: EnvelopeStage,
    current_level: f32,
    time_in_stage: f32,
    gate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EnvelopeStage {
    Attack,
    Decay,
    Sustain,
    Release,
    Idle,
}

impl AdsrEnvelope {
    /// Create a new ADSR envelope
    pub fn new(sample_rate: f32) -> Self {
        Self {
            attack_time: 0.1,
            decay_time: 0.2,
            sustain_level: 0.7,
            release_time: 0.5,
            sample_rate,
            stage: EnvelopeStage::Idle,
            current_level: 0.0,
            time_in_stage: 0.0,
            gate: false,
        }
    }

    /// Set ADSR parameters
    pub fn set_adsr(&mut self, attack: f32, decay: f32, sustain: f32, release: f32) {
        self.attack_time = attack.max(0.001);
        self.decay_time = decay.max(0.001);
        self.sustain_level = sustain.clamp(0.0, 1.0);
        self.release_time = release.max(0.001);
    }

    /// Trigger the envelope (note on)
    pub fn note_on(&mut self) {
        self.gate = true;
        self.stage = EnvelopeStage::Attack;
        self.time_in_stage = 0.0;
    }

    /// Release the envelope (note off)
    pub fn note_off(&mut self) {
        self.gate = false;
        if self.stage != EnvelopeStage::Release && self.stage != EnvelopeStage::Idle {
            self.stage = EnvelopeStage::Release;
            self.time_in_stage = 0.0;
        }
    }

    /// Generate the next envelope value
    pub fn generate(&mut self) -> f32 {
        let dt = 1.0 / self.sample_rate;
        self.time_in_stage += dt;

        match self.stage {
            EnvelopeStage::Attack => {
                self.current_level = self.time_in_stage / self.attack_time;
                if self.current_level >= 1.0 {
                    self.current_level = 1.0;
                    self.stage = EnvelopeStage::Decay;
                    self.time_in_stage = 0.0;
                }
            },
            EnvelopeStage::Decay => {
                let decay_progress = self.time_in_stage / self.decay_time;
                self.current_level = 1.0 - decay_progress * (1.0 - self.sustain_level);
                if decay_progress >= 1.0 {
                    self.current_level = self.sustain_level;
                    self.stage = EnvelopeStage::Sustain;
                    self.time_in_stage = 0.0;
                }
            },
            EnvelopeStage::Sustain => {
                self.current_level = self.sustain_level;
                if !self.gate {
                    self.stage = EnvelopeStage::Release;
                    self.time_in_stage = 0.0;
                }
            },
            EnvelopeStage::Release => {
                let release_progress = self.time_in_stage / self.release_time;
                let start_level = if self.stage == EnvelopeStage::Release { self.current_level } else { self.sustain_level };
                self.current_level = start_level * (1.0 - release_progress);
                if release_progress >= 1.0 {
                    self.current_level = 0.0;
                    self.stage = EnvelopeStage::Idle;
                }
            },
            EnvelopeStage::Idle => {
                self.current_level = 0.0;
            },
        }

        self.current_level.clamp(0.0, 1.0)
    }

    /// Check if the envelope is active
    pub fn is_active(&self) -> bool {
        self.stage != EnvelopeStage::Idle
    }

    /// Reset the envelope
    pub fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.current_level = 0.0;
        self.time_in_stage = 0.0;
        self.gate = false;
    }
}

/// Voice for polyphonic synthesis
#[derive(Debug, Clone)]
pub struct SynthVoice {
    pub fm_synth: FmSynth,
    pub envelope: AdsrEnvelope,
    pub note: u8,
    pub velocity: f32,
    pub is_active: bool,
}

impl SynthVoice {
    /// Create a new synthesis voice
    pub fn new(sample_rate: f32, note: u8, velocity: f32) -> Self {
        let frequency = midi_to_frequency(note);
        let mut voice = Self {
            fm_synth: FmSynth::new(sample_rate, frequency, FmAlgorithm::Mixed),
            envelope: AdsrEnvelope::new(sample_rate),
            note,
            velocity,
            is_active: true,
        };

        voice.envelope.note_on();
        voice
    }

    /// Generate the next sample
    pub fn generate(&mut self) -> f32 {
        if !self.is_active {
            return 0.0;
        }

        let env_level = self.envelope.generate();
        if !self.envelope.is_active() {
            self.is_active = false;
            return 0.0;
        }

        let synth_output = self.fm_synth.generate();
        synth_output * env_level * self.velocity
    }

    /// Release the voice
    pub fn note_off(&mut self) {
        self.envelope.note_off();
    }

    /// Reset the voice
    pub fn reset(&mut self) {
        self.fm_synth.reset();
        self.envelope.reset();
        self.is_active = false;
    }
}

/// Convert MIDI note number to frequency
pub fn midi_to_frequency(midi_note: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
}

/// Preset FM configurations for different timbres
pub mod fm_presets {
    use super::*;

    /// Create an electric piano-like FM patch
    pub fn electric_piano(sample_rate: f32, frequency: f32) -> FmSynth {
        let mut synth = FmSynth::new(sample_rate, frequency, FmAlgorithm::Mixed);
        synth.set_modulation_index(1, 2.0);  // Moderate modulation for bell-like tone
        synth.set_modulation_index(2, 1.5);
        synth.set_modulation_index(3, 0.8);
        synth
    }

    /// Create a warm bell-like FM patch
    pub fn warm_bell(sample_rate: f32, frequency: f32) -> FmSynth {
        let mut synth = FmSynth::new(sample_rate, frequency, FmAlgorithm::Stack);
        synth.set_modulation_index(1, 3.0);  // Higher modulation for metallic sound
        synth.set_modulation_index(2, 2.0);
        synth.set_modulation_index(3, 1.0);
        synth
    }

    /// Create a soft pad-like FM patch
    pub fn soft_pad(sample_rate: f32, frequency: f32) -> FmSynth {
        let mut synth = FmSynth::new(sample_rate, frequency, FmAlgorithm::Parallel);
        synth.set_modulation_index(1, 0.5);  // Low modulation for smooth sound
        synth.set_modulation_index(2, 0.3);
        synth.set_modulation_index(3, 0.2);
        synth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillator() {
        let mut osc = Oscillator::new(44100.0, 440.0, OscillatorType::Sine);
        let sample = osc.generate();
        assert!(sample.abs() <= 1.0);
    }

    #[test]
    fn test_fm_operator() {
        let mut op = FmOperator::new(44100.0, 440.0, 2.0);
        let sample = op.generate();
        assert!(sample.abs() <= 10.0); // FM can exceed normal bounds
    }

    #[test]
    fn test_fm_synth() {
        let mut synth = FmSynth::new(44100.0, 440.0, FmAlgorithm::Mixed);
        let sample = synth.generate();
        assert!(sample.is_finite());
    }

    #[test]
    fn test_adsr_envelope() {
        let mut env = AdsrEnvelope::new(44100.0);
        env.set_adsr(0.1, 0.1, 0.5, 0.2);
        env.note_on();

        let mut max_level = 0.0;
        for _ in 0..1000 {
            let level = env.generate();
            max_level = max_level.max(level);
        }

        assert!(max_level > 0.0);
    }

    #[test]
    fn test_synth_voice() {
        let mut voice = SynthVoice::new(44100.0, 60, 0.8); // C4
        let sample = voice.generate();
        assert!(voice.is_active);
        assert!(sample.is_finite());
    }

    #[test]
    fn test_midi_to_frequency() {
        assert!((midi_to_frequency(69) - 440.0).abs() < 0.1); // A4 = 440 Hz
        assert!((midi_to_frequency(60) - 261.63).abs() < 0.1); // C4 ≈ 261.63 Hz
    }
}