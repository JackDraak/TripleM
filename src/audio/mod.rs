//! Audio processing pipeline and related types

pub mod pipeline;
pub mod buffer;
pub mod mixer;
pub mod transition;
pub mod variation;
pub mod filters;
pub mod synthesis;
pub mod granular;
pub mod drums;
pub mod wavetables;
pub mod additive;
pub mod unified_synth;
pub mod crossfade;
pub mod unified_controller;
pub mod voice_coordination;

pub use pipeline::AudioPipeline;
pub use buffer::AudioBuffer;
pub use mixer::{OutputMixer, MoodWeights};
pub use transition::TransitionManager;
pub use variation::{NaturalVariation, MicroTiming, DynamicRange};
pub use filters::{StateVariableFilter, OnePoleFilter, MorphingFilter, BiquadFilter, FilterBank, FilterType, FilterOutput};
pub use synthesis::{Oscillator, OscillatorType, FmOperator, FmSynth, FmAlgorithm, AdsrEnvelope, SynthVoice, midi_to_frequency};
pub use granular::GranularEngine;
pub use drums::{DrumSynthesizer, EDMStyle, PercussionType};
pub use wavetables::{MorphingWavetableSynth, WaveCharacter, WavetableBank};
pub use additive::{AdditiveSynthesizer, PolyharmonicCharacter, HarmonicOscillator};
pub use unified_synth::{UnifiedSynthesizer, SynthesisMorpher, VoiceManager};
pub use crossfade::{CrossfadeManager, CrossfadeParameter, CrossfadePriority};
pub use unified_controller::{
    UnifiedController, ControlParameter, ParameterConstraints, ParameterCurve,
    PresetManager, Preset, PresetMetadata, SystemStatus, ControllerDiagnostics,
    OutputLevels, PerformanceMetrics, ChangeSource,
};
pub use voice_coordination::{
    VoiceCoordinator, MusicalContext, VoicePool, Voice, VoiceId, MusicalNote,
    VoiceEnvelope, VoiceRequest, VoiceState, VoiceRequirements, AudioEvent,
    AudioEventType, EventPriority, GeneratorTarget, MusicalKey, HarmonicContext,
};

// Re-export public types from voice coordination for easy access
pub use voice_coordination::{
    MusicalRole, Articulation, EnvelopePhase, ConflictResolutionStrategy,
    HarmonicFunction, RhythmicPosition, ScaleType, MusicalMode, ChordType,
    ChordExtension, ChordVoicing, RhythmInstrument, PhraseType, MusicalTime,
    Chord, VoiceAllocation, ConflictResolver, MusicalPriorityRules,
};

/// Represents a single audio frame (mono sample)
pub type AudioFrame = f32;

/// Represents a stereo audio frame
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoFrame {
    pub left: f32,
    pub right: f32,
}

impl StereoFrame {
    pub fn new(left: f32, right: f32) -> Self {
        Self { left, right }
    }

    pub fn mono(sample: f32) -> Self {
        Self { left: sample, right: sample }
    }

    pub fn silence() -> Self {
        Self { left: 0.0, right: 0.0 }
    }

    pub fn to_mono(&self) -> f32 {
        (self.left + self.right) * 0.5
    }
}

impl From<f32> for StereoFrame {
    fn from(sample: f32) -> Self {
        Self::mono(sample)
    }
}

impl From<(f32, f32)> for StereoFrame {
    fn from((left, right): (f32, f32)) -> Self {
        Self::new(left, right)
    }
}

/// Audio processing utilities
pub mod utils {
    use std::f32::consts::PI;

    /// Convert decibels to linear amplitude
    pub fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear amplitude to decibels
    pub fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.log10()
    }

    /// Apply soft clipping to prevent harsh distortion
    pub fn soft_clip(sample: f32, threshold: f32) -> f32 {
        if sample.abs() <= threshold {
            sample
        } else {
            let sign = sample.signum();
            sign * (threshold + (sample.abs() - threshold) / (1.0 + (sample.abs() - threshold)))
        }
    }

    /// Simple lowpass filter coefficient calculation
    pub fn lowpass_coefficient(cutoff_hz: f32, sample_rate: f32) -> f32 {
        let rc = 1.0 / (2.0 * PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        dt / (dt + rc)
    }

    /// Interpolate between two values using cosine interpolation
    pub fn cosine_interpolate(a: f32, b: f32, t: f32) -> f32 {
        let t_smooth = (1.0 - (t * PI).cos()) * 0.5;
        a * (1.0 - t_smooth) + b * t_smooth
    }

    /// Generate white noise sample
    pub fn white_noise(rng: &mut impl rand::Rng) -> f32 {
        rng.gen_range(-1.0..1.0)
    }

    /// Generate pink noise using a simple approximation
    pub fn pink_noise_simple(white: f32, state: &mut [f32; 7]) -> f32 {
        // Simple pink noise filter approximation
        state[0] = 0.99886 * state[0] + white * 0.0555179;
        state[1] = 0.99332 * state[1] + white * 0.0750759;
        state[2] = 0.96900 * state[2] + white * 0.1538520;
        state[3] = 0.86650 * state[3] + white * 0.3104856;
        state[4] = 0.55000 * state[4] + white * 0.5329522;
        state[5] = -0.7616 * state[5] - white * 0.0168980;

        let pink = state[0] + state[1] + state[2] + state[3] + state[4] + state[5] + state[6] + white * 0.5362;
        state[6] = white * 0.115926;

        pink * 0.11
    }

    /// Clamp a value between min and max
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        value.max(min).min(max)
    }

    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    /// Exponential smoothing for parameter changes
    pub fn exponential_smooth(current: f32, target: f32, factor: f32) -> f32 {
        current + (target - current) * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;

    #[test]
    fn test_stereo_frame() {
        let frame = StereoFrame::new(0.5, -0.3);
        assert_eq!(frame.left, 0.5);
        assert_eq!(frame.right, -0.3);
        assert_eq!(frame.to_mono(), 0.1);
    }

    #[test]
    fn test_db_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_linear(-6.0) - 0.5).abs() < 0.01);
        assert!((linear_to_db(1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_soft_clip() {
        assert_eq!(soft_clip(0.5, 0.8), 0.5);
        assert!(soft_clip(1.5, 0.8) < 1.5);
        assert!(soft_clip(1.5, 0.8) > 0.8);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(lerp(0.0, 1.0, 0.5), 0.5);
        assert_eq!(lerp(10.0, 20.0, 0.25), 12.5);

        let cos_interp = cosine_interpolate(0.0, 1.0, 0.5);
        assert!(cos_interp > 0.4 && cos_interp < 0.6);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(clamp(1.5, 0.0, 1.0), 1.0);
    }
}