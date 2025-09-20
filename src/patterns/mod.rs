//! Pattern generation and management for mood music

pub mod markov;
pub mod genetic;
pub mod rhythm;
pub mod unified_rhythm;
pub mod melody;
pub mod unified_melody;
pub mod harmony;
pub mod unified_harmony;

use rand::Rng;
use crate::error::Result;

pub use markov::MarkovChain;
pub use genetic::{GeneticRhythm, RhythmChromosome, FitnessCriteria};
pub use rhythm::{RhythmPattern, RhythmGenerator};
pub use unified_rhythm::{UnifiedRhythmGenerator, AdaptivePattern, GrooveMorpher};
pub use melody::{MelodyPattern, MelodyGenerator};
pub use unified_melody::{UnifiedMelodyGenerator, ContinuousScaleMorpher, AdaptivePhraseGenerator};
pub use harmony::{HarmonyPattern, HarmonyGenerator};
pub use unified_harmony::{UnifiedHarmonyGenerator, ChordStructure, AdaptiveChordProgression};

/// Base trait for all pattern generators
pub trait PatternGenerator {
    type Output;

    /// Generate the next element in the pattern
    fn next(&mut self) -> Self::Output;

    /// Reset the generator to its initial state
    fn reset(&mut self);

    /// Update the generator's parameters
    fn update_parameters(&mut self, params: &PatternParameters);

    /// Get the current pattern length in samples
    fn pattern_length(&self) -> usize;

    /// Check if the pattern has completed a cycle
    fn is_cycle_complete(&self) -> bool;
}

/// Common parameters for pattern generation
#[derive(Debug, Clone)]
pub struct PatternParameters {
    /// Intensity level (0.0 to 1.0)
    pub intensity: f32,

    /// Complexity level (0.0 to 1.0)
    pub complexity: f32,

    /// Variation amount (0.0 to 1.0)
    pub variation: f32,

    /// Tempo in BPM
    pub tempo: f32,

    /// Scale or key information
    pub scale: Scale,

    /// Time signature
    pub time_signature: TimeSignature,

    /// Random seed for reproducible patterns
    pub seed: Option<u64>,
}

impl Default for PatternParameters {
    fn default() -> Self {
        Self {
            intensity: 0.5,
            complexity: 0.5,
            variation: 0.3,
            tempo: 120.0,
            scale: Scale::CMajor,
            time_signature: TimeSignature::FourFour,
            seed: None,
        }
    }
}

/// Musical scale definitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    CMajor,
    DMinor,
    GMajor,
    AMinor,
    FMajor,
    EMinor,
    Pentatonic,
    Blues,
    Chromatic,
}

impl Scale {
    /// Get the note intervals for this scale (in semitones from root)
    pub fn intervals(&self) -> &'static [u8] {
        match self {
            Scale::CMajor => &[0, 2, 4, 5, 7, 9, 11],
            Scale::DMinor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::GMajor => &[0, 2, 4, 5, 7, 9, 11],
            Scale::AMinor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::FMajor => &[0, 2, 4, 5, 7, 9, 11],
            Scale::EMinor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::Pentatonic => &[0, 2, 4, 7, 9],
            Scale::Blues => &[0, 3, 5, 6, 7, 10],
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }

    /// Get a random note from this scale
    pub fn random_note<R: Rng>(&self, rng: &mut R, octave: u8) -> u8 {
        let intervals = self.intervals();
        let interval = intervals[rng.gen_range(0..intervals.len())];
        (octave * 12) + interval
    }

    /// Get the root note (as MIDI note number)
    pub fn root_note(&self) -> u8 {
        match self {
            Scale::CMajor => 60, // C4
            Scale::DMinor => 62, // D4
            Scale::GMajor => 67, // G4
            Scale::AMinor => 57, // A3
            Scale::FMajor => 65, // F4
            Scale::EMinor => 64, // E4
            Scale::Pentatonic => 60, // C4
            Scale::Blues => 60, // C4
            Scale::Chromatic => 60, // C4
        }
    }
}

/// Time signature definitions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSignature {
    FourFour,    // 4/4
    ThreeFour,   // 3/4
    TwoFour,     // 2/4
    SixEight,    // 6/8
    SevenEight,  // 7/8
    FiveFour,    // 5/4
}

impl TimeSignature {
    /// Get the number of beats per measure
    pub fn beats_per_measure(&self) -> u8 {
        match self {
            TimeSignature::FourFour => 4,
            TimeSignature::ThreeFour => 3,
            TimeSignature::TwoFour => 2,
            TimeSignature::SixEight => 6,
            TimeSignature::SevenEight => 7,
            TimeSignature::FiveFour => 5,
        }
    }

    /// Get the note value that gets one beat
    pub fn beat_unit(&self) -> u8 {
        match self {
            TimeSignature::FourFour => 4,  // Quarter note
            TimeSignature::ThreeFour => 4, // Quarter note
            TimeSignature::TwoFour => 4,   // Quarter note
            TimeSignature::SixEight => 8,  // Eighth note
            TimeSignature::SevenEight => 8, // Eighth note
            TimeSignature::FiveFour => 4,  // Quarter note
        }
    }
}

/// Pattern manager that coordinates multiple pattern generators
pub struct PatternManager {
    rhythm_generator: RhythmGenerator,
    melody_generator: MelodyGenerator,
    harmony_generator: HarmonyGenerator,
    parameters: PatternParameters,
    sample_rate: f32,
    current_position: usize,
    pattern_length: usize,
}

impl PatternManager {
    /// Create a new pattern manager
    pub fn new(sample_rate: f32, parameters: PatternParameters) -> Result<Self> {
        let rhythm_generator = RhythmGenerator::new(&parameters)?;
        let melody_generator = MelodyGenerator::new(&parameters)?;
        let harmony_generator = HarmonyGenerator::new(&parameters)?;

        // Calculate pattern length in samples (e.g., 4 measures)
        let measures = 4;
        let beats_per_measure = parameters.time_signature.beats_per_measure();
        let total_beats = measures * beats_per_measure as usize;
        let samples_per_beat = (60.0 / parameters.tempo * sample_rate) as usize;
        let pattern_length = total_beats * samples_per_beat;

        Ok(Self {
            rhythm_generator,
            melody_generator,
            harmony_generator,
            parameters,
            sample_rate,
            current_position: 0,
            pattern_length,
        })
    }

    /// Get the next pattern state
    pub fn next(&mut self) -> PatternState {
        let rhythm = self.rhythm_generator.next();
        let melody = self.melody_generator.next();
        let harmony = self.harmony_generator.next();

        self.current_position += 1;
        if self.current_position >= self.pattern_length {
            self.current_position = 0;
        }

        PatternState {
            rhythm,
            melody,
            harmony,
            position: self.current_position,
            progress: self.current_position as f32 / self.pattern_length as f32,
        }
    }

    /// Update pattern parameters
    pub fn update_parameters(&mut self, new_params: PatternParameters) {
        self.rhythm_generator.update_parameters(&new_params);
        self.melody_generator.update_parameters(&new_params);
        self.harmony_generator.update_parameters(&new_params);
        self.parameters = new_params;

        // Recalculate pattern length if tempo changed
        let measures = 4;
        let beats_per_measure = self.parameters.time_signature.beats_per_measure();
        let total_beats = measures * beats_per_measure as usize;
        let samples_per_beat = (60.0 / self.parameters.tempo * self.sample_rate) as usize;
        self.pattern_length = total_beats * samples_per_beat;
    }

    /// Reset all generators
    pub fn reset(&mut self) {
        self.rhythm_generator.reset();
        self.melody_generator.reset();
        self.harmony_generator.reset();
        self.current_position = 0;
    }

    /// Check if pattern cycle is complete
    pub fn is_cycle_complete(&self) -> bool {
        self.current_position == 0
    }

    /// Get current pattern progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        self.current_position as f32 / self.pattern_length as f32
    }
}

/// Current state of all pattern generators
#[derive(Debug, Clone)]
pub struct PatternState {
    pub rhythm: RhythmPattern,
    pub melody: MelodyPattern,
    pub harmony: HarmonyPattern,
    pub position: usize,
    pub progress: f32,
}

/// Utility functions for pattern generation
pub mod utils {
    use super::*;
    use rand::Rng;

    /// Generate a random scale appropriate for the given mood
    pub fn scale_for_mood<R: Rng>(rng: &mut R, mood: f32) -> Scale {
        match mood {
            m if m < 0.25 => {
                // Environmental - simple, natural scales
                let scales = [Scale::Pentatonic, Scale::CMajor, Scale::AMinor];
                scales[rng.gen_range(0..scales.len())]
            }
            m if m < 0.5 => {
                // Gentle melodic - consonant scales
                let scales = [Scale::CMajor, Scale::FMajor, Scale::GMajor];
                scales[rng.gen_range(0..scales.len())]
            }
            m if m < 0.75 => {
                // Active ambient - more variety
                let scales = [Scale::GMajor, Scale::DMinor, Scale::EMinor];
                scales[rng.gen_range(0..scales.len())]
            }
            _ => {
                // EDM - more complex and chromatic
                let scales = [Scale::Blues, Scale::Chromatic, Scale::DMinor];
                scales[rng.gen_range(0..scales.len())]
            }
        }
    }

    /// Generate a tempo appropriate for the given mood
    pub fn tempo_for_mood<R: Rng>(rng: &mut R, mood: f32) -> f32 {
        match mood {
            m if m < 0.25 => rng.gen_range(60.0..80.0),    // Slow, ambient
            m if m < 0.5 => rng.gen_range(70.0..90.0),     // Gentle
            m if m < 0.75 => rng.gen_range(90.0..120.0),   // Active
            _ => rng.gen_range(120.0..140.0),              // Energetic EDM
        }
    }

    /// Generate complexity level appropriate for the given mood
    pub fn complexity_for_mood(mood: f32) -> f32 {
        match mood {
            m if m < 0.25 => 0.2,  // Very simple
            m if m < 0.5 => 0.4,   // Simple
            m if m < 0.75 => 0.6,  // Moderate
            _ => 0.8,              // Complex
        }
    }

    /// Convert MIDI note number to frequency in Hz
    pub fn midi_to_frequency(midi_note: u8) -> f32 {
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    /// Convert frequency in Hz to MIDI note number
    pub fn frequency_to_midi(frequency: f32) -> u8 {
        (69.0 + 12.0 * (frequency / 440.0).log2()).round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_scale_intervals() {
        let c_major = Scale::CMajor;
        assert_eq!(c_major.intervals(), &[0, 2, 4, 5, 7, 9, 11]);

        let pentatonic = Scale::Pentatonic;
        assert_eq!(pentatonic.intervals(), &[0, 2, 4, 7, 9]);
    }

    #[test]
    fn test_random_note_generation() {
        let mut rng = StdRng::seed_from_u64(42);
        let scale = Scale::CMajor;

        for _ in 0..100 {
            let note = scale.random_note(&mut rng, 4);
            assert!(note >= 48 && note <= 83); // Valid MIDI range for octave 4
        }
    }

    #[test]
    fn test_time_signature() {
        assert_eq!(TimeSignature::FourFour.beats_per_measure(), 4);
        assert_eq!(TimeSignature::ThreeFour.beats_per_measure(), 3);
        assert_eq!(TimeSignature::SixEight.beats_per_measure(), 6);
    }

    #[test]
    fn test_pattern_parameters_default() {
        let params = PatternParameters::default();
        assert_eq!(params.tempo, 120.0);
        assert_eq!(params.scale, Scale::CMajor);
        assert_eq!(params.time_signature, TimeSignature::FourFour);
    }

    #[test]
    fn test_utils_midi_frequency_conversion() {
        use utils::*;

        // A4 = 440 Hz = MIDI note 69
        assert!((midi_to_frequency(69) - 440.0).abs() < 0.01);
        assert_eq!(frequency_to_midi(440.0), 69);

        // C4 = ~261.63 Hz = MIDI note 60
        let c4_freq = midi_to_frequency(60);
        assert!((c4_freq - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_mood_based_generation() {
        use utils::*;
        let mut rng = StdRng::seed_from_u64(42);

        // Test scale selection for different moods
        let env_scale = scale_for_mood(&mut rng, 0.1);
        let edm_scale = scale_for_mood(&mut rng, 0.9);

        // Environmental should be simpler
        assert!(matches!(env_scale, Scale::Pentatonic | Scale::CMajor | Scale::AMinor));

        // Test tempo for different moods
        let env_tempo = tempo_for_mood(&mut rng, 0.1);
        let edm_tempo = tempo_for_mood(&mut rng, 0.9);

        assert!(env_tempo < 80.0);
        assert!(edm_tempo > 120.0);

        // Test complexity
        assert!(complexity_for_mood(0.1) < complexity_for_mood(0.9));
    }
}