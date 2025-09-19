use crate::patterns::{PatternGenerator, PatternParameters};
use crate::error::Result;

/// Melody pattern representation
#[derive(Debug, Clone)]
pub struct MelodyPattern {
    pub note: u8,           // MIDI note number
    pub velocity: u8,       // Note velocity (0-127)
    pub duration: f32,      // Note duration in beats
    pub is_rest: bool,      // Whether this is a rest
}

impl Default for MelodyPattern {
    fn default() -> Self {
        Self {
            note: 60,     // C4
            velocity: 64, // Medium velocity
            duration: 1.0, // Quarter note
            is_rest: false,
        }
    }
}

/// Melody pattern generator
pub struct MelodyGenerator {
    parameters: PatternParameters,
    current_pattern: MelodyPattern,
    pattern_position: usize,
    pattern_length: usize,
}

impl MelodyGenerator {
    pub fn new(parameters: &PatternParameters) -> Result<Self> {
        Ok(Self {
            parameters: parameters.clone(),
            current_pattern: MelodyPattern::default(),
            pattern_position: 0,
            pattern_length: 16, // 16 notes default
        })
    }
}

impl PatternGenerator for MelodyGenerator {
    type Output = MelodyPattern;

    fn next(&mut self) -> Self::Output {
        // Simple placeholder melody
        let scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]; // C major scale
        let note_index = self.pattern_position % scale_notes.len();

        self.current_pattern = MelodyPattern {
            note: scale_notes[note_index],
            velocity: (64.0 * self.parameters.intensity) as u8,
            duration: 1.0,
            is_rest: self.pattern_position % 8 == 7, // Rest on 8th beat
        };

        self.pattern_position = (self.pattern_position + 1) % self.pattern_length;
        self.current_pattern.clone()
    }

    fn reset(&mut self) {
        self.pattern_position = 0;
    }

    fn update_parameters(&mut self, params: &PatternParameters) {
        self.parameters = params.clone();
    }

    fn pattern_length(&self) -> usize {
        self.pattern_length
    }

    fn is_cycle_complete(&self) -> bool {
        self.pattern_position == 0
    }
}