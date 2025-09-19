use crate::patterns::{PatternGenerator, PatternParameters};
use crate::error::Result;

/// Harmony pattern representation
#[derive(Debug, Clone)]
pub struct HarmonyPattern {
    pub root: u8,           // Root note of the chord
    pub chord_type: ChordType,
    pub inversion: u8,      // Chord inversion (0, 1, 2)
    pub voicing: Vec<u8>,   // Actual notes in the chord
}

#[derive(Debug, Clone, Copy)]
pub enum ChordType {
    Major,
    Minor,
    Diminished,
    Augmented,
    Major7,
    Minor7,
    Dominant7,
}

impl Default for HarmonyPattern {
    fn default() -> Self {
        Self {
            root: 60,     // C4
            chord_type: ChordType::Major,
            inversion: 0,
            voicing: vec![60, 64, 67], // C major triad
        }
    }
}

impl ChordType {
    /// Get the intervals for this chord type
    pub fn intervals(&self) -> &'static [u8] {
        match self {
            ChordType::Major => &[0, 4, 7],
            ChordType::Minor => &[0, 3, 7],
            ChordType::Diminished => &[0, 3, 6],
            ChordType::Augmented => &[0, 4, 8],
            ChordType::Major7 => &[0, 4, 7, 11],
            ChordType::Minor7 => &[0, 3, 7, 10],
            ChordType::Dominant7 => &[0, 4, 7, 10],
        }
    }
}

/// Harmony pattern generator
pub struct HarmonyGenerator {
    parameters: PatternParameters,
    current_pattern: HarmonyPattern,
    pattern_position: usize,
    pattern_length: usize,
}

impl HarmonyGenerator {
    pub fn new(parameters: &PatternParameters) -> Result<Self> {
        Ok(Self {
            parameters: parameters.clone(),
            current_pattern: HarmonyPattern::default(),
            pattern_position: 0,
            pattern_length: 8, // 8 chord changes default
        })
    }

    fn build_chord(&self, root: u8, chord_type: ChordType, inversion: u8) -> Vec<u8> {
        let intervals = chord_type.intervals();
        let mut notes: Vec<u8> = intervals.iter().map(|&interval| root + interval).collect();

        // Apply inversion
        for _ in 0..inversion {
            if let Some(note) = notes.first().cloned() {
                notes.remove(0);
                notes.push(note + 12); // Move to next octave
            }
        }

        notes
    }
}

impl PatternGenerator for HarmonyGenerator {
    type Output = HarmonyPattern;

    fn next(&mut self) -> Self::Output {
        // Simple chord progression: I-vi-IV-V in C major
        let progression_roots = [60, 57, 65, 67]; // C, A, F, G
        let progression_types = [
            ChordType::Major,
            ChordType::Minor,
            ChordType::Major,
            ChordType::Major,
        ];

        let chord_index = self.pattern_position % progression_roots.len();
        let root = progression_roots[chord_index];
        let chord_type = progression_types[chord_index];

        self.current_pattern = HarmonyPattern {
            root,
            chord_type,
            inversion: 0,
            voicing: self.build_chord(root, chord_type, 0),
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