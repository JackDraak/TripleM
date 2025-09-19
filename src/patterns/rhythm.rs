use crate::patterns::{PatternGenerator, PatternParameters};
use crate::error::Result;

/// Rhythm pattern representation
#[derive(Debug, Clone)]
pub struct RhythmPattern {
    pub kick: bool,
    pub snare: bool,
    pub hihat: bool,
    pub intensity: f32,
}

impl Default for RhythmPattern {
    fn default() -> Self {
        Self {
            kick: false,
            snare: false,
            hihat: false,
            intensity: 0.0,
        }
    }
}

/// Rhythm pattern generator
pub struct RhythmGenerator {
    parameters: PatternParameters,
    current_pattern: RhythmPattern,
    pattern_position: usize,
    pattern_length: usize,
}

impl RhythmGenerator {
    pub fn new(parameters: &PatternParameters) -> Result<Self> {
        Ok(Self {
            parameters: parameters.clone(),
            current_pattern: RhythmPattern::default(),
            pattern_position: 0,
            pattern_length: 32, // 32 steps default
        })
    }
}

impl PatternGenerator for RhythmGenerator {
    type Output = RhythmPattern;

    fn next(&mut self) -> Self::Output {
        // Simple placeholder rhythm
        self.current_pattern = RhythmPattern {
            kick: self.pattern_position % 4 == 0,
            snare: self.pattern_position % 8 == 4,
            hihat: self.pattern_position % 2 == 1,
            intensity: self.parameters.intensity,
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