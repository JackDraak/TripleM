//! Unified melody generator with continuous scale morphing and adaptive complexity
//!
//! This module implements a melody generator that responds continuously to input values,
//! creating seamless transitions from simple ambient textures to complex melodic patterns
//! with scale morphing, adaptive note density, and intelligent phrase generation.

use crate::patterns::{PatternParameters, Scale, MelodyPattern};
use crate::audio::NaturalVariation;
use crate::error::Result;
use rand::Rng;
use std::f32::consts::TAU;
use std::collections::VecDeque;

/// Unified melody generator with continuous adaptation
#[derive(Debug, Clone)]
pub struct UnifiedMelodyGenerator {
    /// Current input value controlling all parameters (0.0-3.0)
    input_value: f32,

    /// Continuous scale morphing system
    scale_morpher: ContinuousScaleMorpher,

    /// Adaptive melodic characteristics
    note_density: f32,           // Notes per beat
    melodic_range: f32,          // Interval range (semitones)
    rhythmic_subdivision: f32,   // Note timing precision

    /// Phrase and structure control
    phrase_generator: AdaptivePhraseGenerator,
    motif_development: MotifDevelopment,

    /// Markov chain for intelligent note sequences
    adaptive_markov: AdaptiveMarkov,

    /// Pattern memory and evolution
    melody_memory: MelodyMemory,
    evolution_engine: MelodyEvolutionEngine,

    /// Natural variation integration
    variation: NaturalVariation,

    /// Current state
    current_note: u8,            // MIDI note number
    phrase_position: f32,        // Position within current phrase (0.0-1.0)
    last_note_time: f32,         // Time since last note

    sample_rate: f32,
}

/// Continuous scale morphing system
#[derive(Debug, Clone)]
pub struct ContinuousScaleMorpher {
    /// Available scale templates
    scale_templates: Vec<ScaleTemplate>,

    /// Current interpolation weights
    scale_weights: Vec<f32>,

    /// Morphing speed and smoothness
    morph_rate: f32,
    smooth_factor: f32,
}

/// Scale template with characteristics
#[derive(Debug, Clone)]
pub struct ScaleTemplate {
    /// Base scale
    scale: Scale,

    /// Input range where this scale is most prominent
    input_range: (f32, f32),

    /// Emotional characteristics
    brightness: f32,    // -1.0 = dark, 1.0 = bright
    tension: f32,       // 0.0 = consonant, 1.0 = dissonant
    complexity: f32,    // 0.0 = simple, 1.0 = complex

    /// Additional notes for morphing
    extended_intervals: Vec<u8>,
}

/// Adaptive phrase generation
#[derive(Debug, Clone)]
pub struct AdaptivePhraseGenerator {
    /// Phrase templates for different complexity levels
    phrase_templates: Vec<PhraseTemplate>,

    /// Current phrase state
    current_phrase: PhraseStructure,
    phrase_progress: f32,

    /// Phrase length adaptation
    adaptive_length: AdaptiveLength,

    /// Breathing and rest patterns
    rest_probability: f32,
    breath_timing: BreathTiming,
}

/// Phrase template defining structure
#[derive(Debug, Clone)]
pub struct PhraseTemplate {
    /// Length in beats
    length_beats: f32,

    /// Note density curve (0.0-1.0 over phrase)
    density_curve: Vec<f32>,

    /// Melodic direction preferences
    direction_tendency: Vec<MelodicDirection>,

    /// Rest points within phrase
    rest_points: Vec<f32>,

    /// Complexity level this template represents
    complexity_level: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum MelodicDirection {
    Up,
    Down,
    Static,
    Arch,      // Up then down
    Valley,    // Down then up
    Random,
}

/// Current phrase structure
#[derive(Debug, Clone)]
pub struct PhraseStructure {
    /// Total length in samples
    total_length: usize,

    /// Current position in samples
    current_position: usize,

    /// Note timing points
    note_timings: Vec<f32>,

    /// Melodic contour
    contour: Vec<MelodicDirection>,
}

/// Adaptive phrase length system
#[derive(Debug, Clone)]
pub struct AdaptiveLength {
    /// Base length multiplier
    base_length: f32,

    /// Length variation based on input
    length_curve: Vec<f32>,

    /// Minimum and maximum lengths
    length_range: (f32, f32),
}

/// Breathing and rest timing
#[derive(Debug, Clone)]
pub struct BreathTiming {
    /// Time between phrases
    inter_phrase_gap: f32,

    /// Natural breathing points within phrases
    breath_points: Vec<f32>,

    /// Humanization factor
    breath_variation: f32,
}

/// Motif development system
#[derive(Debug, Clone)]
pub struct MotifDevelopment {
    /// Current active motifs
    active_motifs: Vec<Motif>,

    /// Development techniques
    development_techniques: Vec<DevelopmentTechnique>,

    /// Motif memory for coherence
    motif_memory: VecDeque<Motif>,

    /// Evolution parameters
    variation_rate: f32,
    development_intensity: f32,
}

/// Musical motif
#[derive(Debug, Clone)]
pub struct Motif {
    /// Note sequence (intervals from root)
    intervals: Vec<i8>,

    /// Rhythm pattern
    rhythm: Vec<f32>,

    /// Motif strength/importance
    weight: f32,

    /// How often this motif appears
    frequency: f32,

    /// Transformation state
    transformations: Vec<Transformation>,
}

#[derive(Debug, Clone, Copy)]
pub enum DevelopmentTechnique {
    Sequence,     // Repeat at different pitches
    Inversion,    // Flip intervals
    Retrograde,   // Reverse order
    Augmentation, // Longer note values
    Diminution,   // Shorter note values
    Fragmentation, // Use partial motif
    Elaboration,  // Add decorative notes
}

#[derive(Debug, Clone)]
pub struct Transformation {
    technique: DevelopmentTechnique,
    intensity: f32,
    applied_at: f32, // Time when applied
}

/// Adaptive Markov chain for note sequences
#[derive(Debug, Clone)]
pub struct AdaptiveMarkov {
    /// Transition matrices for different contexts
    transition_matrices: Vec<TransitionMatrix>,

    /// Current context (based on input value)
    current_context: usize,

    /// Learning and adaptation parameters
    learning_rate: f32,
    adaptation_speed: f32,

    /// Memory of recent sequences
    sequence_memory: VecDeque<Vec<u8>>,
}

/// Transition matrix for Markov chains
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Input range this matrix applies to
    input_range: (f32, f32),

    /// Note-to-note transition probabilities
    transitions: Vec<Vec<f32>>,

    /// Rhythm transition probabilities
    rhythm_transitions: Vec<Vec<f32>>,

    /// Context information
    style_bias: StyleBias,
}

#[derive(Debug, Clone)]
pub struct StyleBias {
    /// Preference for stepwise motion vs leaps
    stepwise_tendency: f32,

    /// Preference for consonant intervals
    consonance_preference: f32,

    /// Rhythmic regularity preference
    rhythmic_regularity: f32,

    /// Scale adherence vs chromatic tendency
    scale_adherence: f32,
}

/// Melody memory for avoiding repetition
#[derive(Debug, Clone)]
pub struct MelodyMemory {
    /// Recent melodic fragments
    recent_fragments: VecDeque<MelodicFragment>,

    /// Memory depth
    memory_depth: usize,

    /// Repetition avoidance strength
    avoidance_strength: f32,
}

#[derive(Debug, Clone)]
pub struct MelodicFragment {
    /// Note sequence
    notes: Vec<u8>,

    /// Rhythm pattern
    rhythm: Vec<f32>,

    /// Input value when created
    input_context: f32,

    /// Timestamp
    created_at: f32,
}

/// Long-term melody evolution
#[derive(Debug, Clone)]
pub struct MelodyEvolutionEngine {
    /// Evolution trajectories
    evolution_curves: Vec<MelodicEvolutionCurve>,

    /// Current evolution phase
    evolution_phase: f32,

    /// Evolution speed
    evolution_speed: f32,

    /// Mutation parameters
    mutation_rate: f32,
    mutation_intensity: f32,
}

#[derive(Debug, Clone)]
pub struct MelodicEvolutionCurve {
    /// Input range this curve affects
    input_range: (f32, f32),

    /// How melodic complexity evolves
    complexity_evolution: Vec<f32>,

    /// How melodic range evolves
    range_evolution: Vec<f32>,

    /// How rhythmic complexity evolves
    rhythmic_evolution: Vec<f32>,

    /// Mutation points for significant changes
    mutation_points: Vec<f32>,
}

impl UnifiedMelodyGenerator {
    /// Create a new unified melody generator
    pub fn new(sample_rate: f32) -> Result<Self> {
        let variation = NaturalVariation::new(None);

        Ok(Self {
            input_value: 0.0,
            scale_morpher: ContinuousScaleMorpher::new(),
            note_density: 0.5,
            melodic_range: 12.0, // One octave
            rhythmic_subdivision: 0.25, // Quarter notes

            phrase_generator: AdaptivePhraseGenerator::new(),
            motif_development: MotifDevelopment::new(),
            adaptive_markov: AdaptiveMarkov::new(),

            melody_memory: MelodyMemory::new(),
            evolution_engine: MelodyEvolutionEngine::new(),

            variation,

            current_note: 60, // Middle C
            phrase_position: 0.0,
            last_note_time: 0.0,

            sample_rate,
        })
    }

    /// Set input value and update all parameters
    pub fn set_input_value(&mut self, input: f32) {
        self.input_value = input.clamp(0.0, 3.0);
        self.update_parameters_from_input();
    }

    /// Update all parameters based on input value
    fn update_parameters_from_input(&mut self) {
        // Update note density based on input (more notes = higher energy)
        self.note_density = self.calculate_note_density(self.input_value);

        // Update melodic range (wider intervals at higher inputs)
        self.melodic_range = self.calculate_melodic_range(self.input_value);

        // Update rhythmic subdivision (faster notes at higher inputs)
        self.rhythmic_subdivision = self.calculate_rhythmic_subdivision(self.input_value);

        // Update scale morphing
        self.scale_morpher.set_input_value(self.input_value);

        // Update phrase generation
        self.phrase_generator.set_input_value(self.input_value);

        // Update motif development
        self.motif_development.set_input_value(self.input_value);

        // Update Markov context
        self.adaptive_markov.set_input_value(self.input_value);

        // Update evolution engine
        self.evolution_engine.set_input_value(self.input_value);
    }

    /// Calculate note density from input value
    fn calculate_note_density(&self, input: f32) -> f32 {
        match input {
            // Ambient range (0.0-1.0): Very sparse to occasional notes
            i if i <= 1.0 => 0.1 + (i * 0.3), // 0.1 to 0.4 notes per beat
            // Active range (1.0-2.0): Moderate note density
            i if i <= 2.0 => {
                let t = i - 1.0;
                0.4 + (t * 0.4) // 0.4 to 0.8 notes per beat
            },
            // EDM range (2.0-3.0): High note density
            _ => {
                let t = input - 2.0;
                0.8 + (t * 0.6) // 0.8 to 1.4 notes per beat
            }
        }
    }

    /// Calculate melodic range from input value
    fn calculate_melodic_range(&self, input: f32) -> f32 {
        match input {
            // Ambient: Small intervals
            i if i <= 1.0 => 3.0 + (i * 4.0), // 3 to 7 semitones
            // Active: Moderate intervals
            i if i <= 2.0 => {
                let t = i - 1.0;
                7.0 + (t * 5.0) // 7 to 12 semitones
            },
            // EDM: Wide intervals
            _ => {
                let t = input - 2.0;
                12.0 + (t * 8.0) // 12 to 20 semitones
            }
        }
    }

    /// Calculate rhythmic subdivision from input value
    fn calculate_rhythmic_subdivision(&self, input: f32) -> f32 {
        match input {
            // Ambient: Slow, whole/half notes
            i if i <= 1.0 => 1.0 - (i * 0.6), // 1.0 to 0.4 (slower = larger values)
            // Active: Quarter to eighth notes
            i if i <= 2.0 => {
                let t = i - 1.0;
                0.4 - (t * 0.15) // 0.4 to 0.25
            },
            // EDM: Fast sixteenth notes and beyond
            _ => {
                let t = input - 2.0;
                0.25 - (t * 0.125) // 0.25 to 0.125
            }
        }
    }

    /// Generate the next melody sample
    pub fn process_sample(&mut self, beat_phase: f32) -> MelodyPattern {
        // Update natural variation
        self.variation.update();

        // Update evolution
        self.evolution_engine.update();

        // Update phrase position
        self.phrase_generator.update(beat_phase);

        // Determine if we should generate a new note
        let should_generate_note = self.should_generate_note(beat_phase);

        let mut melody = MelodyPattern {
            note: self.current_note,
            velocity: 64, // Default MIDI velocity
            duration: self.rhythmic_subdivision,
            is_rest: true, // Default to rest
        };

        if should_generate_note {
            // Generate new note using adaptive systems
            let new_note = self.generate_new_note();
            let velocity = self.calculate_velocity();

            melody.note = new_note;
            melody.velocity = (velocity * 127.0) as u8; // Convert to MIDI velocity
            melody.duration = self.rhythmic_subdivision;
            melody.is_rest = false;

            // Update current state
            self.current_note = new_note;
            self.last_note_time = 0.0;

            // Store in memory
            self.melody_memory.add_note(new_note, self.input_value);
        }

        // Update timing
        self.last_note_time += 1.0 / self.sample_rate;

        melody
    }

    /// Determine if a new note should be generated
    fn should_generate_note(&self, beat_phase: f32) -> bool {
        // Calculate time since last note in beats
        let beats_since_last = self.last_note_time * (self.calculate_current_bpm() / 60.0);

        // Check if enough time has passed based on note density and rhythmic subdivision
        let min_interval = self.rhythmic_subdivision;
        let density_probability = self.note_density;

        if beats_since_last >= min_interval {
            // Use phrase template to determine if we should play a note here
            let template_probability = self.phrase_generator.get_note_probability(beat_phase);
            let final_probability = density_probability * template_probability;

            rand::random::<f32>() < final_probability
        } else {
            false
        }
    }

    /// Generate a new note using all adaptive systems
    fn generate_new_note(&mut self) -> u8 {
        // Get current scale from morpher
        let current_scale = self.scale_morpher.get_current_scale();

        // Use Markov chain to get next note suggestion
        let markov_suggestion = self.adaptive_markov.get_next_note(self.current_note);

        // Apply motif development
        let motif_suggestion = self.motif_development.get_next_note(self.current_note);

        // Combine suggestions with scale constraints
        let scale_constrained = self.apply_scale_constraints(
            vec![markov_suggestion, motif_suggestion],
            &current_scale
        );

        // Apply range constraints
        let range_constrained = self.apply_range_constraints(scale_constrained);

        // Check against memory to avoid repetition
        let final_note = self.melody_memory.avoid_repetition(range_constrained);

        final_note
    }

    /// Apply scale constraints to note suggestions
    fn apply_scale_constraints(&self, suggestions: Vec<u8>, scale: &Scale) -> u8 {
        let scale_intervals = scale.intervals();
        let root_note = scale.root_note();

        // Find the best suggestion that fits the scale
        for &suggestion in &suggestions {
            let interval = (suggestion as i16 - root_note as i16) % 12;
            let interval = if interval < 0 { interval + 12 } else { interval } as u8;

            if scale_intervals.contains(&interval) {
                return suggestion;
            }
        }

        // If no suggestion fits exactly, find closest scale note
        let target_suggestion = suggestions[0];
        let mut closest_note = root_note;
        let mut closest_distance = i16::MAX;

        for octave in 3..7 { // Search reasonable octave range
            for &interval in scale_intervals {
                let scale_note = octave * 12 + interval;
                let distance = (scale_note as i16 - target_suggestion as i16).abs();

                if distance < closest_distance {
                    closest_distance = distance;
                    closest_note = scale_note;
                }
            }
        }

        closest_note
    }

    /// Apply melodic range constraints
    fn apply_range_constraints(&self, note: u8) -> u8 {
        let max_distance = self.melodic_range as i16;
        let current_note = self.current_note as i16;

        let distance = (note as i16 - current_note).abs();

        if distance <= max_distance {
            note
        } else {
            // Clamp to range
            if note as i16 > current_note {
                (current_note + max_distance).min(127) as u8
            } else {
                (current_note - max_distance).max(21) as u8
            }
        }
    }

    /// Calculate velocity based on input and phrase position
    fn calculate_velocity(&self) -> f32 {
        let base_velocity = 0.6 + (self.input_value / 3.0) * 0.3; // 0.6 to 0.9

        // Apply phrase dynamics
        let phrase_dynamics = self.phrase_generator.get_dynamics_at_position(self.phrase_position);

        // Apply natural variation
        let velocity_variation = self.variation.get_amplitude_variation();

        let final_velocity = base_velocity * phrase_dynamics * (1.0 + velocity_variation * 0.2);

        final_velocity.clamp(0.1, 1.0)
    }

    /// Calculate timbre morphing parameter
    fn calculate_timbre_morph(&self) -> f32 {
        // Morph from organic (0.0) to synthetic (1.0) based on input
        let base_morph = self.input_value / 3.0;

        // Add some evolution over time
        let evolution_factor = self.evolution_engine.get_timbre_evolution();

        // Add natural variation
        let variation_factor = self.variation.get_timbre_variation() * 0.1;

        (base_morph + evolution_factor + variation_factor).clamp(0.0, 1.0)
    }

    /// Calculate current BPM (this should be synchronized with rhythm generator)
    fn calculate_current_bpm(&self) -> f32 {
        match self.input_value {
            i if i <= 1.0 => 48.0 + (i * 32.0),
            i if i <= 2.0 => {
                let t = i - 1.0;
                80.0 + (t * 50.0)
            },
            _ => {
                let t = self.input_value - 2.0;
                130.0 + (t * 50.0)
            }
        }
    }

    /// Get current input value
    pub fn input_value(&self) -> f32 {
        self.input_value
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.phrase_position = 0.0;
        self.last_note_time = 0.0;
        self.current_note = 60;
        self.melody_memory.clear();
        self.evolution_engine.reset();
    }
}

// Implementation stubs for supporting structures
impl ContinuousScaleMorpher {
    fn new() -> Self {
        Self {
            scale_templates: Self::create_scale_templates(),
            scale_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            morph_rate: 0.01,
            smooth_factor: 0.95,
        }
    }

    fn create_scale_templates() -> Vec<ScaleTemplate> {
        vec![
            // Ambient scales
            ScaleTemplate {
                scale: Scale::Pentatonic,
                input_range: (0.0, 1.2),
                brightness: -0.3,
                tension: 0.1,
                complexity: 0.2,
                extended_intervals: vec![],
            },
            // Gentle scales
            ScaleTemplate {
                scale: Scale::CMajor,
                input_range: (0.8, 1.8),
                brightness: 0.1,
                tension: 0.2,
                complexity: 0.4,
                extended_intervals: vec![],
            },
            // Active scales
            ScaleTemplate {
                scale: Scale::DMinor,
                input_range: (1.5, 2.5),
                brightness: 0.0,
                tension: 0.4,
                complexity: 0.6,
                extended_intervals: vec![],
            },
            // EDM scales
            ScaleTemplate {
                scale: Scale::Blues,
                input_range: (2.0, 3.0),
                brightness: 0.4,
                tension: 0.6,
                complexity: 0.8,
                extended_intervals: vec![1, 6, 10], // Add chromatic notes
            },
        ]
    }

    fn set_input_value(&mut self, input: f32) {
        // Update scale weights based on input value
        for (i, template) in self.scale_templates.iter().enumerate() {
            let distance = if input >= template.input_range.0 && input <= template.input_range.1 {
                0.0 // Inside range
            } else {
                let dist_to_start = (input - template.input_range.0).abs();
                let dist_to_end = (input - template.input_range.1).abs();
                dist_to_start.min(dist_to_end)
            };

            // Convert distance to weight (closer = higher weight)
            let target_weight = if distance == 0.0 {
                1.0
            } else {
                (1.0 / (1.0 + distance)).powf(2.0)
            };

            // Smooth interpolation
            self.scale_weights[i] = self.scale_weights[i] * self.smooth_factor +
                                  target_weight * (1.0 - self.smooth_factor);
        }

        // Normalize weights
        let total_weight: f32 = self.scale_weights.iter().sum();
        if total_weight > 0.0 {
            for weight in &mut self.scale_weights {
                *weight /= total_weight;
            }
        }
    }

    fn get_current_scale(&self) -> Scale {
        // Find the scale with highest weight
        let mut max_weight = 0.0;
        let mut best_scale = Scale::CMajor;

        for (i, &weight) in self.scale_weights.iter().enumerate() {
            if weight > max_weight {
                max_weight = weight;
                best_scale = self.scale_templates[i].scale;
            }
        }

        best_scale
    }
}

impl AdaptivePhraseGenerator {
    fn new() -> Self {
        Self {
            phrase_templates: Self::create_phrase_templates(),
            current_phrase: PhraseStructure::default(),
            phrase_progress: 0.0,
            adaptive_length: AdaptiveLength::new(),
            rest_probability: 0.2,
            breath_timing: BreathTiming::new(),
        }
    }

    fn create_phrase_templates() -> Vec<PhraseTemplate> {
        vec![
            // Ambient phrase
            PhraseTemplate {
                length_beats: 8.0,
                density_curve: vec![0.1, 0.15, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05],
                direction_tendency: vec![MelodicDirection::Static, MelodicDirection::Valley],
                rest_points: vec![0.5, 0.75],
                complexity_level: 0.2,
            },
            // Active phrase
            PhraseTemplate {
                length_beats: 4.0,
                density_curve: vec![0.8, 0.6, 0.7, 0.9],
                direction_tendency: vec![MelodicDirection::Arch, MelodicDirection::Up],
                rest_points: vec![0.25],
                complexity_level: 0.7,
            },
        ]
    }

    fn set_input_value(&mut self, _input: f32) {
        // Update phrase generation parameters based on input
        // Implementation...
    }

    fn update(&mut self, _beat_phase: f32) {
        // Update phrase state
        // Implementation...
    }

    fn get_note_probability(&self, _beat_phase: f32) -> f32 {
        0.8 // Placeholder
    }

    fn get_dynamics_at_position(&self, _position: f32) -> f32 {
        1.0 // Placeholder
    }
}

impl PhraseStructure {
    fn default() -> Self {
        Self {
            total_length: 0,
            current_position: 0,
            note_timings: vec![],
            contour: vec![],
        }
    }
}

impl AdaptiveLength {
    fn new() -> Self {
        Self {
            base_length: 4.0,
            length_curve: vec![8.0, 6.0, 4.0, 2.0], // Longer phrases at lower inputs
            length_range: (2.0, 16.0),
        }
    }
}

impl BreathTiming {
    fn new() -> Self {
        Self {
            inter_phrase_gap: 1.0,
            breath_points: vec![0.5],
            breath_variation: 0.2,
        }
    }
}

impl MotifDevelopment {
    fn new() -> Self {
        Self {
            active_motifs: vec![],
            development_techniques: vec![
                DevelopmentTechnique::Sequence,
                DevelopmentTechnique::Inversion,
                DevelopmentTechnique::Fragmentation,
            ],
            motif_memory: VecDeque::new(),
            variation_rate: 0.1,
            development_intensity: 0.5,
        }
    }

    fn set_input_value(&mut self, _input: f32) {
        // Update motif development based on input
        // Implementation...
    }

    fn get_next_note(&mut self, _current_note: u8) -> u8 {
        60 // Placeholder
    }
}

impl AdaptiveMarkov {
    fn new() -> Self {
        Self {
            transition_matrices: Self::create_transition_matrices(),
            current_context: 0,
            learning_rate: 0.01,
            adaptation_speed: 0.1,
            sequence_memory: VecDeque::new(),
        }
    }

    fn create_transition_matrices() -> Vec<TransitionMatrix> {
        vec![
            // Ambient matrix
            TransitionMatrix {
                input_range: (0.0, 1.0),
                transitions: vec![vec![0.1; 128]; 128], // Simple placeholder
                rhythm_transitions: vec![vec![0.1; 8]; 8],
                style_bias: StyleBias {
                    stepwise_tendency: 0.8,
                    consonance_preference: 0.9,
                    rhythmic_regularity: 0.6,
                    scale_adherence: 0.9,
                },
            },
            // EDM matrix
            TransitionMatrix {
                input_range: (2.0, 3.0),
                transitions: vec![vec![0.1; 128]; 128],
                rhythm_transitions: vec![vec![0.1; 8]; 8],
                style_bias: StyleBias {
                    stepwise_tendency: 0.4,
                    consonance_preference: 0.5,
                    rhythmic_regularity: 0.8,
                    scale_adherence: 0.6,
                },
            },
        ]
    }

    fn set_input_value(&mut self, input: f32) {
        // Find appropriate context based on input
        for (i, matrix) in self.transition_matrices.iter().enumerate() {
            if input >= matrix.input_range.0 && input <= matrix.input_range.1 {
                self.current_context = i;
                break;
            }
        }
    }

    fn get_next_note(&mut self, current_note: u8) -> u8 {
        // Simple random walk for now
        let step = if rand::random::<bool>() { 1 } else { -1 };
        (current_note as i16 + step).clamp(21, 108) as u8
    }
}

impl MelodyMemory {
    fn new() -> Self {
        Self {
            recent_fragments: VecDeque::new(),
            memory_depth: 16,
            avoidance_strength: 0.7,
        }
    }

    fn add_note(&mut self, note: u8, input_value: f32) {
        // Add to memory (simplified)
        let fragment = MelodicFragment {
            notes: vec![note],
            rhythm: vec![1.0],
            input_context: input_value,
            created_at: 0.0, // Should be actual time
        };

        self.recent_fragments.push_back(fragment);

        if self.recent_fragments.len() > self.memory_depth {
            self.recent_fragments.pop_front();
        }
    }

    fn avoid_repetition(&self, note: u8) -> u8 {
        // Simple repetition avoidance
        if let Some(last_fragment) = self.recent_fragments.back() {
            if let Some(&last_note) = last_fragment.notes.last() {
                if last_note == note && rand::random::<f32>() < self.avoidance_strength {
                    // Slightly modify the note
                    let modification = if rand::random::<bool>() { 1 } else { -1 };
                    return (note as i16 + modification).clamp(21, 108) as u8;
                }
            }
        }
        note
    }

    fn clear(&mut self) {
        self.recent_fragments.clear();
    }
}

impl MelodyEvolutionEngine {
    fn new() -> Self {
        Self {
            evolution_curves: vec![],
            evolution_phase: 0.0,
            evolution_speed: 0.01,
            mutation_rate: 0.05,
            mutation_intensity: 0.3,
        }
    }

    fn set_input_value(&mut self, _input: f32) {
        // Update evolution parameters
        // Implementation...
    }

    fn update(&mut self) {
        self.evolution_phase += self.evolution_speed;
        if self.evolution_phase >= 1.0 {
            self.evolution_phase -= 1.0;
        }
    }

    fn get_timbre_evolution(&self) -> f32 {
        (self.evolution_phase * TAU).sin() * 0.1
    }

    fn reset(&mut self) {
        self.evolution_phase = 0.0;
    }
}