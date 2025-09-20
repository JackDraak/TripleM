//! Unified harmony generator with adaptive chord complexity and bass patterns
//!
//! This module implements a harmony generator that responds continuously to input values,
//! creating seamless transitions from simple ambient drones to complex polyphonic harmonies
//! with intelligent voice leading, bass patterns, and harmonic progression.

use crate::patterns::{PatternGenerator, PatternParameters, Scale};
use crate::patterns::harmony::{HarmonyPattern, ChordType as ExternalChordType};
use crate::audio::NaturalVariation;
use crate::error::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::TAU;
use std::collections::VecDeque;

/// Unified harmony generator with continuous adaptation
#[derive(Debug, Clone)]
pub struct UnifiedHarmonyGenerator {
    /// Current input value controlling all parameters (0.0-1.0)
    input_value: f32,

    /// Chord progression system
    chord_progression: AdaptiveChordProgression,

    /// Adaptive harmonic characteristics
    chord_complexity: f32,       // Simple triads -> complex extended chords
    voicing_spread: f32,         // Tight -> wide voicings
    harmonic_rhythm: f32,        // How often chords change

    /// Bass behavior
    bass_generator: AdaptiveBassGenerator,

    /// Harmonic movement and tension
    tension_resolution: TensionResolutionEngine,
    modulation_system: ModulationSystem,

    /// Voice leading and texture
    voice_leading: VoiceLeadingEngine,
    texture_manager: HarmonicTextureManager,

    /// Pattern memory and evolution
    harmony_memory: HarmonyMemory,
    evolution_engine: HarmonyEvolutionEngine,

    /// Natural variation integration
    variation: NaturalVariation,

    /// Current state
    current_chord: ChordStructure,
    current_bass_note: u8,
    progression_position: usize,
    time_since_change: f32,

    sample_rate: f32,
    rng: StdRng,
}

/// Adaptive chord progression system
#[derive(Debug, Clone)]
pub struct AdaptiveChordProgression {
    /// Available progression templates
    progression_templates: Vec<ProgressionTemplate>,

    /// Current active progression
    current_progression: Vec<ChordFunction>,

    /// Progression weights based on input
    template_weights: Vec<f32>,

    /// Progression speed adaptation
    progression_speed: f32,

    /// Chord substitution system
    substitution_system: ChordSubstitutionSystem,
}

/// Progression template for different styles
#[derive(Debug, Clone)]
pub struct ProgressionTemplate {
    /// Name of the progression
    name: String,

    /// Chord functions in the progression
    functions: Vec<ChordFunction>,

    /// Input range where this template is most appropriate
    input_range: (f32, f32),

    /// Style characteristics
    style_traits: ProgressionTraits,

    /// Loop characteristics
    loop_length: usize,
    loop_point: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ChordFunction {
    Tonic,         // I
    Supertonic,    // ii
    Mediant,       // iii
    Subdominant,   // IV
    Dominant,      // V
    Submediant,    // vi
    Leading,       // vii
    // Extended functions
    SecondaryDominant(u8), // V/x
    NeapolitanSixth,       // bII6
    AugmentedSixth,        // +6
    Diminished,            // dim
}

/// Style traits for progressions
#[derive(Debug, Clone)]
pub struct ProgressionTraits {
    /// Tendency toward modal vs tonal harmony
    modal_tendency: f32,

    /// Amount of chromaticism
    chromatic_level: f32,

    /// Tension vs resolution preference
    tension_preference: f32,

    /// Harmonic rhythm (fast vs slow changes)
    rhythm_density: f32,
}

/// Chord substitution system
#[derive(Debug, Clone)]
pub struct ChordSubstitutionSystem {
    /// Available substitutions by function
    substitution_map: Vec<(ChordFunction, Vec<ChordFunction>)>,

    /// Substitution probability based on input
    substitution_probability: f32,

    /// Complexity of substitutions
    substitution_complexity: f32,
}

/// Individual chord structure
#[derive(Debug, Clone)]
pub struct ChordStructure {
    /// Root note (MIDI number)
    root: u8,

    /// Chord type
    chord_type: ChordType,

    /// Voicing (specific notes)
    voicing: Vec<u8>,

    /// Inversion
    inversion: u8,

    /// Extensions and alterations
    extensions: Vec<ChordExtension>,

    /// Voice assignment (which synthesizer voice plays which note)
    voice_assignment: Vec<VoiceAssignment>,
}

#[derive(Debug, Clone, Copy)]
pub enum ChordType {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    MinorMajor7,
    Diminished7,
    HalfDiminished7,
    // Extended chords
    Major9,
    Minor9,
    Dominant9,
    Major11,
    Minor11,
    Dominant11,
    Major13,
    Minor13,
    Dominant13,
    // Altered chords
    Altered,
    Suspended2,
    Suspended4,
}

#[derive(Debug, Clone, Copy)]
pub enum ChordExtension {
    Add9,
    Add11,
    Add13,
    Flat5,
    Sharp5,
    Flat9,
    Sharp9,
    Sharp11,
    Flat13,
}

#[derive(Debug, Clone)]
pub struct VoiceAssignment {
    /// MIDI note number
    note: u8,

    /// Which voice/synthesizer plays this note
    voice_index: usize,

    /// Voice characteristics
    voice_character: VoiceCharacter,
}

#[derive(Debug, Clone, Copy)]
pub enum VoiceCharacter {
    Bass,        // Bass voice
    Root,        // Root/foundation
    Harmony,     // Inner harmony
    Melody,      // Top voice/melody
    Texture,     // Additional texture
}

/// Adaptive bass generator
#[derive(Debug, Clone)]
pub struct AdaptiveBassGenerator {
    /// Bass pattern templates
    pattern_templates: Vec<BassPatternTemplate>,

    /// Current bass pattern
    current_pattern: BassPattern,

    /// Bass presence (absent -> prominent)
    bass_presence: f32,

    /// Pattern complexity
    pattern_complexity: f32,

    /// Walking bass system
    walking_bass: WalkingBassEngine,

    /// Rhythmic bass patterns
    rhythmic_bass: RhythmicBassEngine,
}

/// Bass pattern template
#[derive(Debug, Clone)]
pub struct BassPatternTemplate {
    /// Pattern name
    name: String,

    /// Note pattern (intervals from chord root)
    note_pattern: Vec<i8>,

    /// Rhythm pattern (beat positions)
    rhythm_pattern: Vec<f32>,

    /// Input range where this pattern works best
    input_range: (f32, f32),

    /// Pattern characteristics
    style_traits: BassStyleTraits,
}

#[derive(Debug, Clone)]
pub struct BassStyleTraits {
    /// How much the bass moves around
    movement_level: f32,

    /// Rhythmic activity
    rhythmic_activity: f32,

    /// Harmonic function (root movement vs independent)
    harmonic_function: f32,
}

/// Current bass pattern state
#[derive(Debug, Clone)]
pub struct BassPattern {
    /// Current notes in the pattern
    notes: Vec<u8>,

    /// Timing for each note
    timings: Vec<f32>,

    /// Current position in pattern
    position: usize,

    /// Pattern length
    length: usize,
}

/// Walking bass line generator
#[derive(Debug, Clone)]
pub struct WalkingBassEngine {
    /// Current walking pattern
    walking_pattern: Vec<u8>,

    /// Step patterns (chromatic, diatonic, etc.)
    step_patterns: Vec<StepPattern>,

    /// Target notes (chord tones to hit)
    target_notes: Vec<u8>,

    /// Walking speed
    walking_speed: f32,
}

#[derive(Debug, Clone)]
pub struct StepPattern {
    /// Type of step motion
    step_type: StepType,

    /// Probability weight
    weight: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum StepType {
    Chromatic,    // Half steps
    Diatonic,     // Scale steps
    ChordTone,    // Chord tone jumps
    Approach,     // Approach notes
}

/// Rhythmic bass pattern generator
#[derive(Debug, Clone)]
pub struct RhythmicBassEngine {
    /// Rhythmic patterns
    rhythmic_patterns: Vec<RhythmicPattern>,

    /// Current pattern
    current_pattern: usize,

    /// Syncopation level
    syncopation_level: f32,

    /// Groove integration
    groove_lock: f32, // How locked to main rhythm
}

#[derive(Debug, Clone)]
pub struct RhythmicPattern {
    /// Beat pattern
    beats: Vec<bool>,

    /// Accent pattern
    accents: Vec<f32>,

    /// Note choices for each beat
    note_choices: Vec<NoteChoice>,
}

#[derive(Debug, Clone, Copy)]
pub enum NoteChoice {
    Root,
    Fifth,
    Octave,
    ChordTone,
    Passing,
}

/// Tension and resolution engine
#[derive(Debug, Clone)]
pub struct TensionResolutionEngine {
    /// Current tension level (0.0 to 1.0)
    current_tension: f32,

    /// Tension curve over time
    tension_curve: TensionCurve,

    /// Resolution patterns
    resolution_patterns: Vec<ResolutionPattern>,

    /// Tension sources
    tension_sources: Vec<TensionSource>,
}

#[derive(Debug, Clone)]
pub struct TensionCurve {
    /// Control points for tension over time
    control_points: Vec<(f32, f32)>, // (time, tension)

    /// Curve interpolation type
    interpolation_type: CurveInterpolation,
}

#[derive(Debug, Clone, Copy)]
pub enum CurveInterpolation {
    Linear,
    Smooth,
    Exponential,
}

#[derive(Debug, Clone)]
pub struct ResolutionPattern {
    /// Chord sequence for resolution
    chord_sequence: Vec<ChordFunction>,

    /// Tension release amount
    release_amount: f32,

    /// Time to resolve
    resolution_time: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum TensionSource {
    Dissonance,      // Dissonant intervals
    Chromaticism,    // Chromatic movement
    Suspension,      // Suspended notes
    Extension,       // Extended harmony
    Modulation,      // Key changes
}

/// Modulation and key change system
#[derive(Debug, Clone)]
pub struct ModulationSystem {
    /// Current key/scale
    current_scale: Scale,

    /// Modulation rate based on input
    modulation_rate: f32,

    /// Available modulation targets
    modulation_targets: Vec<ModulationTarget>,

    /// Pivot chord system
    pivot_system: PivotChordSystem,
}

#[derive(Debug, Clone)]
pub struct ModulationTarget {
    /// Target scale
    target_scale: Scale,

    /// Relationship to current key
    relationship: KeyRelationship,

    /// Transition difficulty
    difficulty: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum KeyRelationship {
    RelativeMajor,
    RelativeMinor,
    Dominant,
    Subdominant,
    Circle5th,
    Chromatic,
}

/// Pivot chord modulation system
#[derive(Debug, Clone)]
pub struct PivotChordSystem {
    /// Available pivot chords between keys
    pivot_chords: Vec<PivotChord>,

    /// Current modulation state
    modulation_state: ModulationState,
}

#[derive(Debug, Clone)]
pub struct PivotChord {
    /// Function in original key
    original_function: ChordFunction,

    /// Function in target key
    target_function: ChordFunction,

    /// Chord structure
    chord: ChordStructure,
}

#[derive(Debug, Clone, Copy)]
pub enum ModulationState {
    Stable,      // In key
    Pivoting,    // Using pivot chord
    Modulating,  // In transition
    Establishing, // Establishing new key
}

/// Voice leading engine
#[derive(Debug, Clone)]
pub struct VoiceLeadingEngine {
    /// Voice leading rules
    voice_leading_rules: Vec<VoiceLeadingRule>,

    /// Current voice positions
    voice_positions: Vec<u8>,

    /// Voice motion preferences
    motion_preferences: MotionPreferences,

    /// Parallel motion avoidance
    parallel_avoidance: ParallelAvoidance,
}

#[derive(Debug, Clone)]
pub struct VoiceLeadingRule {
    /// Rule type
    rule_type: VoiceLeadingRuleType,

    /// Rule strength (0.0 to 1.0)
    strength: f32,

    /// When this rule applies
    context: VoiceLeadingContext,
}

#[derive(Debug, Clone, Copy)]
pub enum VoiceLeadingRuleType {
    SmallestMotion,     // Prefer smallest voice movement
    CommonTones,        // Keep common tones
    StepwiseMotion,     // Prefer stepwise motion
    AvoidParallels,     // Avoid parallel 5ths/octaves
    ResolveTendency,    // Resolve tendency tones
}

#[derive(Debug, Clone)]
pub struct VoiceLeadingContext {
    /// Chord progression context
    progression_context: ProgressionContext,

    /// Voice range considerations
    voice_ranges: Vec<(u8, u8)>, // (min, max) for each voice
}

#[derive(Debug, Clone, Copy)]
pub enum ProgressionContext {
    Any,
    TonicToDominant,
    DominantToTonic,
    SecondaryDominant,
    Modulation,
}

#[derive(Debug, Clone)]
pub struct MotionPreferences {
    /// Preference for contrary motion
    contrary_motion: f32,

    /// Preference for oblique motion
    oblique_motion: f32,

    /// Preference for similar motion
    similar_motion: f32,

    /// Preference for parallel motion
    parallel_motion: f32,
}

#[derive(Debug, Clone)]
pub struct ParallelAvoidance {
    /// Avoid parallel 5ths
    parallel_fifths: f32,

    /// Avoid parallel octaves
    parallel_octaves: f32,

    /// Avoid parallel unisons
    parallel_unisons: f32,
}

/// Harmonic texture manager
#[derive(Debug, Clone)]
pub struct HarmonicTextureManager {
    /// Current number of voices active
    active_voices: usize,

    /// Maximum voices based on input
    max_voices: usize,

    /// Voice density curve
    density_curve: DensityCurve,

    /// Texture types
    texture_types: Vec<TextureType>,

    /// Current texture
    current_texture: TextureType,
}

#[derive(Debug, Clone)]
pub struct DensityCurve {
    /// How texture density changes with input
    density_points: Vec<(f32, f32)>, // (input, density)

    /// Smoothing factor
    smoothing: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum TextureType {
    Monophonic,    // Single voice
    Homophonic,    // Melody with accompaniment
    Polyphonic,    // Multiple independent voices
    Chordal,       // Block chords
    Arpeggiated,   // Broken chords
    Sustained,     // Long sustained notes
    Rhythmic,      // Rhythmic chords
}

/// Harmony memory for avoiding repetition
#[derive(Debug, Clone)]
pub struct HarmonyMemory {
    /// Recent chord progressions
    recent_progressions: VecDeque<Vec<ChordFunction>>,

    /// Recent chord structures
    recent_chords: VecDeque<ChordStructure>,

    /// Memory depth
    memory_depth: usize,

    /// Repetition avoidance strength
    avoidance_strength: f32,
}

/// Long-term harmony evolution
#[derive(Debug, Clone)]
pub struct HarmonyEvolutionEngine {
    /// Evolution trajectories
    evolution_curves: Vec<HarmonyEvolutionCurve>,

    /// Current evolution phase
    evolution_phase: f32,

    /// Evolution speed
    evolution_speed: f32,

    /// Harmonic complexity evolution
    complexity_evolution: ComplexityEvolution,
}

#[derive(Debug, Clone)]
pub struct HarmonyEvolutionCurve {
    /// Input range this curve affects
    input_range: (f32, f32),

    /// How harmonic complexity evolves
    complexity_curve: Vec<f32>,

    /// How voice count evolves
    voice_count_curve: Vec<f32>,

    /// How harmonic rhythm evolves
    rhythm_curve: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ComplexityEvolution {
    /// Chord complexity over time
    chord_complexity: Vec<f32>,

    /// Extension usage over time
    extension_usage: Vec<f32>,

    /// Substitution rate over time
    substitution_rate: Vec<f32>,
}

impl UnifiedHarmonyGenerator {
    /// Create a new unified harmony generator
    pub fn new(sample_rate: f32) -> Result<Self> {
        let variation = NaturalVariation::new(None);

        Ok(Self {
            input_value: 0.0,
            chord_progression: AdaptiveChordProgression::new(),
            chord_complexity: 0.3,
            voicing_spread: 12.0, // One octave
            harmonic_rhythm: 4.0, // Change every 4 beats

            bass_generator: AdaptiveBassGenerator::new(),
            tension_resolution: TensionResolutionEngine::new(),
            modulation_system: ModulationSystem::new(),

            voice_leading: VoiceLeadingEngine::new(),
            texture_manager: HarmonicTextureManager::new(),

            harmony_memory: HarmonyMemory::new(),
            evolution_engine: HarmonyEvolutionEngine::new(),

            variation,

            current_chord: ChordStructure::default(),
            current_bass_note: 36, // Low C
            progression_position: 0,
            time_since_change: 0.0,

            sample_rate,
            rng: StdRng::from_entropy(),
        })
    }

    /// Set input value and update all parameters
    pub fn set_input_value(&mut self, input: f32) {
        self.input_value = input.clamp(0.0, 1.0);
        self.update_parameters_from_input();
    }

    /// Update all parameters based on input value
    fn update_parameters_from_input(&mut self) {
        // Update chord complexity (simple triads -> complex extended chords)
        self.chord_complexity = self.calculate_chord_complexity(self.input_value);

        // Update voicing spread (tight -> wide voicings)
        self.voicing_spread = self.calculate_voicing_spread(self.input_value);

        // Update harmonic rhythm (how often chords change)
        self.harmonic_rhythm = self.calculate_harmonic_rhythm(self.input_value);

        // Update chord progression system
        self.chord_progression.set_input_value(self.input_value);

        // Update bass generator
        self.bass_generator.set_input_value(self.input_value);

        // Update texture manager
        self.texture_manager.set_input_value(self.input_value);

        // Update evolution systems
        self.evolution_engine.set_input_value(self.input_value);
        self.tension_resolution.set_input_value(self.input_value);
        self.modulation_system.set_input_value(self.input_value);
    }

    /// Calculate chord complexity from input value
    fn calculate_chord_complexity(&self, input: f32) -> f32 {
        match input {
            // Ambient: Simple triads and drones
            i if i <= 1.0 => 0.1 + (i * 0.2), // 0.1 to 0.3
            // Active: Seventh chords and some extensions
            i if i <= 2.0 => {
                let t = i - 1.0;
                0.3 + (t * 0.4) // 0.3 to 0.7
            },
            // EDM: Complex extended and altered chords
            _ => {
                let t = input - 2.0;
                0.7 + (t * 0.3) // 0.7 to 1.0
            }
        }
    }

    /// Calculate voicing spread from input value
    fn calculate_voicing_spread(&self, input: f32) -> f32 {
        match input {
            // Ambient: Tight voicings
            i if i <= 1.0 => 3.0 + (i * 5.0), // 3 to 8 semitones
            // Active: Moderate spread
            i if i <= 2.0 => {
                let t = i - 1.0;
                8.0 + (t * 8.0) // 8 to 16 semitones
            },
            // EDM: Wide voicings
            _ => {
                let t = input - 2.0;
                16.0 + (t * 12.0) // 16 to 28 semitones
            }
        }
    }

    /// Calculate harmonic rhythm from input value
    fn calculate_harmonic_rhythm(&self, input: f32) -> f32 {
        match input {
            // Ambient: Very slow harmonic rhythm
            i if i <= 1.0 => 8.0 - (i * 4.0), // 8 to 4 beats per change
            // Active: Moderate harmonic rhythm
            i if i <= 2.0 => {
                let t = i - 1.0;
                4.0 - (t * 2.0) // 4 to 2 beats per change
            },
            // EDM: Fast harmonic rhythm
            _ => {
                let t = input - 2.0;
                2.0 - (t * 1.0) // 2 to 1 beat per change
            }
        }
    }

    /// Generate the next harmony sample
    pub fn process_sample(&mut self, beat_phase: f32) -> HarmonyPattern {
        // Update natural variation
        self.variation.update();

        // Update evolution systems
        self.evolution_engine.update();

        // Update time tracking
        self.time_since_change += 1.0 / self.sample_rate;

        // Determine if we should change harmony
        let should_change_harmony = self.should_change_harmony(beat_phase);

        if should_change_harmony {
            self.generate_new_harmony();
            self.time_since_change = 0.0;
        }

        // Create harmony pattern with existing structure
        HarmonyPattern {
            root: self.current_chord.root,
            chord_type: self.convert_chord_type(self.current_chord.chord_type),
            inversion: self.current_chord.inversion,
            voicing: self.current_chord.voicing.clone(),
        }
    }

    /// Determine if harmony should change
    fn should_change_harmony(&self, beat_phase: f32) -> bool {
        let beats_elapsed = self.time_since_change * (self.calculate_current_bpm() / 60.0);

        // Check if enough time has passed based on harmonic rhythm
        if beats_elapsed >= self.harmonic_rhythm {
            // Check if we're on a good beat to change (avoid mid-beat changes)
            let beat_fraction = beat_phase % 1.0;
            beat_fraction < 0.1 || beat_fraction > 0.9
        } else {
            false
        }
    }

    /// Generate new harmony using all systems
    fn generate_new_harmony(&mut self) {
        // Get next chord from progression
        let next_function = self.chord_progression.get_next_chord_function(self.progression_position);

        // Apply substitutions if appropriate
        let final_function = self.chord_progression.apply_substitution(next_function, &mut self.rng);

        // Generate chord structure
        let new_chord = self.generate_chord_from_function(final_function);

        // Apply voice leading
        let voiced_chord = self.voice_leading.apply_voice_leading(&self.current_chord, &new_chord);

        // Update tension system
        self.tension_resolution.update_for_chord(&voiced_chord);

        // Store in memory
        self.harmony_memory.add_chord(voiced_chord.clone());

        // Update current state
        self.current_chord = voiced_chord;
        self.progression_position += 1;
    }

    /// Generate chord structure from function
    fn generate_chord_from_function(&mut self, function: ChordFunction) -> ChordStructure {
        // Get current scale
        let current_scale = self.modulation_system.current_scale;
        let scale_intervals = current_scale.intervals();
        let root_note = current_scale.root_note();

        // Map function to scale degree
        let scale_degree = self.function_to_scale_degree(function);
        let chord_root = root_note + scale_intervals[scale_degree % scale_intervals.len()];

        // Determine chord type based on scale and complexity
        let chord_type = self.determine_chord_type(function, scale_degree);

        // Generate voicing
        let voicing = self.generate_voicing(chord_root, chord_type);

        ChordStructure {
            root: chord_root,
            chord_type,
            voicing,
            inversion: 0,
            extensions: vec![],
            voice_assignment: vec![],
        }
    }

    /// Map chord function to scale degree
    fn function_to_scale_degree(&self, function: ChordFunction) -> usize {
        match function {
            ChordFunction::Tonic => 0,
            ChordFunction::Supertonic => 1,
            ChordFunction::Mediant => 2,
            ChordFunction::Subdominant => 3,
            ChordFunction::Dominant => 4,
            ChordFunction::Submediant => 5,
            ChordFunction::Leading => 6,
            _ => 0, // Simplified for complex functions
        }
    }

    /// Determine chord type based on context and complexity
    fn determine_chord_type(&mut self, _function: ChordFunction, _scale_degree: usize) -> ChordType {
        // Simple implementation - could be much more sophisticated
        if self.chord_complexity < 0.3 {
            if self.rng.gen::<bool>() { ChordType::Major } else { ChordType::Minor }
        } else if self.chord_complexity < 0.7 {
            match self.rng.gen::<u8>() % 4 {
                0 => ChordType::Major7,
                1 => ChordType::Minor7,
                2 => ChordType::Dominant7,
                _ => ChordType::Major,
            }
        } else {
            // Complex chords for high complexity
            match self.rng.gen::<u8>() % 6 {
                0 => ChordType::Major9,
                1 => ChordType::Minor9,
                2 => ChordType::Dominant9,
                3 => ChordType::Major11,
                4 => ChordType::Altered,
                _ => ChordType::Dominant7,
            }
        }
    }

    /// Generate chord voicing
    fn generate_voicing(&self, root: u8, chord_type: ChordType) -> Vec<u8> {
        let mut voicing = vec![root];

        // Add chord tones based on type
        match chord_type {
            ChordType::Major => {
                voicing.push(root + 4);  // Major third
                voicing.push(root + 7);  // Perfect fifth
            },
            ChordType::Minor => {
                voicing.push(root + 3);  // Minor third
                voicing.push(root + 7);  // Perfect fifth
            },
            ChordType::Dominant7 => {
                voicing.push(root + 4);  // Major third
                voicing.push(root + 7);  // Perfect fifth
                voicing.push(root + 10); // Minor seventh
            },
            ChordType::Major7 => {
                voicing.push(root + 4);  // Major third
                voicing.push(root + 7);  // Perfect fifth
                voicing.push(root + 11); // Major seventh
            },
            ChordType::Minor7 => {
                voicing.push(root + 3);  // Minor third
                voicing.push(root + 7);  // Perfect fifth
                voicing.push(root + 10); // Minor seventh
            },
            _ => {
                // Simplified - add basic triad
                voicing.push(root + 4);
                voicing.push(root + 7);
            }
        }

        // Apply voicing spread
        self.apply_voicing_spread(voicing)
    }

    /// Apply voicing spread to chord
    fn apply_voicing_spread(&self, mut voicing: Vec<u8>) -> Vec<u8> {
        if voicing.len() <= 1 {
            return voicing;
        }

        // Sort the voicing
        voicing.sort();

        // Apply spread based on voicing_spread parameter
        let spread_factor = self.voicing_spread / 12.0; // Convert to octaves

        for i in 1..voicing.len() {
            let additional_octaves = ((i as f32) * spread_factor * 0.3) as u8;
            voicing[i] += additional_octaves * 12;
        }

        // Ensure notes are in reasonable range
        for note in &mut voicing {
            *note = (*note).clamp(24, 96);
        }

        voicing
    }

    /// Convert unified chord type to harmony pattern chord type
    fn convert_chord_type(&self, chord_type: ChordType) -> ExternalChordType {
        match chord_type {
            ChordType::Major => ExternalChordType::Major,
            ChordType::Minor => ExternalChordType::Minor,
            ChordType::Dominant7 => ExternalChordType::Dominant7,
            ChordType::Major7 => ExternalChordType::Major7,
            ChordType::Minor7 => ExternalChordType::Minor7,
            ChordType::Diminished => ExternalChordType::Diminished,
            ChordType::Augmented => ExternalChordType::Augmented,
            _ => ExternalChordType::Major, // Default fallback
        }
    }

    /// Calculate current BPM (synchronized with rhythm generator)
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
        self.progression_position = 0;
        self.time_since_change = 0.0;
        self.current_chord = ChordStructure::default();
        self.harmony_memory.clear();
        self.evolution_engine.reset();
    }
}

impl PatternGenerator for UnifiedHarmonyGenerator {
    type Output = HarmonyPattern;

    fn next(&mut self) -> Self::Output {
        // Simple implementation - process with beat phase 0.0
        self.process_sample(0.0)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn update_parameters(&mut self, params: &PatternParameters) {
        // Map pattern parameters to input value (simplified)
        let input_value = params.intensity; // Intensity already in 0-1 range
        self.set_input_value(input_value);
    }

    fn pattern_length(&self) -> usize {
        // Calculate pattern length based on harmonic rhythm and BPM
        let bpm = self.calculate_current_bpm();
        let beats_per_chord = self.harmonic_rhythm;
        let pattern_chords = 4; // 4 chords per pattern
        let total_beats = beats_per_chord * pattern_chords as f32;
        let samples_per_beat = (60.0 / bpm) * self.sample_rate;
        (total_beats * samples_per_beat) as usize
    }

    fn is_cycle_complete(&self) -> bool {
        self.progression_position % 4 == 0 // Complete when back to start of 4-chord progression
    }
}

// Implementation stubs for supporting structures
impl ChordStructure {
    fn default() -> Self {
        Self {
            root: 60, // Middle C
            chord_type: ChordType::Major,
            voicing: vec![60, 64, 67], // C major triad
            inversion: 0,
            extensions: vec![],
            voice_assignment: vec![],
        }
    }
}

impl AdaptiveChordProgression {
    fn new() -> Self {
        Self {
            progression_templates: Self::create_progression_templates(),
            current_progression: vec![
                ChordFunction::Tonic,
                ChordFunction::Subdominant,
                ChordFunction::Dominant,
                ChordFunction::Tonic,
            ],
            template_weights: vec![1.0, 0.0, 0.0],
            progression_speed: 1.0,
            substitution_system: ChordSubstitutionSystem::new(),
        }
    }

    fn create_progression_templates() -> Vec<ProgressionTemplate> {
        vec![
            // Simple ambient progression
            ProgressionTemplate {
                name: "Ambient".to_string(),
                functions: vec![
                    ChordFunction::Tonic,
                    ChordFunction::Submediant,
                    ChordFunction::Subdominant,
                    ChordFunction::Tonic,
                ],
                input_range: (0.0, 1.2),
                style_traits: ProgressionTraits {
                    modal_tendency: 0.7,
                    chromatic_level: 0.1,
                    tension_preference: 0.2,
                    rhythm_density: 0.3,
                },
                loop_length: 4,
                loop_point: 0,
            },
            // Pop progression
            ProgressionTemplate {
                name: "Pop".to_string(),
                functions: vec![
                    ChordFunction::Tonic,
                    ChordFunction::Submediant,
                    ChordFunction::Subdominant,
                    ChordFunction::Dominant,
                ],
                input_range: (0.8, 2.2),
                style_traits: ProgressionTraits {
                    modal_tendency: 0.3,
                    chromatic_level: 0.3,
                    tension_preference: 0.5,
                    rhythm_density: 0.6,
                },
                loop_length: 4,
                loop_point: 0,
            },
            // EDM progression
            ProgressionTemplate {
                name: "EDM".to_string(),
                functions: vec![
                    ChordFunction::Tonic,
                    ChordFunction::Leading,
                    ChordFunction::Submediant,
                    ChordFunction::Dominant,
                ],
                input_range: (0.6, 1.0),
                style_traits: ProgressionTraits {
                    modal_tendency: 0.2,
                    chromatic_level: 0.6,
                    tension_preference: 0.8,
                    rhythm_density: 0.9,
                },
                loop_length: 4,
                loop_point: 0,
            },
        ]
    }

    fn set_input_value(&mut self, input: f32) {
        // Update template weights based on input
        for (i, template) in self.progression_templates.iter().enumerate() {
            let in_range = input >= template.input_range.0 && input <= template.input_range.1;
            self.template_weights[i] = if in_range { 1.0 } else { 0.1 };
        }

        // Update substitution system
        self.substitution_system.set_input_value(input);
    }

    fn get_next_chord_function(&mut self, progression_position: usize) -> ChordFunction {
        // Find most appropriate template
        let template_index = self.find_best_template();
        let template = &self.progression_templates[template_index];

        // Get next function from progression
        let function_index = progression_position % template.functions.len();
        template.functions[function_index]
    }

    fn find_best_template(&self) -> usize {
        let mut best_index = 0;
        let mut best_weight = 0.0;

        for (i, &weight) in self.template_weights.iter().enumerate() {
            if weight > best_weight {
                best_weight = weight;
                best_index = i;
            }
        }

        best_index
    }

    fn apply_substitution(&mut self, function: ChordFunction, rng: &mut StdRng) -> ChordFunction {
        self.substitution_system.apply_substitution(function, rng)
    }
}

impl ChordSubstitutionSystem {
    fn new() -> Self {
        Self {
            substitution_map: vec![],
            substitution_probability: 0.1,
            substitution_complexity: 0.3,
        }
    }

    fn set_input_value(&mut self, input: f32) {
        self.substitution_probability = input * 0.3;
        self.substitution_complexity = input;
    }

    fn apply_substitution(&self, function: ChordFunction, rng: &mut StdRng) -> ChordFunction {
        if rng.gen::<f32>() < self.substitution_probability {
            // Simple substitution logic
            match function {
                ChordFunction::Dominant => {
                    if rng.gen::<bool>() {
                        ChordFunction::Leading
                    } else {
                        function
                    }
                },
                _ => function,
            }
        } else {
            function
        }
    }
}

impl AdaptiveBassGenerator {
    fn new() -> Self {
        Self {
            pattern_templates: Self::create_bass_templates(),
            current_pattern: BassPattern::default(),
            bass_presence: 0.0,
            pattern_complexity: 0.3,
            walking_bass: WalkingBassEngine::new(),
            rhythmic_bass: RhythmicBassEngine::new(),
        }
    }

    fn create_bass_templates() -> Vec<BassPatternTemplate> {
        vec![
            BassPatternTemplate {
                name: "Root".to_string(),
                note_pattern: vec![0], // Just root
                rhythm_pattern: vec![0.0],
                input_range: (0.0, 1.0),
                style_traits: BassStyleTraits {
                    movement_level: 0.1,
                    rhythmic_activity: 0.2,
                    harmonic_function: 1.0,
                },
            },
            BassPatternTemplate {
                name: "Root-Fifth".to_string(),
                note_pattern: vec![0, 7], // Root and fifth
                rhythm_pattern: vec![0.0, 2.0],
                input_range: (0.8, 2.0),
                style_traits: BassStyleTraits {
                    movement_level: 0.4,
                    rhythmic_activity: 0.5,
                    harmonic_function: 0.8,
                },
            },
        ]
    }

    fn set_input_value(&mut self, input: f32) {
        self.bass_presence = input.powf(0.7); // Exponential curve
        self.pattern_complexity = input;
    }

    fn get_current_bass_note(&mut self, chord: &ChordStructure, _beat_phase: f32) -> u8 {
        if self.bass_presence < 0.1 {
            return 0; // No bass
        }

        // Simple bass note generation
        let bass_octave = 2; // Bass range
        chord.root - (5 - bass_octave) * 12
    }
}

impl BassPattern {
    fn default() -> Self {
        Self {
            notes: vec![36], // Low C
            timings: vec![0.0],
            position: 0,
            length: 1,
        }
    }
}

impl WalkingBassEngine {
    fn new() -> Self {
        Self {
            walking_pattern: vec![],
            step_patterns: vec![],
            target_notes: vec![],
            walking_speed: 1.0,
        }
    }
}

impl RhythmicBassEngine {
    fn new() -> Self {
        Self {
            rhythmic_patterns: vec![],
            current_pattern: 0,
            syncopation_level: 0.0,
            groove_lock: 0.8,
        }
    }
}

// Additional implementation stubs...
impl TensionResolutionEngine {
    fn new() -> Self {
        Self {
            current_tension: 0.0,
            tension_curve: TensionCurve::default(),
            resolution_patterns: vec![],
            tension_sources: vec![],
        }
    }

    fn set_input_value(&mut self, _input: f32) {
        // Update tension parameters based on input
    }

    fn update_for_chord(&mut self, _chord: &ChordStructure) {
        // Update tension based on new chord
    }
}

impl TensionCurve {
    fn default() -> Self {
        Self {
            control_points: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 0.0)],
            interpolation_type: CurveInterpolation::Smooth,
        }
    }
}

impl ModulationSystem {
    fn new() -> Self {
        Self {
            current_scale: Scale::CMajor,
            modulation_rate: 0.01,
            modulation_targets: vec![],
            pivot_system: PivotChordSystem::new(),
        }
    }

    fn set_input_value(&mut self, input: f32) {
        self.modulation_rate = input * 0.05;
    }
}

impl PivotChordSystem {
    fn new() -> Self {
        Self {
            pivot_chords: vec![],
            modulation_state: ModulationState::Stable,
        }
    }
}

impl VoiceLeadingEngine {
    fn new() -> Self {
        Self {
            voice_leading_rules: vec![],
            voice_positions: vec![60, 64, 67, 72], // C major chord
            motion_preferences: MotionPreferences::default(),
            parallel_avoidance: ParallelAvoidance::default(),
        }
    }

    fn apply_voice_leading(&self, _from: &ChordStructure, to: &ChordStructure) -> ChordStructure {
        // Simple implementation - just return the target chord
        to.clone()
    }
}

impl MotionPreferences {
    fn default() -> Self {
        Self {
            contrary_motion: 0.7,
            oblique_motion: 0.5,
            similar_motion: 0.3,
            parallel_motion: 0.1,
        }
    }
}

impl ParallelAvoidance {
    fn default() -> Self {
        Self {
            parallel_fifths: 0.9,
            parallel_octaves: 0.8,
            parallel_unisons: 0.7,
        }
    }
}

impl HarmonicTextureManager {
    fn new() -> Self {
        Self {
            active_voices: 1,
            max_voices: 6,
            density_curve: DensityCurve::default(),
            texture_types: vec![
                TextureType::Monophonic,
                TextureType::Homophonic,
                TextureType::Chordal,
                TextureType::Polyphonic,
            ],
            current_texture: TextureType::Monophonic,
        }
    }

    fn set_input_value(&mut self, input: f32) {
        // Update active voices based on input
        self.active_voices = (input * self.max_voices as f32) as usize + 1;
        self.active_voices = self.active_voices.min(self.max_voices);

        // Update texture type
        self.current_texture = match input {
            i if i <= 0.5 => TextureType::Monophonic,
            i if i <= 1.0 => TextureType::Sustained,
            i if i <= 2.0 => TextureType::Homophonic,
            _ => TextureType::Chordal,
        };
    }
}

impl DensityCurve {
    fn default() -> Self {
        Self {
            density_points: vec![(0.0, 0.1), (0.33, 0.4), (0.67, 0.7), (1.0, 1.0)],
            smoothing: 0.9,
        }
    }
}

impl HarmonyMemory {
    fn new() -> Self {
        Self {
            recent_progressions: VecDeque::new(),
            recent_chords: VecDeque::new(),
            memory_depth: 8,
            avoidance_strength: 0.6,
        }
    }

    fn add_chord(&mut self, chord: ChordStructure) {
        self.recent_chords.push_back(chord);
        if self.recent_chords.len() > self.memory_depth {
            self.recent_chords.pop_front();
        }
    }

    fn clear(&mut self) {
        self.recent_progressions.clear();
        self.recent_chords.clear();
    }
}

impl HarmonyEvolutionEngine {
    fn new() -> Self {
        Self {
            evolution_curves: vec![],
            evolution_phase: 0.0,
            evolution_speed: 0.01,
            complexity_evolution: ComplexityEvolution::default(),
        }
    }

    fn set_input_value(&mut self, _input: f32) {
        // Update evolution parameters
    }

    fn update(&mut self) {
        self.evolution_phase += self.evolution_speed;
        if self.evolution_phase >= 1.0 {
            self.evolution_phase -= 1.0;
        }
    }

    fn reset(&mut self) {
        self.evolution_phase = 0.0;
    }
}

impl ComplexityEvolution {
    fn default() -> Self {
        Self {
            chord_complexity: vec![0.3, 0.5, 0.7, 0.4],
            extension_usage: vec![0.1, 0.3, 0.6, 0.2],
            substitution_rate: vec![0.1, 0.2, 0.4, 0.15],
        }
    }
}