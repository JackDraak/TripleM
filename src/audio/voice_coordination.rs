//! Polyphonic Voice Coordination System
//!
//! This module implements the voice coordination system that enables multiple generators
//! to work together musically with shared musical context and intelligent voice allocation.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use crate::error::{Result, MoodMusicError};

/// Core musical context shared across all generators
#[derive(Debug, Clone)]
pub struct MusicalContext {
    /// Current musical key
    pub key: MusicalKey,

    /// Current tempo in BPM
    pub tempo: f32,

    /// Time signature (beats per measure, note value)
    pub time_signature: (u8, u8),

    /// Current position in musical time
    pub musical_time: MusicalTime,

    /// Master clock phase for synchronization
    pub master_phase: f64,

    /// Current harmonic context
    pub harmonic_context: HarmonicContext,

    /// Active musical voices
    pub active_voices: Vec<VoiceState>,
}

#[derive(Debug, Clone)]
pub struct MusicalKey {
    pub root: u8,              // MIDI note number
    pub scale_type: ScaleType,
    pub mode: MusicalMode,
}

#[derive(Debug, Clone, Copy)]
pub enum ScaleType {
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    Pentatonic,
    Blues,
    Dorian,
    Mixolydian,
    Chromatic,
}

#[derive(Debug, Clone, Copy)]
pub enum MusicalMode {
    Ionian,     // Major
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,    // Natural minor
    Locrian,
}

#[derive(Debug, Clone)]
pub struct MusicalTime {
    /// Current measure number
    pub measure: u32,

    /// Current beat within measure (0.0-beats_per_measure)
    pub beat: f32,

    /// Current subdivision (for precise timing)
    pub subdivision: f32,

    /// Global time in samples
    pub sample_time: u64,
}

#[derive(Debug, Clone)]
pub struct HarmonicContext {
    /// Current chord being played
    pub current_chord: Chord,

    /// Chord progression context
    pub chord_progression: Vec<Chord>,

    /// Position in progression
    pub progression_position: usize,

    /// Harmonic tension level (0.0-1.0)
    pub tension_level: f32,
}

#[derive(Debug, Clone)]
pub struct Chord {
    pub root: u8,              // MIDI note
    pub chord_type: ChordType,
    pub extensions: Vec<ChordExtension>,
    pub voicing: ChordVoicing,
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
    Sus2,
    Sus4,
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
}

#[derive(Debug, Clone)]
pub enum ChordVoicing {
    Root,          // Root position
    FirstInversion,
    SecondInversion,
    Open,          // Spread across multiple octaves
    Close,         // Tight spacing
    Custom(Vec<u8>), // Specific MIDI notes
}

/// Manages polyphonic voices across all generators
#[derive(Debug)]
pub struct VoiceCoordinator {
    /// Pool of available voices
    pub voice_pool: VoicePool,

    /// Current voice allocations
    pub allocations: HashMap<VoiceId, VoiceAllocation>,

    /// Voice conflict resolution strategy
    pub conflict_resolver: ConflictResolver,

    /// Musical context for intelligent allocation
    pub musical_context: Arc<Mutex<MusicalContext>>,
}

#[derive(Debug, Clone)]
pub struct VoicePool {
    /// Maximum polyphony across all generators
    pub max_voices: usize,

    /// Available voices per generator
    pub generator_voice_limits: HashMap<GeneratorTarget, usize>,

    /// Currently active voices
    pub active_voices: HashMap<VoiceId, Voice>,

    /// Voice allocation queue
    pub allocation_queue: VecDeque<VoiceRequest>,
}

#[derive(Debug, Clone)]
pub struct Voice {
    pub id: VoiceId,
    pub generator: GeneratorTarget,
    pub note: MusicalNote,
    pub envelope: VoiceEnvelope,
    pub parameters: HashMap<String, f32>,
    pub musical_role: MusicalRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceId(pub u32);

#[derive(Debug, Clone)]
pub struct MusicalNote {
    pub pitch: f32,         // MIDI note (can be fractional)
    pub velocity: f32,      // 0.0-1.0
    pub duration: Option<f32>, // None = sustain until note off
    pub articulation: Articulation,
}

#[derive(Debug, Clone, Copy)]
pub enum Articulation {
    Legato,      // Smooth connection
    Staccato,    // Short and detached
    Marcato,     // Emphasized
    Tenuto,      // Held full value
    Natural,     // Default articulation
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicalRole {
    Bass,        // Low fundamental frequencies
    Harmony,     // Chord tones and extensions
    Melody,      // Primary melodic line
    Counter,     // Counter-melody
    Rhythm,      // Rhythmic elements
    Texture,     // Background texture/atmosphere
    Lead,        // Primary focus/solo
}

#[derive(Debug, Clone)]
pub struct VoiceEnvelope {
    pub attack: f32,     // Attack time in seconds
    pub decay: f32,      // Decay time in seconds
    pub sustain: f32,    // Sustain level (0.0-1.0)
    pub release: f32,    // Release time in seconds
    pub current_phase: EnvelopePhase,
    pub current_level: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum EnvelopePhase {
    Attack,
    Decay,
    Sustain,
    Release,
    Finished,
}

#[derive(Debug, Clone)]
pub struct VoiceAllocation {
    pub voice: Voice,
    pub start_time: f64,
    pub priority: EventPriority,
    pub musical_importance: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum EventPriority {
    Background,    // Ambient, environmental
    Rhythmic,      // Beat, pulse
    Melodic,       // Main melody line
    Harmonic,      // Chord progression
    Lead,          // Primary musical focus
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratorTarget {
    Environmental,
    GentleMelodic,
    ActiveAmbient,
    EdmStyle,
    All,
    Coordination(u8), // ID for coordinated groups
}

#[derive(Debug, Clone)]
pub struct VoiceRequest {
    pub note: MusicalNote,
    pub musical_role: MusicalRole,
    pub priority: EventPriority,
    pub preferred_generator: Option<GeneratorTarget>,
    pub coordination_group: Option<String>, // For voice grouping
}

#[derive(Debug, Clone)]
pub struct VoiceState {
    pub voice_id: VoiceId,
    pub note: MusicalNote,
    pub envelope: VoiceEnvelope,
    pub musical_role: MusicalRole,
    pub current_amplitude: f32,
    pub harmonic_function: Option<HarmonicFunction>,
}

/// Handles voice allocation conflicts intelligently
#[derive(Debug)]
pub struct ConflictResolver {
    pub strategy: ConflictResolutionStrategy,
    pub musical_priority: MusicalPriorityRules,
}

#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    /// Steal oldest voice
    OldestFirst,

    /// Steal lowest priority voice
    PriorityBased,

    /// Steal voice with lowest amplitude
    QuietestFirst,

    /// Musical intelligence (avoid stealing important notes)
    MusicallyIntelligent,

    /// Deny new allocation if no voices available
    Deny,
}

#[derive(Debug, Clone)]
pub struct MusicalPriorityRules {
    /// Priority weights by musical role
    pub role_priorities: HashMap<MusicalRole, f32>,

    /// Priority based on harmonic importance
    pub harmonic_priorities: HashMap<HarmonicFunction, f32>,

    /// Priority based on rhythmic position
    pub rhythmic_priorities: HashMap<RhythmicPosition, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonicFunction {
    Root,        // Chord root
    Third,       // Major/minor third
    Fifth,       // Perfect fifth
    Seventh,     // Seventh interval
    Extension,   // 9th, 11th, 13th
    Tension,     // Dissonant intervals
    Bass,        // Bass note
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RhythmicPosition {
    Downbeat,    // Beat 1
    StrongBeat,  // Other strong beats
    WeakBeat,    // Weak beats
    Offbeat,     // Between beats
    Syncopated,  // Syncopated position
}

impl Default for MusicalContext {
    fn default() -> Self {
        Self {
            key: MusicalKey {
                root: 60, // C4
                scale_type: ScaleType::Major,
                mode: MusicalMode::Ionian,
            },
            tempo: 120.0,
            time_signature: (4, 4),
            musical_time: MusicalTime {
                measure: 0,
                beat: 0.0,
                subdivision: 0.0,
                sample_time: 0,
            },
            master_phase: 0.0,
            harmonic_context: HarmonicContext {
                current_chord: Chord {
                    root: 60,
                    chord_type: ChordType::Major,
                    extensions: Vec::new(),
                    voicing: ChordVoicing::Root,
                },
                chord_progression: Vec::new(),
                progression_position: 0,
                tension_level: 0.0,
            },
            active_voices: Vec::new(),
        }
    }
}

impl VoiceCoordinator {
    pub fn new(max_voices: usize) -> Self {
        let mut generator_limits = HashMap::new();
        generator_limits.insert(GeneratorTarget::Environmental, max_voices / 4);
        generator_limits.insert(GeneratorTarget::GentleMelodic, max_voices / 4);
        generator_limits.insert(GeneratorTarget::ActiveAmbient, max_voices / 4);
        generator_limits.insert(GeneratorTarget::EdmStyle, max_voices / 4);

        Self {
            voice_pool: VoicePool {
                max_voices,
                generator_voice_limits: generator_limits,
                active_voices: HashMap::new(),
                allocation_queue: VecDeque::new(),
            },
            allocations: HashMap::new(),
            conflict_resolver: ConflictResolver::default(),
            musical_context: Arc::new(Mutex::new(MusicalContext::default())),
        }
    }

    /// Allocate a voice for polyphonic playback
    pub fn allocate_voice(&mut self, request: VoiceRequest) -> Result<VoiceId> {
        // Check if we have available voices
        if self.voice_pool.active_voices.len() >= self.voice_pool.max_voices {
            return self.resolve_voice_conflict(request);
        }

        // Create new voice
        let voice_id = VoiceId(self.allocations.len() as u32);
        let generator = request.preferred_generator.unwrap_or(GeneratorTarget::GentleMelodic);

        let voice = Voice {
            id: voice_id,
            generator,
            note: request.note.clone(),
            envelope: VoiceEnvelope::default(),
            parameters: HashMap::new(),
            musical_role: request.musical_role,
        };

        let allocation = VoiceAllocation {
            voice: voice.clone(),
            start_time: 0.0, // Would be actual time
            priority: request.priority,
            musical_importance: self.calculate_musical_importance(&request),
        };

        self.voice_pool.active_voices.insert(voice_id, voice);
        self.allocations.insert(voice_id, allocation);

        Ok(voice_id)
    }

    /// Release a specific voice
    pub fn release_voice(&mut self, voice_id: VoiceId) -> Result<()> {
        if let Some(voice) = self.voice_pool.active_voices.remove(&voice_id) {
            self.allocations.remove(&voice_id);
            Ok(())
        } else {
            Err(MoodMusicError::VoiceCoordinationError(format!("Voice {:?} not found", voice_id)))
        }
    }

    /// Update voice parameters in real-time
    pub fn update_voice(&mut self, voice_id: VoiceId, parameters: HashMap<String, f32>) -> Result<()> {
        if let Some(voice) = self.voice_pool.active_voices.get_mut(&voice_id) {
            voice.parameters.extend(parameters);
            Ok(())
        } else {
            Err(MoodMusicError::VoiceCoordinationError(format!("Voice {:?} not found", voice_id)))
        }
    }

    /// Get current voice states for coordination
    pub fn get_voice_states(&self) -> Vec<VoiceState> {
        self.voice_pool.active_voices.values().map(|voice| {
            VoiceState {
                voice_id: voice.id,
                note: voice.note.clone(),
                envelope: voice.envelope.clone(),
                musical_role: voice.musical_role,
                current_amplitude: 0.5, // Would be calculated
                harmonic_function: None, // Would be calculated
            }
        }).collect()
    }

    /// Set musical context for intelligent generation
    pub fn set_musical_context(&mut self, context: MusicalContext) {
        if let Ok(mut ctx) = self.musical_context.lock() {
            *ctx = context;
        }
    }

    fn resolve_voice_conflict(&mut self, request: VoiceRequest) -> Result<VoiceId> {
        match self.conflict_resolver.strategy {
            ConflictResolutionStrategy::Deny => {
                Err(MoodMusicError::VoiceCoordinationError("No voices available".to_string()))
            }
            ConflictResolutionStrategy::OldestFirst => {
                // Find oldest voice and steal it
                if let Some((oldest_id, _)) = self.allocations.iter()
                    .min_by_key(|(_, alloc)| (alloc.start_time * 1000.0) as u64) {
                    let old_id = *oldest_id;
                    self.release_voice(old_id)?;
                    self.allocate_voice(request)
                } else {
                    Err(MoodMusicError::VoiceCoordinationError("No voices to steal".to_string()))
                }
            }
            ConflictResolutionStrategy::PriorityBased => {
                // Find lowest priority voice
                let new_importance = self.calculate_musical_importance(&request);
                if let Some((lowest_id, _)) = self.allocations.iter()
                    .filter(|(_, alloc)| alloc.musical_importance < new_importance)
                    .min_by(|(_, a), (_, b)| a.musical_importance.partial_cmp(&b.musical_importance).unwrap()) {
                    let old_id = *lowest_id;
                    self.release_voice(old_id)?;
                    self.allocate_voice(request)
                } else {
                    Err(MoodMusicError::VoiceCoordinationError("New voice not important enough".to_string()))
                }
            }
            _ => Err(MoodMusicError::VoiceCoordinationError("Voice stealing strategy not implemented".to_string()))
        }
    }

    fn calculate_musical_importance(&self, request: &VoiceRequest) -> f32 {
        let mut importance = match request.priority {
            EventPriority::Background => 0.1,
            EventPriority::Rhythmic => 0.3,
            EventPriority::Melodic => 0.7,
            EventPriority::Harmonic => 0.5,
            EventPriority::Lead => 1.0,
        };

        // Adjust based on musical role
        importance *= match request.musical_role {
            MusicalRole::Bass => 0.9,
            MusicalRole::Harmony => 0.6,
            MusicalRole::Melody => 0.8,
            MusicalRole::Lead => 1.0,
            MusicalRole::Rhythm => 0.7,
            MusicalRole::Texture => 0.3,
            MusicalRole::Counter => 0.5,
        };

        importance
    }
}

impl Default for VoiceEnvelope {
    fn default() -> Self {
        Self {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            current_phase: EnvelopePhase::Attack,
            current_level: 0.0,
        }
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        let mut role_priorities = HashMap::new();
        role_priorities.insert(MusicalRole::Lead, 1.0);
        role_priorities.insert(MusicalRole::Melody, 0.8);
        role_priorities.insert(MusicalRole::Bass, 0.7);
        role_priorities.insert(MusicalRole::Harmony, 0.6);
        role_priorities.insert(MusicalRole::Rhythm, 0.5);
        role_priorities.insert(MusicalRole::Counter, 0.4);
        role_priorities.insert(MusicalRole::Texture, 0.2);

        let mut harmonic_priorities = HashMap::new();
        harmonic_priorities.insert(HarmonicFunction::Root, 1.0);
        harmonic_priorities.insert(HarmonicFunction::Bass, 0.9);
        harmonic_priorities.insert(HarmonicFunction::Fifth, 0.7);
        harmonic_priorities.insert(HarmonicFunction::Third, 0.6);
        harmonic_priorities.insert(HarmonicFunction::Seventh, 0.5);
        harmonic_priorities.insert(HarmonicFunction::Extension, 0.3);
        harmonic_priorities.insert(HarmonicFunction::Tension, 0.2);

        let mut rhythmic_priorities = HashMap::new();
        rhythmic_priorities.insert(RhythmicPosition::Downbeat, 1.0);
        rhythmic_priorities.insert(RhythmicPosition::StrongBeat, 0.8);
        rhythmic_priorities.insert(RhythmicPosition::WeakBeat, 0.4);
        rhythmic_priorities.insert(RhythmicPosition::Offbeat, 0.3);
        rhythmic_priorities.insert(RhythmicPosition::Syncopated, 0.6);

        Self {
            strategy: ConflictResolutionStrategy::MusicallyIntelligent,
            musical_priority: MusicalPriorityRules {
                role_priorities,
                harmonic_priorities,
                rhythmic_priorities,
            },
        }
    }
}

/// Concrete musical instruction that generators can execute
#[derive(Debug, Clone)]
pub struct AudioEvent {
    /// When this event should occur (in samples from now)
    pub timestamp: u64,

    /// Type of audio event
    pub event_type: AudioEventType,

    /// Event-specific parameters
    pub parameters: HashMap<String, f32>,

    /// Priority for voice allocation conflicts
    pub priority: EventPriority,

    /// Which generator(s) should handle this event
    pub target_generators: Vec<GeneratorTarget>,
}

#[derive(Debug, Clone)]
pub enum AudioEventType {
    /// Start a note with specific pitch and velocity
    NoteOn {
        pitch: f32,           // MIDI note number (can be fractional for microtones)
        velocity: f32,        // 0.0-1.0
        voice_id: Option<u32>, // For polyphonic tracking
    },

    /// Stop a specific note
    NoteOff {
        pitch: f32,
        voice_id: Option<u32>,
    },

    /// Trigger a rhythm element
    RhythmTrigger {
        instrument: RhythmInstrument,
        velocity: f32,
        timing_offset: f32,   // Micro-timing adjustment
    },

    /// Change a generator parameter
    ParameterChange {
        parameter: String,
        value: f32,
        transition_time: f32, // Seconds to reach target
    },

    /// Complex musical instruction
    MusicalPhrase {
        phrase_type: PhraseType,
        duration: f32,
        complexity: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum RhythmInstrument {
    Kick,
    Snare,
    HiHat,
    OpenHat,
    Crash,
    Ride,
    Percussion,
}

#[derive(Debug, Clone, Copy)]
pub enum PhraseType {
    Question,      // Musical phrase that creates tension
    Answer,        // Musical phrase that resolves tension
    Development,   // Phrase that develops existing material
    Transition,    // Bridge between sections
    Climax,        // Peak intensity phrase
}

#[derive(Debug, Clone)]
pub struct VoiceRequirements {
    pub preferred_voice_count: usize,
    pub max_voice_count: usize,
    pub supported_roles: Vec<MusicalRole>,
    pub voice_allocation_preference: VoiceAllocationStrategy,
}

#[derive(Debug, Clone)]
pub enum VoiceAllocationStrategy {
    RoundRobin,    // Cycle through available voices
    Priority,      // Allocate based on event priority
    Musical,       // Allocate based on musical context
    Dynamic,       // Adaptive allocation based on load
}

impl Default for VoiceRequirements {
    fn default() -> Self {
        Self {
            preferred_voice_count: 4,
            max_voice_count: 8,
            supported_roles: vec![
                MusicalRole::Melody,
                MusicalRole::Harmony,
                MusicalRole::Bass,
                MusicalRole::Texture,
            ],
            voice_allocation_preference: VoiceAllocationStrategy::Musical,
        }
    }
}