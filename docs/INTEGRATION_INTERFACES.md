# TripleM Integration Interfaces Specification

## 🎯 **Interface Design for System Integration**

This document specifies the exact interfaces needed to connect the sophisticated parameter control systems to actual audio generation.

## 🔌 **Interface 1: Audio Event System**

### **Core Audio Event Types**
```rust
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
pub enum EventPriority {
    Background,    // Ambient, environmental
    Rhythmic,      // Beat, pulse
    Melodic,       // Main melody line
    Harmonic,      // Chord progression
    Lead,          // Primary musical focus
}

#[derive(Debug, Clone)]
pub enum GeneratorTarget {
    Environmental,
    GentleMelodic,
    ActiveAmbient,
    EdmStyle,
    All,
    Coordination(Vec<GeneratorTarget>), // Multiple generators working together
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
```

## 🎵 **Interface 2: Musical Context Coordination**

### **Shared Musical State**
```rust
/// Global musical context shared across all generators
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
```

## 🎛️ **Interface 3: Pattern-to-Audio Translation**

### **Pattern Event Translator**
```rust
/// Translates sophisticated pattern events into concrete audio instructions
pub trait PatternEventTranslator {
    /// Convert multi-scale rhythm pattern to audio events
    fn translate_rhythm_pattern(
        &self,
        pattern: &MultiScaleRhythmPattern,
        context: &MusicalContext,
        duration: f32,
    ) -> Result<Vec<AudioEvent>>;

    /// Convert melody pattern to audio events
    fn translate_melody_pattern(
        &self,
        pattern: &MelodyPattern,
        context: &MusicalContext,
        voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>>;

    /// Convert harmony pattern to audio events
    fn translate_harmony_pattern(
        &self,
        pattern: &HarmonyPattern,
        context: &MusicalContext,
        voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>>;
}

/// Concrete implementation of pattern translation
pub struct StandardPatternTranslator {
    /// Configuration for translation behavior
    pub config: TranslationConfig,

    /// Current voice mapping state
    pub voice_mapping: HashMap<PatternVoice, GeneratorVoice>,
}

#[derive(Debug, Clone)]
pub struct TranslationConfig {
    /// How to map pattern complexity to audio parameters
    pub complexity_mapping: ComplexityMappingStrategy,

    /// Voice allocation preferences
    pub voice_allocation: VoiceAllocationStrategy,

    /// Timing quantization settings
    pub timing_quantization: TimingQuantization,
}

#[derive(Debug, Clone)]
pub enum ComplexityMappingStrategy {
    Linear,        // Direct 1:1 mapping
    Exponential,   // More dramatic at high complexity
    Logarithmic,   // More sensitive at low complexity
    Musical,       // Musically-informed mapping curves
}

#[derive(Debug, Clone)]
pub enum VoiceAllocationStrategy {
    RoundRobin,    // Cycle through available voices
    Priority,      // Allocate based on event priority
    Musical,       // Allocate based on musical context
    Dynamic,       // Adaptive allocation based on load
}

#[derive(Debug, Clone)]
pub struct TimingQuantization {
    pub quantize_to_grid: bool,
    pub grid_resolution: f32,      // In beats
    pub humanization_amount: f32,  // Random timing variation
    pub swing_amount: f32,         // Rhythmic swing
}
```

## 🎹 **Interface 4: Polyphonic Voice Coordination**

### **Voice Management System**
```rust
/// Manages polyphonic voices across all generators
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

#[derive(Debug, Clone)]
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

/// Handles voice allocation conflicts intelligently
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
```

## 🔗 **Interface 5: Generator Integration Protocol**

### **Enhanced Generator Interface**
```rust
/// Extended generator interface for polyphonic coordination
pub trait PolyphonicGenerator: MoodGenerator {
    /// Process a batch of audio events
    fn process_events(&mut self, events: &[AudioEvent], context: &MusicalContext) -> Result<()>;

    /// Allocate a voice for polyphonic playback
    fn allocate_voice(&mut self, request: &VoiceRequest) -> Result<VoiceId>;

    /// Release a specific voice
    fn release_voice(&mut self, voice_id: VoiceId) -> Result<()>;

    /// Update voice parameters in real-time
    fn update_voice(&mut self, voice_id: VoiceId, parameters: &HashMap<String, f32>) -> Result<()>;

    /// Get current voice states for coordination
    fn get_voice_states(&self) -> Vec<VoiceState>;

    /// Set musical context for intelligent generation
    fn set_musical_context(&mut self, context: &MusicalContext);

    /// Get generator's preferred voice allocation
    fn get_voice_requirements(&self) -> VoiceRequirements;
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

#[derive(Debug, Clone)]
pub struct VoiceRequirements {
    pub preferred_voice_count: usize,
    pub max_voice_count: usize,
    pub supported_roles: Vec<MusicalRole>,
    pub voice_allocation_preference: VoiceAllocationStrategy,
}
```

## 🎯 **Implementation Priority Matrix**

| Interface | Implementation Complexity | Audio Impact | Integration Difficulty |
|-----------|---------------------------|--------------|----------------------|
| Audio Events | Medium | High | Medium |
| Musical Context | High | High | High |
| Pattern Translation | High | Critical | High |
| Voice Coordination | Very High | Medium | Very High |
| Generator Integration | Medium | Critical | Medium |

## 🚀 **Integration Rollout Strategy**

### **Phase 1: Basic Event System**
1. Implement AudioEvent types
2. Create basic PatternEventTranslator
3. Connect to one generator (start with GentleMelodic)

### **Phase 2: Musical Context**
1. Implement MusicalContext sharing
2. Add chord progression coordination
3. Synchronize timing across generators

### **Phase 3: Polyphonic Coordination**
1. Implement VoiceCoordinator
2. Add intelligent voice allocation
3. Enable true polyphonic flexibility

### **Phase 4: Advanced Features**
1. Musical intelligence in conflict resolution
2. Complex pattern coordination
3. Real-time performance optimization

This interface specification provides the blueprint for connecting sophisticated parameter control to actual audio generation while enabling true polyphonic flexibility.