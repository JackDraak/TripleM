//! Pattern Event Translation System
//!
//! Translates sophisticated pattern events into concrete audio instructions
//! as specified in the integration interfaces design.

use crate::audio::{AudioEvent, AudioEventType, EventPriority, GeneratorTarget, MusicalContext};
use crate::audio::voice_coordination::{VoiceRequest, VoiceId, MusicalRole};
use crate::patterns::multi_scale_rhythm::{MultiScaleRhythmPattern, ComplexityProfile};
use crate::patterns::rhythm::RhythmPattern;
use crate::error::Result;
use std::collections::HashMap;

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
        pattern: &RhythmPattern, // Using RhythmPattern as melody source for now
        context: &MusicalContext,
        voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>>;

    /// Convert harmony pattern to audio events
    fn translate_harmony_pattern(
        &self,
        pattern: &RhythmPattern,
        context: &MusicalContext,
        voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>>;
}

/// Concrete implementation of pattern translation
#[derive(Debug)]
pub struct StandardPatternTranslator {
    /// Configuration for translation behavior
    pub config: TranslationConfig,

    /// Current voice mapping state
    pub voice_mapping: HashMap<PatternVoice, GeneratorVoice>,

    /// Sample rate for timing calculations
    sample_rate: f32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternVoice {
    Kick,
    Snare,
    HiHat,
    Melody,
    Bass,
    Harmony,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratorVoice {
    Environmental,
    GentleMelodic,
    ActiveAmbient,
    EdmStyle,
}

// Placeholder for VoiceAllocator - would be implemented in voice coordination
#[derive(Debug)]
pub struct VoiceAllocator {
    // Implementation details would go here
}

impl VoiceAllocator {
    pub fn new() -> Self {
        Self {}
    }
}

impl StandardPatternTranslator {
    /// Create a new pattern translator
    pub fn new(sample_rate: f32) -> Self {
        Self {
            config: TranslationConfig::default(),
            voice_mapping: HashMap::new(),
            sample_rate,
        }
    }

    /// Create translator with custom configuration
    pub fn with_config(sample_rate: f32, config: TranslationConfig) -> Self {
        Self {
            config,
            voice_mapping: HashMap::new(),
            sample_rate,
        }
    }
}

impl PatternEventTranslator for StandardPatternTranslator {
    fn translate_rhythm_pattern(
        &self,
        pattern: &MultiScaleRhythmPattern,
        context: &MusicalContext,
        duration: f32,
    ) -> Result<Vec<AudioEvent>> {
        let mut events = Vec::new();
        let samples_per_beat = (self.sample_rate * 60.0 / context.tempo) as u64;

        // Extract rhythm events from base pattern
        let base_pattern = &pattern.base_pattern;

        // Generate kick events
        if base_pattern.kick {
            events.push(AudioEvent {
                timestamp: 0, // On the beat
                event_type: AudioEventType::RhythmTrigger {
                    instrument: crate::audio::RhythmInstrument::Kick,
                    velocity: base_pattern.velocity * base_pattern.intensity,
                    timing_offset: base_pattern.micro_timing,
                },
                parameters: self.create_rhythm_parameters(base_pattern),
                priority: EventPriority::Rhythmic,
                target_generators: vec![GeneratorTarget::EdmStyle, GeneratorTarget::ActiveAmbient],
            });
        }

        // Generate snare events
        if base_pattern.snare {
            events.push(AudioEvent {
                timestamp: samples_per_beat / 2, // On the backbeat
                event_type: AudioEventType::RhythmTrigger {
                    instrument: crate::audio::RhythmInstrument::Snare,
                    velocity: base_pattern.velocity * base_pattern.intensity,
                    timing_offset: base_pattern.micro_timing,
                },
                parameters: self.create_rhythm_parameters(base_pattern),
                priority: EventPriority::Rhythmic,
                target_generators: vec![GeneratorTarget::EdmStyle],
            });
        }

        // Generate hi-hat events
        if base_pattern.hihat {
            // Create multiple hi-hat hits for rhythmic subdivision
            for i in 0..4 {
                let timestamp = (samples_per_beat / 4) * i as u64;
                let velocity = if i % 2 == 0 {
                    base_pattern.velocity * base_pattern.intensity
                } else {
                    base_pattern.velocity * base_pattern.intensity * 0.7 // Quieter off-beats
                };

                events.push(AudioEvent {
                    timestamp,
                    event_type: AudioEventType::RhythmTrigger {
                        instrument: crate::audio::RhythmInstrument::HiHat,
                        velocity,
                        timing_offset: base_pattern.micro_timing * 0.5,
                    },
                    parameters: self.create_rhythm_parameters(base_pattern),
                    priority: EventPriority::Background,
                    target_generators: vec![GeneratorTarget::EdmStyle, GeneratorTarget::ActiveAmbient],
                });
            }
        }

        // Generate accent events as melodic content
        if base_pattern.accent {
            let pitch = 60.0 + (context.musical_time.beat % 8.0) * 3.0; // Simple melodic pattern
            events.push(AudioEvent {
                timestamp: samples_per_beat / 8, // Slightly off the beat
                event_type: AudioEventType::NoteOn {
                    pitch,
                    velocity: base_pattern.velocity * base_pattern.intensity * 1.2,
                    voice_id: None,
                },
                parameters: self.create_melodic_parameters(base_pattern, pitch),
                priority: EventPriority::Melodic,
                target_generators: vec![GeneratorTarget::GentleMelodic],
            });
        }

        Ok(events)
    }

    fn translate_melody_pattern(
        &self,
        pattern: &RhythmPattern,
        context: &MusicalContext,
        _voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>> {
        let mut events = Vec::new();

        if pattern.intensity > 0.3 {
            // Generate simple melodic sequence based on musical context
            let root_note = context.key.root as f32;
            let scale_intervals = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0]; // Major scale

            for (i, &interval) in scale_intervals.iter().enumerate().take(4) {
                let timestamp = (i as f32 * self.sample_rate * 0.25) as u64; // Quarter notes
                let pitch = root_note + interval;

                events.push(AudioEvent {
                    timestamp,
                    event_type: AudioEventType::NoteOn {
                        pitch,
                        velocity: pattern.velocity * pattern.intensity * 0.8,
                        voice_id: Some(i as u32),
                    },
                    parameters: self.create_melodic_parameters(pattern, pitch),
                    priority: EventPriority::Melodic,
                    target_generators: vec![GeneratorTarget::GentleMelodic],
                });
            }
        }

        Ok(events)
    }

    fn translate_harmony_pattern(
        &self,
        pattern: &RhythmPattern,
        context: &MusicalContext,
        _voice_allocation: &mut VoiceAllocator,
    ) -> Result<Vec<AudioEvent>> {
        let mut events = Vec::new();

        if pattern.intensity > 0.5 {
            // Generate simple chord progression
            let chord_root = context.harmonic_context.current_chord.root as f32;
            let chord_tones = [0.0, 4.0, 7.0]; // Major triad

            for (i, &interval) in chord_tones.iter().enumerate() {
                let timestamp = 0; // All notes start together
                let pitch = chord_root + interval;

                events.push(AudioEvent {
                    timestamp,
                    event_type: AudioEventType::NoteOn {
                        pitch,
                        velocity: pattern.velocity * pattern.intensity * 0.6,
                        voice_id: Some((100 + i) as u32), // Different voice ID range
                    },
                    parameters: self.create_harmonic_parameters(pattern, pitch),
                    priority: EventPriority::Harmonic,
                    target_generators: vec![GeneratorTarget::GentleMelodic, GeneratorTarget::Environmental],
                });
            }
        }

        Ok(events)
    }
}

impl StandardPatternTranslator {
    fn create_rhythm_parameters(&self, pattern: &RhythmPattern) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("swing".to_string(), pattern.groove_swing);
        params.insert("micro_timing".to_string(), pattern.micro_timing);
        params.insert("intensity".to_string(), pattern.intensity);
        params
    }

    fn create_melodic_parameters(&self, pattern: &RhythmPattern, pitch: f32) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("pitch".to_string(), pitch);
        params.insert("velocity".to_string(), pattern.velocity);
        params.insert("swing".to_string(), pattern.groove_swing * 0.5); // Less swing for melody
        params
    }

    fn create_harmonic_parameters(&self, pattern: &RhythmPattern, pitch: f32) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("pitch".to_string(), pitch);
        params.insert("velocity".to_string(), pattern.velocity * 0.8); // Quieter for harmony
        params.insert("sustain".to_string(), 1.0); // Longer sustain for chords
        params
    }
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            complexity_mapping: ComplexityMappingStrategy::Musical,
            voice_allocation: VoiceAllocationStrategy::Musical,
            timing_quantization: TimingQuantization {
                quantize_to_grid: true,
                grid_resolution: 0.25, // Sixteenth notes
                humanization_amount: 0.02,
                swing_amount: 0.1,
            },
        }
    }
}