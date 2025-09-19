# Mood Music Module Architecture

## Overview
This document outlines the architecture for the Rust mood music module that generates continuous audio streams based on a single 0-1 float input parameter.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MoodMusicModule                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │  Input Handler  │───▶│        Audio Pipeline           │ │
│  └─────────────────┘    └──────────────────────────────────┘ │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │ Mode Classifier │    │       Output Mixer              │ │
│  └─────────────────┘    └──────────────────────────────────┘ │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │ Generator Pool  │    │       Audio Stream              │ │
│  └─────────────────┘    └──────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MoodMusicModule (Main Interface)
```rust
pub struct MoodMusicModule {
    current_mood: f32,          // 0.0 to 1.0
    audio_pipeline: AudioPipeline,
    is_running: AtomicBool,
}

impl MoodMusicModule {
    pub fn new(sample_rate: u32) -> Self;
    pub fn set_mood(&mut self, mood: f32);
    pub fn get_audio_frame(&mut self) -> AudioFrame;
    pub fn start(&mut self);
    pub fn stop(&mut self);
}
```

### 2. Audio Pipeline
```rust
struct AudioPipeline {
    generators: GeneratorPool,
    mixer: OutputMixer,
    transition_manager: TransitionManager,
    sample_rate: u32,
    buffer_size: usize,
}
```

### 3. Generator Pool
Houses all four mood generators and manages their lifecycle:

```rust
struct GeneratorPool {
    environmental: EnvironmentalGenerator,
    gentle_melodic: GentleMelodicGenerator,
    active_ambient: ActiveAmbientGenerator,
    edm_style: EdmStyleGenerator,
}
```

### 4. Mode Generators

#### Base Generator Trait
```rust
trait MoodGenerator {
    fn generate_sample(&mut self, time: f64) -> f32;
    fn set_intensity(&mut self, intensity: f32);  // 0.0 to 1.0
    fn reset_pattern(&mut self);
    fn update_time(&mut self, delta: f64);
}
```

#### Environmental Generator (0.0-0.25)
- **Components**: Pink noise base, wave simulation, wind synthesis
- **Patterns**: 3-minute soundscape cycles
- **Techniques**: Additive synthesis for layered environmental sounds

#### Gentle Melodic Generator (0.25-0.5)
- **Components**: Harmonic progression, gentle melody, ambient pad
- **Patterns**: 2-minute key shifts
- **Techniques**: Markov chains for chord progressions, smooth synthesis

#### Active Ambient Generator (0.5-0.75)
- **Components**: Rhythmic foundation, ambient textures, productivity-focused
- **Patterns**: Steady 4-bar rhythmic cycles
- **Techniques**: Structured rhythms with ambient synthesis

#### EDM Style Generator (0.75-1.0)
- **Components**: Beat generation, bass synthesis, lead synthesis
- **Patterns**: 3-minute evolution cycles with 4-8 bar repetitions
- **Techniques**: Genetic algorithms for rhythm, FM synthesis for leads

## Audio Processing Architecture

### 1. Real-time Audio Thread
```rust
// Audio callback running at 44.1kHz
fn audio_callback(output_buffer: &mut [f32], module: &mut MoodMusicModule) {
    for sample in output_buffer.iter_mut() {
        *sample = module.get_next_sample();
    }
}
```

### 2. Sample Generation Pipeline
```
Input Mood (0-1) → Mode Classification → Generator Weights → Mixed Output
                                     ↓
                              Transition Management
                                     ↓
                              Smooth Interpolation
```

### 3. Mode Classification Logic
```rust
#[derive(Debug, Clone)]
struct MoodWeights {
    environmental: f32,
    gentle_melodic: f32,
    active_ambient: f32,
    edm_style: f32,
}

impl MoodWeights {
    fn from_mood_value(mood: f32) -> Self {
        // Overlapping regions for smooth transitions
        match mood {
            m if m <= 0.25 => Self::environmental_dominant(m),
            m if m <= 0.5  => Self::gentle_transition(m),
            m if m <= 0.75 => Self::active_transition(m),
            m => Self::edm_dominant(m),
        }
    }
}
```

## Memory Management

### 1. Lock-Free Design
- Use `Arc<AtomicF32>` for mood parameter sharing
- Lock-free circular buffers for audio data
- Atomic operations for state changes

### 2. Buffer Management
```rust
struct AudioBuffer {
    data: Vec<f32>,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    capacity: usize,
}
```

### 3. Pattern Storage
- Pre-generated pattern libraries stored in static arrays
- Lazy loading of pattern variations
- Memory pooling for temporary calculations

## Transition System

### 1. TransitionManager
```rust
struct TransitionManager {
    current_weights: MoodWeights,
    target_weights: MoodWeights,
    transition_time: f32,
    transition_duration: f32,
}

impl TransitionManager {
    fn update(&mut self, target_mood: f32, delta_time: f32) -> MoodWeights;
    fn start_transition(&mut self, target_mood: f32);
    fn is_transitioning(&self) -> bool;
}
```

### 2. Crossfade Implementation
- 3-5 second fade duration for mood changes
- Smooth interpolation using cosine curves
- Maintain audio continuity during transitions

## Pattern Generation System

### 1. Markov Chain Implementation
```rust
struct MarkovChain<T> {
    transitions: HashMap<Vec<T>, HashMap<T, f32>>,
    order: usize,
    current_sequence: Vec<T>,
}

impl<T> MarkovChain<T> {
    fn next(&mut self) -> Option<T>;
    fn add_sequence(&mut self, sequence: &[T]);
}
```

### 2. Rhythm Generation
```rust
struct RhythmPattern {
    kick: [bool; 32],
    snare: [bool; 32],
    hihat: [bool; 32],
    bpm: f32,
}

impl RhythmPattern {
    fn evolve(&mut self, fitness_fn: impl Fn(&RhythmPattern) -> f32);
    fn crossover(&self, other: &RhythmPattern) -> RhythmPattern;
}
```

## Performance Considerations

### 1. Real-time Constraints
- Target: <1ms processing time per audio callback
- Pre-compute expensive operations outside audio thread
- Use lookup tables for trigonometric functions
- Minimize dynamic allocation

### 2. CPU Optimization
- SIMD instructions for bulk sample processing
- Efficient filter implementations
- Optimized random number generation
- Pattern caching strategies

### 3. Memory Layout
- Structure of Arrays (SoA) for better cache performance
- Aligned memory allocation for SIMD
- Ring buffers for streaming data

## Error Handling

### 1. Graceful Degradation
- Continue audio output even with generator failures
- Fallback to simpler synthesis methods
- Audio glitch prevention during errors

### 2. Recovery Mechanisms
- Automatic pattern regeneration on corruption
- Safe state restoration
- Logging without blocking audio thread

## Configuration and Extensibility

### 1. Configuration Structure
```rust
#[derive(Debug, Clone)]
pub struct MoodConfig {
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub transition_duration: f32,
    pub pattern_cycle_lengths: [f32; 4],  // Per mood type
    pub synthesis_quality: SynthesisQuality,
}
```

### 2. Plugin Interface
```rust
pub trait MoodGeneratorPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn mood_range(&self) -> (f32, f32);
    fn generate_sample(&mut self, context: &GeneratorContext) -> f32;
}
```

This architecture provides a solid foundation for implementing the mood music module with real-time performance, smooth transitions, and extensible design patterns.