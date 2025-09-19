# Procedural Mood Music Research

## Overview
This document summarizes research findings for implementing a procedural mood music module in Rust. The module will generate continuous audio streams based on a single 0-1 float input, with four distinct mood categories that blend seamlessly.

## Core Algorithmic Approaches

### 1. Markov Chains
- **Foundation**: Most effective foundational technique for musical pattern generation
- **Implementation**: Scan example music and build probability tables
- **Simple case**: "C is 20% likely to follow A"
- **Advanced**: Look at sequences: "C A B" is 15% likely to be followed by B, 4% by Bb
- **Application**: Useful for melody, harmony, and rhythm generation across all four mood categories

### 2. Pink Noise (1/f Noise)
- **Purpose**: Starting point for natural-sounding randomness
- **Application**: Use to select durations and notes from scales
- **Benefit**: Sounds more musical than pure random, good baseline for environmental sounds

### 3. Synthesis Techniques
- **Additive Synthesis**: Build sounds by adding sine waves (good for complex tones)
- **Subtractive Synthesis**: Start with rich sound, remove frequencies (bass sounds)
- **FM Synthesis**: One waveform modulates another's frequency (bells, complex textures)
- **Granular Synthesis**: Manipulate small audio grains for unique textures

## Mode-Specific Research Findings

### Environmental Sounds (0.0-0.25 range)
**Ocean Sound Synthesis**:
- Pink noise as foundation (sounds like waterfall/rough ocean)
- Add "far waves" (bass, spatial, breaking in distance)
- Add "close waves" (smaller, high-frequency, directional)
- White noise for wind simulation

**Rain Generation**:
- Pure synthesis without samples
- Millions of bubbles merging = pink noise spectrum
- Dynamic weather system integration possible

**General Environmental**:
- Real-time procedural generation preferred over loops
- Web Audio API techniques adaptable to Rust
- Avoid repetitive feeling through natural variation

### Gentle Melodic Music (0.25-0.5 range)
**Spa/Massage Music Characteristics**:
- Key shifts every ~2 minutes as specified
- Harmonic progressions using probability chains
- Gentle tempo (60-80 BPM typical)
- Ambient synthesis with soft attack/release
- Layered approach: bass, harmony, ambient texture

### Active Ambient Music (0.5-0.75 range)
**Productivity Music Requirements**:
- Strong rhythmic foundation
- 90-120 BPM range typical
- Repetitive patterns for focus
- Layered composition: rhythm + ambient + occasional melodic elements
- Avoid distracting melodic complexity

### EDM-Style Music (0.75-1.0 range)
**Research Findings**:
- **GEDMAS**: Generative Electronic Dance Music Algorithmic System uses Markov chains
- **Genetic Algorithms**: Evolve rhythm sections with binary chromosomes
  - 32 bits for kick drum pattern
  - 32 bits for snare
  - 32 bits for hi-hat, etc.
- **Key Principle**: Repetition is crucial - repeat 4-8 bars, 4-8 times
- **Rhythm Challenge**: "Hardest thing to generate is rhythm"
- **Implementation**: Start with samples in same key, evolve to MIDI generation

## Technical Implementation Requirements

### Audio Quality Standards
- **Sample Rate**: 44.1kHz confirmed as industry standard
- **Bit Depth**: 16-bit minimum, 24-bit preferred for synthesis
- **Real-time Requirement**: 44,100 samples/second processing
- **Performance**: Highly efficient callback functions essential

### Rust Audio Libraries
**CPAL (Cross-Platform Audio Library)**:
- Pure Rust, low-level audio I/O
- Submit PCM to audio output
- Real-time callback system
- Cross-platform support

**Rodio**:
- Built on CPAL
- Higher-level audio playback
- Source trait for sound representation
- Automatic mixing of multiple sources
- Background thread management

**Hardware Requirements**:
- 32-bit float (f32) support
- 32-bit atomic operations minimum
- Lock-free data structures recommended for production

## Implementation Architecture Recommendations

### 1. Layered Audio System
- Base ambient layer (always present)
- Rhythmic layer (scales with input)
- Melodic layer (varies by mode)
- Texture layer (environmental effects)

### 2. Interpolation Strategy
- Linear interpolation between adjacent modes
- Crossfade implementation for smooth transitions
- Maintain continuity during parameter changes

### 3. Pattern Management
- Pre-generate pattern libraries for each mode
- Use probability tables for variations
- Implement 2-3 minute cycle timing system
- Smooth transitions between patterns

### 4. Real-time Constraints
- Lock-free circular buffers for audio data
- Separate threads for generation vs. playback
- Atomic operations for parameter updates
- Minimize allocations in audio callback

## Key Design Principles

1. **Repetition**: Essential for musical coherence (4-8 bar patterns, repeated 4-8 times)
2. **Avoid Pure Randomness**: Results in "1970's computer beeps" - use structured probability
3. **Natural Variation**: Prevent loop detection through organic parameter drift
4. **Efficient Processing**: Audio callback must complete in ~1ms at 44.1kHz
5. **Seamless Transitions**: Fade between modes over several seconds
6. **Memory Management**: Pre-allocate buffers, avoid real-time allocation

## Next Steps
1. Design module architecture based on these findings
2. Set up Rust project with CPAL/Rodio dependencies
3. Implement core audio engine with real-time constraints
4. Create individual generators for each mood category
5. Implement interpolation and transition systems