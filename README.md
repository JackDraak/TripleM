# TripleM - Mood Music Module

A sophisticated Rust-based procedural music generation system that creates continuous audio streams with seamless real-time parameter morphing. Features both simple and advanced interfaces with professional-grade cross-fading capabilities.

## Features

### Core Capabilities
- **Unified Parameter Control**: Single 0.0-1.0 input seamlessly morphs across entire musical spectrum
- **Continuous Musical Evolution**: From ambient environmental sounds to high-energy electronic music
- **Advanced Cross-fading**: Artifact-free real-time parameter transitions with multiple curve types
- **Dual Interface Design**: Simple interface for basic use, advanced interface for professional control
- **Real-time Performance**: Optimized lock-free audio pipeline with comprehensive monitoring

### Musical Progression (0.0-1.0)
- **Environmental (0.0-0.33)**: Natural ambient sounds with subtle textures
- **Gentle Melodic (0.33-0.67)**: Relaxing melodies with increasing complexity
- **Active Rhythmic (0.67-1.0)**: Energetic electronic music with complex patterns
- **Seamless Morphing**: Continuous transitions without discrete boundaries

### Advanced Features
- **Preset Management**: Save/load parameter configurations with metadata
- **Real-time Monitoring**: CPU usage, audio levels, performance metrics
- **Parameter Validation**: Intelligent constraint handling and curve mapping
- **Cross-fade Statistics**: Monitor transition timing and system health
- **Comprehensive API**: Both simple and advanced control interfaces

## Applications

### GUI Controller (Recommended for Testing)

The easiest way to test the module with intuitive sliders:

```bash
cargo run --release --bin mood_gui
```

Features:
- 🎭 **Mood Slider**: Drag to change between 0.0-1.0
- 🔊 **Volume Slider**: Control master volume
- ▶️ **Play/Stop**: Start and stop audio playback
- 📊 **Generator Status**: Real-time display of active generators

### Console Interface

For command-line testing:

```bash
cargo run --release --bin audio_demo
```

Enter mood values (0.0-1.0) or commands:
- `0.1` - Environmental sounds
- `0.35` - Gentle melodic music
- `0.6` - Active ambient music
- `0.85` - EDM-style music
- `q` - Quit
- `v 0.8` - Set volume to 80%

### Testing Tools

```bash
# Test audio levels and generation
cargo run --release --bin test_audio

# Debug generator internals
cargo run --release --bin debug_audio

# Test unified controller interface (NEW)
cargo run --release --bin test_unified_controller

# Quick test across mood ranges
./test_moods.sh
```

## Library Usage

### Simple Interface (Recommended for Basic Use)

```rust
use mood_music_module::MoodMusicModule;

// Create and start the module
let mut module = MoodMusicModule::new(44100)?;
module.start();

// Control mood and volume
module.set_mood(0.35);   // Gentle melodic range
module.set_volume(0.8);  // 80% volume

// Generate audio
let sample = module.get_next_sample();

// Or fill buffers efficiently
let mut buffer = vec![0.0; 1024];
module.fill_buffer(&mut buffer);
```

### Advanced Interface (Professional Control)

```rust
use mood_music_module::{
    UnifiedController, MoodConfig, ControlParameter,
    ChangeSource, PresetMetadata
};

// Create advanced controller
let config = MoodConfig::default_with_sample_rate(44100);
let mut controller = UnifiedController::new(config)?;
controller.start()?;

// Smooth parameter changes with cross-fading
controller.set_mood_intensity(0.7)?;
controller.set_parameter_smooth(
    ControlParameter::RhythmicDensity,
    0.8,
    ChangeSource::UserInterface
)?;

// Advanced parameter control
controller.set_parameter_smooth(
    ControlParameter::HarmonicComplexity, 0.6,
    ChangeSource::UserInterface
)?;
controller.set_parameter_smooth(
    ControlParameter::StereoWidth, 0.9,
    ChangeSource::UserInterface
)?;

// Preset management
let metadata = PresetMetadata {
    description: "Focus work session".to_string(),
    category: "Productivity".to_string(),
    tags: vec!["focus".to_string(), "ambient".to_string()],
    created_time: std::time::SystemTime::now(),
    user_rating: Some(5),
};
controller.save_preset("My Focus".to_string(), metadata)?;
controller.load_preset("My Focus")?;

// Real-time monitoring
let status = controller.get_system_status();
println!("CPU Usage: {:.1}%", status.cpu_usage * 100.0);
println!("Active crossfades: {}", status.crossfade_stats.active_crossfades);

// Generate audio with monitoring
let sample = controller.get_next_sample();
let levels = controller.get_output_levels();
```

### Available Control Parameters

```rust
// Musical parameters
ControlParameter::MoodIntensity      // Primary 0-1 control
ControlParameter::RhythmicDensity    // Beat complexity
ControlParameter::MelodicDensity     // Note density
ControlParameter::HarmonicComplexity // Chord complexity
ControlParameter::MusicalComplexity  // Overall complexity

// Synthesis parameters
ControlParameter::SynthesisCharacter // Wavetable ↔ Additive
ControlParameter::TimbralBrightness  // Sound brightness
ControlParameter::TexturalComplexity // Smooth ↔ Granular

// Spatial and effects
ControlParameter::StereoWidth        // Stereo field width
ControlParameter::AmbientSpace       // Reverb/space
ControlParameter::DynamicRange       // Compression

// Advanced controls
ControlParameter::Tempo              // BPM control
ControlParameter::Humanization       // Natural variation
ControlParameter::TransitionSmoothing // Cross-fade behavior
```

## Technical Architecture

### Core Systems
- **Unified Controller**: Advanced interface with comprehensive parameter control
- **Cross-fade Manager**: Seamless real-time parameter transitions with multiple curve types
- **Lock-free Audio Pipeline**: Real-time performance with atomic parameter updates
- **Parameter State Management**: Intelligent validation, constraints, and curve mapping
- **Real-time Monitoring**: CPU usage, audio levels, and performance tracking

### Audio Generation
- **Unified Generators**: Continuous parameter response across 0.0-1.0 range
- **Pattern Systems**: Unified rhythm, melody, and harmony generators
- **Synthesis Integration**: Seamless wavetable ↔ additive synthesis morphing
- **Output Processing**: Advanced mixing with limiter and spatial effects

### Advanced Features
- **Preset System**: Complete preset management with metadata and categorization
- **Parameter Curves**: Linear, exponential, logarithmic, and sigmoid mapping
- **Change Source Tracking**: Monitor parameter changes from different sources
- **Performance Metrics**: Comprehensive system health and timing statistics

## Audio Quality

All generators produce excellent audio levels:
- **RMS Levels**: ~0.058-0.070 (good audible range)
- **Sample Rate**: 44.1kHz
- **Bit Depth**: 32-bit float
- **Channels**: Mono (easily expandable to stereo)

## Building

```bash
# Development build
cargo build

# Optimized release build
cargo build --release

# Run tests
cargo test

# Build specific applications
cargo build --release --bin mood_gui
cargo build --release --bin audio_demo
```

## Requirements

- Rust 1.70+
- Audio output device
- Linux: ALSA development libraries

## License

MIT OR Apache-2.0