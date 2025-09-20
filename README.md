# TripleM - Mood Music Module

A Rust-based procedural mood music generation module that creates continuous audio streams based on a single 0.0-1.0 mood parameter.

## Features

- **Single Parameter Control**: One float input (0.0-1.0) controls the entire mood spectrum
- **Four Distinct Mood Categories**:
  - **Environmental (0.0-0.25)**: Natural sounds (ocean, wind, forest, rain)
  - **Gentle Melodic (0.25-0.5)**: Relaxing spa/massage music with gentle melodies
  - **Active Ambient (0.5-0.75)**: Productivity-focused ambient music with rhythmic elements
  - **EDM Style (0.75-1.0)**: High-energy electronic dance music with beats and drops
- **Smooth Transitions**: Seamless blending between mood categories
- **Real-time Audio**: 44.1kHz output using CPAL for cross-platform compatibility
- **Procedural Generation**: Uses Markov chains and mathematical synthesis for infinite variety

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

# Quick test across mood ranges
./test_moods.sh
```

## Library Usage

```rust
use mood_music_module::{MoodMusicModule, MoodConfig};

// Create and configure the module
let config = MoodConfig::default();
let mut module = MoodMusicModule::with_config(config)?;

// Start audio generation
module.start();
module.set_mood(0.35);  // Gentle melodic
module.set_volume(0.8); // 80% volume

// Get audio samples
let sample = module.get_next_sample();

// Or fill a buffer
let mut buffer = vec![0.0; 1024];
module.fill_buffer(&mut buffer);
```

## Technical Architecture

- **Lock-free Audio Pipeline**: Real-time performance with atomic parameter updates
- **Generator Pool**: Four specialized generators for different mood ranges
- **Transition Manager**: Smooth crossfading between mood categories
- **Output Mixer**: Advanced mixing with limiter and master volume control
- **Markov Chains**: Procedural pattern generation for melodies and rhythms

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