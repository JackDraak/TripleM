use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{MoodGenerator, GeneratorState};
use crate::patterns::markov::{MarkovChain, presets};
use crate::audio::{NaturalVariation, StateVariableFilter, FilterType, FmSynth, FmAlgorithm, AdsrEnvelope, midi_to_frequency};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Enhanced gentle melodic music generator (0.25-0.5 mood range)
/// Generates sophisticated spa-like music with FM synthesis, voice leading, and natural variation
#[derive(Debug)]
pub struct GentleMelodicGenerator {
    intensity: f32,
    sample_rate: f32,
    time: f64,

    // Enhanced harmony generation with voice leading
    chord_voices: Vec<HarmonyVoice>,  // Individual FM voices for harmony
    chord_progression: Vec<Vec<u8>>,  // Sophisticated chord progression
    chord_index: usize,
    chord_timer: f32,
    chord_duration: f32,

    // Enhanced melody generation
    melody_chain: MarkovChain<u8>,
    melody_voice: MelodyVoice,
    melody_timer: f32,
    melody_note_duration: f32,

    // Key modulation system
    current_key: u8,
    key_timer: f32,
    key_change_duration: f32,

    // Natural variation for organic evolution
    variation: NaturalVariation,

    // Advanced filtering for ambient textures
    harmony_filter: StateVariableFilter,
    melody_filter: StateVariableFilter,

    // Ambient texture layer
    ambient_layer: AmbientTextureLayer,

    // Random number generator
    rng: StdRng,
}

/// Individual harmony voice with FM synthesis
#[derive(Debug, Clone)]
struct HarmonyVoice {
    fm_synth: FmSynth,
    envelope: AdsrEnvelope,
    target_note: u8,
    current_note: u8,
    is_active: bool,
    voice_leading_timer: f32,
    voice_leading_duration: f32,
}

/// Enhanced melody voice with expression
#[derive(Debug, Clone)]
struct MelodyVoice {
    fm_synth: FmSynth,
    envelope: AdsrEnvelope,
    current_note: u8,
    expression_lfo_phase: f32,  // For vibrato and expression
    is_active: bool,
}

/// Ambient texture layer for atmospheric depth
#[derive(Debug, Clone)]
struct AmbientTextureLayer {
    texture_oscillators: Vec<TextureOscillator>,
    texture_filter: StateVariableFilter,
    texture_timer: f32,
    texture_evolution_duration: f32,
}

/// Individual texture oscillator for ambient layer
#[derive(Debug, Clone)]
struct TextureOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    detune_amount: f32,
}

impl GentleMelodicGenerator {
    pub fn new(config: &MoodConfig) -> Result<Self> {
        let mut rng = StdRng::from_entropy();

        // Initialize with C major key
        let current_key = 60; // C4

        // Create gentle chord progression in C major
        let chord_progression = Self::create_gentle_chord_progression(current_key);
        let current_chord = Self::chord_to_frequencies(&chord_progression[0]);

        // Initialize Markov chain for melody
        let melody_chain = Self::create_melody_chain();

        Ok(Self {
            intensity: 0.0,
            sample_rate: config.sample_rate as f32,
            time: 0.0,

            current_chord,
            chord_progression,
            chord_index: 0,
            chord_timer: 0.0,
            chord_duration: 4.0, // 4 seconds per chord

            melody_chain,
            current_melody_note: current_key + 12, // Start melody an octave higher
            melody_timer: 0.0,
            melody_note_duration: 2.0, // 2 seconds per melody note

            current_key,
            key_timer: 0.0,
            key_change_duration: 120.0, // 2 minutes

            chord_phases: vec![0.0; 4], // Up to 4 voices per chord
            melody_phase: 0.0,

            rng,

            chord_envelope: 0.0,
            melody_envelope: 0.0,
        })
    }

    /// Create a gentle chord progression suitable for spa music
    fn create_gentle_chord_progression(root: u8) -> Vec<Vec<u8>> {
        // I - vi - IV - V progression in close voicing
        vec![
            vec![root, root + 4, root + 7],           // I (C major)
            vec![root + 9, root + 12, root + 16],     // vi (A minor)
            vec![root + 5, root + 9, root + 12],      // IV (F major)
            vec![root + 7, root + 11, root + 14],     // V (G major)
            vec![root, root + 4, root + 7],           // I (back to C major)
            vec![root + 2, root + 5, root + 9],       // ii (D minor)
            vec![root + 5, root + 9, root + 12],      // IV (F major)
            vec![root, root + 4, root + 7],           // I (C major)
        ]
    }

    /// Convert MIDI notes to frequencies
    fn chord_to_frequencies(chord: &[u8]) -> Vec<f32> {
        chord.iter().map(|&note| Self::midi_to_frequency(note)).collect()
    }

    /// Convert MIDI note to frequency
    fn midi_to_frequency(midi_note: u8) -> f32 {
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    /// Create a Markov chain for gentle melodies
    fn create_melody_chain() -> MarkovChain<u8> {
        let mut chain = MarkovChain::new(2);

        // Add gentle melodic patterns (C major scale patterns)
        let patterns = vec![
            vec![72, 74, 72, 71, 69, 67, 69, 72],    // Gentle ascending/descending
            vec![67, 69, 71, 72, 71, 69, 67, 65],    // Scale patterns
            vec![72, 67, 69, 72, 71, 67, 69, 72],    // Arpeggiated patterns
            vec![69, 71, 72, 74, 72, 71, 69, 67],    // Flowing patterns
            vec![65, 67, 69, 71, 69, 67, 65, 64],    // Lower register
            vec![76, 74, 72, 71, 72, 74, 76, 77],    // Higher register
        ];

        for pattern in patterns {
            chain.add_sequence(&pattern);
        }

        chain.normalize();
        chain
    }

    /// Generate chord harmony sample
    fn generate_chord_sample(&mut self, time: f64) -> f32 {
        if self.current_chord.is_empty() {
            return 0.0;
        }

        let mut chord_sample = 0.0;

        for (i, &freq) in self.current_chord.iter().enumerate() {
            if i < self.chord_phases.len() {
                // Soft sine wave with slight detuning for warmth
                let detune = 1.0 + (i as f32 * 0.002); // Slight detuning
                let phase = self.chord_phases[i] * 2.0 * std::f32::consts::PI;
                chord_sample += (phase.sin() * 0.3) / self.current_chord.len() as f32;

                // Update phase
                self.chord_phases[i] += (freq * detune) / self.sample_rate;
                if self.chord_phases[i] >= 1.0 {
                    self.chord_phases[i] -= 1.0;
                }
            }
        }

        // Apply envelope for smooth chord changes
        self.update_chord_envelope(time);
        chord_sample * self.chord_envelope
    }

    /// Generate melody sample
    fn generate_melody_sample(&mut self, time: f64) -> f32 {
        let freq = Self::midi_to_frequency(self.current_melody_note);

        // Gentle sine wave with soft attack
        let phase = self.melody_phase * 2.0 * std::f32::consts::PI;
        let melody_sample = phase.sin() * 0.2;

        // Update phase
        self.melody_phase += freq / self.sample_rate;
        if self.melody_phase >= 1.0 {
            self.melody_phase -= 1.0;
        }

        // Apply envelope for smooth note changes
        self.update_melody_envelope(time);
        melody_sample * self.melody_envelope
    }

    /// Update chord envelope for smooth transitions
    fn update_chord_envelope(&mut self, _time: f64) {
        let target = if self.intensity > 0.0 { 1.0 } else { 0.0 };
        let rate = 0.002; // Smooth envelope
        self.chord_envelope += (target - self.chord_envelope) * rate;
    }

    /// Update melody envelope for smooth note transitions
    fn update_melody_envelope(&mut self, _time: f64) {
        let target = if self.intensity > 0.0 { 1.0 } else { 0.0 };
        let rate = 0.001; // Even smoother for melody
        self.melody_envelope += (target - self.melody_envelope) * rate;
    }

    /// Update chord progression
    fn update_chord_progression(&mut self, delta_time: f32) {
        self.chord_timer += delta_time;

        if self.chord_timer >= self.chord_duration {
            self.chord_timer = 0.0;
            self.chord_index = (self.chord_index + 1) % self.chord_progression.len();
            self.current_chord = Self::chord_to_frequencies(&self.chord_progression[self.chord_index]);

            // Reset chord phases for smooth transition
            self.chord_phases.fill(0.0);
        }
    }

    /// Update melody progression
    fn update_melody_progression(&mut self, delta_time: f32) {
        self.melody_timer += delta_time;

        if self.melody_timer >= self.melody_note_duration {
            self.melody_timer = 0.0;

            // Generate next melody note using Markov chain
            if let Some(next_note) = self.melody_chain.next(&mut self.rng) {
                self.current_melody_note = next_note;
            }

            // Vary note durations slightly for more natural feel
            self.melody_note_duration = 1.5 + self.rng.gen::<f32>() * 1.0; // 1.5-2.5 seconds
        }
    }

    /// Update key modulation (every 2 minutes)
    fn update_key_modulation(&mut self, delta_time: f32) {
        self.key_timer += delta_time;

        if self.key_timer >= self.key_change_duration {
            self.key_timer = 0.0;

            // Modulate to related keys (perfect fifth, relative minor, etc.)
            let modulation_options = [
                self.current_key + 7,  // Perfect fifth up
                self.current_key - 5,  // Perfect fourth down
                self.current_key + 3,  // Minor third up (relative minor)
                self.current_key - 3,  // Minor third down
            ];

            let new_key = modulation_options[self.rng.gen_range(0..modulation_options.len())];
            self.current_key = new_key.clamp(48, 72); // Keep in reasonable range

            // Regenerate chord progression for new key
            self.chord_progression = Self::create_gentle_chord_progression(self.current_key);
            self.chord_index = 0;
            self.current_chord = Self::chord_to_frequencies(&self.chord_progression[0]);

            // Reset timers
            self.chord_timer = 0.0;
            self.melody_timer = 0.0;
        }
    }
}

impl MoodGenerator for GentleMelodicGenerator {
    fn generate_sample(&mut self, time: f64) -> f32 {
        self.time = time;

        if self.intensity <= 0.0 {
            return 0.0;
        }

        let delta_time = 1.0 / self.sample_rate;

        // Update all progression timers
        self.update_chord_progression(delta_time);
        self.update_melody_progression(delta_time);
        self.update_key_modulation(delta_time);

        // Generate chord harmony (background)
        let chord_sample = self.generate_chord_sample(time);

        // Generate melody (foreground)
        let melody_sample = self.generate_melody_sample(time);

        // Mix chord and melody with appropriate balance
        let mixed = chord_sample * 0.6 + melody_sample * 0.4;

        // Apply intensity scaling and gentle limiting
        let output = mixed * self.intensity;
        output.clamp(-0.8, 0.8) // Gentle limiting
    }

    fn generate_batch(&mut self, output: &mut [f32], start_time: f64) {
        let sample_duration = 1.0 / self.sample_rate as f64;
        for (i, sample) in output.iter_mut().enumerate() {
            let time = start_time + i as f64 * sample_duration;
            *sample = self.generate_sample(time);
        }
    }

    fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    fn intensity(&self) -> f32 {
        self.intensity
    }

    fn reset(&mut self) {
        self.time = 0.0;
        self.chord_index = 0;
        self.chord_timer = 0.0;
        self.melody_timer = 0.0;
        self.key_timer = 0.0;
        self.chord_phases.fill(0.0);
        self.melody_phase = 0.0;
        self.chord_envelope = 0.0;
        self.melody_envelope = 0.0;

        // Reset to C major
        self.current_key = 60;
        self.chord_progression = Self::create_gentle_chord_progression(self.current_key);
        self.current_chord = Self::chord_to_frequencies(&self.chord_progression[0]);

        // Reset melody chain
        self.melody_chain = Self::create_melody_chain();
        self.current_melody_note = self.current_key + 12;
    }

    fn update_focus_parameters(&mut self) {
        // When gentle melodic is dominant, make chord changes more frequent
        self.chord_duration = 3.0 + self.rng.gen::<f32>() * 2.0; // 3-5 seconds
        self.melody_note_duration = 1.0 + self.rng.gen::<f32>() * 1.5; // 1-2.5 seconds
    }

    fn get_state(&self) -> GeneratorState {
        let chord_names = ["I", "vi", "IV", "V", "I", "ii", "IV", "I"];
        let current_chord_name = chord_names.get(self.chord_index)
            .unwrap_or(&"Unknown")
            .to_string();

        let key_name = match self.current_key % 12 {
            0 => "C", 1 => "C#", 2 => "D", 3 => "D#", 4 => "E", 5 => "F",
            6 => "F#", 7 => "G", 8 => "G#", 9 => "A", 10 => "A#", 11 => "B",
            _ => "?",
        };

        GeneratorState {
            name: "Gentle Melodic".to_string(),
            intensity: self.intensity,
            is_active: self.intensity > 0.0,
            current_pattern: format!("{} major - {} chord", key_name, current_chord_name),
            pattern_progress: self.key_timer / self.key_change_duration,
            cpu_usage: 0.15, // Higher CPU usage due to harmony + melody
        }
    }
}