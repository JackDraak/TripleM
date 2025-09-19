use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{MoodGenerator, GeneratorState};
use crate::patterns::markov::MarkovChain;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Active ambient music generator (0.5-0.75 mood range)
/// Generates productivity-focused music with rhythmic elements and ambient textures
#[derive(Debug)]
pub struct ActiveAmbientGenerator {
    intensity: f32,
    sample_rate: f32,
    time: f64,

    // Rhythmic foundation
    beat_pattern: Vec<f32>,          // Current beat pattern (0.0 = silence, 1.0 = accent)
    beat_index: usize,
    beat_timer: f32,
    beat_duration: f32,              // Duration per beat (60/BPM seconds)
    current_bpm: f32,

    // Ambient pad synthesis
    pad_oscillators: Vec<PadOscillator>,
    pad_frequencies: Vec<f32>,
    pad_progression: Vec<Vec<u8>>,   // Chord progression for pads
    pad_chord_index: usize,
    pad_timer: f32,
    pad_chord_duration: f32,         // Duration per chord change

    // Rhythmic pulse (filtered noise + sine waves)
    pulse_phase: f32,
    pulse_filter_state: f32,
    pulse_filter_cutoff: f32,

    // Arpeggiated sequences
    arp_chain: MarkovChain<u8>,
    current_arp_note: u8,
    arp_timer: f32,
    arp_note_duration: f32,
    arp_phase: f32,

    // Pattern evolution (4-minute cycles)
    pattern_timer: f32,
    pattern_duration: f32,           // 4 minutes = 240 seconds
    current_pattern: usize,

    // Random number generator
    rng: StdRng,

    // Volume envelopes for smooth mixing
    beat_envelope: f32,
    pad_envelope: f32,
    arp_envelope: f32,
}

#[derive(Debug, Clone)]
struct PadOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    filter_state: f32,
    envelope: f32,
}

impl PadOscillator {
    fn new(frequency: f32) -> Self {
        Self {
            phase: 0.0,
            frequency,
            amplitude: 0.3,
            filter_state: 0.0,
            envelope: 0.0,
        }
    }

    fn generate_sample(&mut self, sample_rate: f32) -> f32 {
        // Warm, detuned sawtooth wave
        let phase_increment = self.frequency / sample_rate;
        self.phase += phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Sawtooth wave
        let raw_sample = (self.phase * 2.0 - 1.0) * self.amplitude;

        // Simple lowpass filter for warmth
        let alpha = 0.1; // Low cutoff for warm sound
        self.filter_state += (raw_sample - self.filter_state) * alpha;

        self.filter_state * self.envelope
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    fn update_envelope(&mut self, target: f32) {
        let rate = 0.001; // Slow attack/release for pads
        self.envelope += (target - self.envelope) * rate;
    }
}

impl ActiveAmbientGenerator {
    pub fn new(config: &MoodConfig) -> Result<Self> {
        let mut rng = StdRng::from_entropy();

        // Initialize with 120 BPM (moderate productivity tempo)
        let current_bpm = 120.0;
        let beat_duration = 60.0 / current_bpm;

        // Create initial beat pattern (4/4 with emphasis on 1 and 3)
        let beat_pattern = vec![1.0, 0.3, 0.7, 0.3]; // Strong-weak-medium-weak

        // Initialize pad oscillators (4 voices for ambient chords)
        let pad_oscillators = vec![
            PadOscillator::new(261.63), // C4
            PadOscillator::new(329.63), // E4
            PadOscillator::new(392.00), // G4
            PadOscillator::new(523.25), // C5
        ];

        // Create ambient chord progression (more complex than gentle melodic)
        let pad_progression = Self::create_ambient_chord_progression();
        let pad_frequencies = Self::chord_to_frequencies(&pad_progression[0]);

        // Initialize arpeggio Markov chain
        let arp_chain = Self::create_arpeggio_chain();

        Ok(Self {
            intensity: 0.0,
            sample_rate: config.sample_rate as f32,
            time: 0.0,

            beat_pattern,
            beat_index: 0,
            beat_timer: 0.0,
            beat_duration,
            current_bpm,

            pad_oscillators,
            pad_frequencies,
            pad_progression,
            pad_chord_index: 0,
            pad_timer: 0.0,
            pad_chord_duration: 8.0, // 8 seconds per chord

            pulse_phase: 0.0,
            pulse_filter_state: 0.0,
            pulse_filter_cutoff: 800.0,

            arp_chain,
            current_arp_note: 72, // C5
            arp_timer: 0.0,
            arp_note_duration: 0.25, // 16th notes
            arp_phase: 0.0,

            pattern_timer: 0.0,
            pattern_duration: 240.0, // 4 minutes

            current_pattern: 0,
            rng,

            beat_envelope: 0.0,
            pad_envelope: 0.0,
            arp_envelope: 0.0,
        })
    }

    /// Create ambient chord progression with extensions and inversions
    fn create_ambient_chord_progression() -> Vec<Vec<u8>> {
        // More complex progression with 7th chords and extensions
        vec![
            vec![60, 64, 67, 71],        // Cmaj7
            vec![57, 60, 64, 67],        // Am7
            vec![65, 69, 72, 76],        // Fmaj7
            vec![62, 65, 69, 72],        // Dm7
            vec![67, 71, 74, 77],        // Gmaj7
            vec![64, 67, 71, 74],        // Em7
            vec![65, 69, 72, 76],        // Fmaj7
            vec![60, 64, 67, 71],        // Cmaj7
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

    /// Create Markov chain for arpeggiated patterns
    fn create_arpeggio_chain() -> MarkovChain<u8> {
        let mut chain = MarkovChain::new(2);

        // Productivity-focused arpeggio patterns
        let patterns = vec![
            vec![72, 76, 79, 84, 79, 76, 72, 67],    // Ascending/descending
            vec![60, 64, 67, 72, 67, 64, 60, 55],    // Lower register patterns
            vec![76, 72, 79, 76, 84, 79, 76, 72],    // Interlocking patterns
            vec![67, 71, 74, 79, 74, 71, 67, 62],    // Flowing sequences
            vec![72, 67, 76, 72, 79, 76, 72, 67],    // Rhythmic patterns
            vec![84, 79, 76, 72, 76, 79, 84, 88],    // Higher register
        ];

        for pattern in patterns {
            chain.add_sequence(&pattern);
        }

        chain.normalize();
        chain
    }

    /// Generate rhythmic pulse sample
    fn generate_pulse_sample(&mut self, time: f64) -> f32 {
        // Current beat strength
        let beat_strength = if self.beat_pattern.is_empty() {
            0.0
        } else {
            self.beat_pattern[self.beat_index]
        };

        // Filtered noise pulse
        let noise = (self.rng.gen::<f32>() - 0.5) * 2.0;
        let alpha = self.pulse_filter_cutoff / self.sample_rate;
        self.pulse_filter_state += (noise - self.pulse_filter_state) * alpha;

        // Sine wave pulse at fundamental frequency
        let pulse_freq = self.current_bpm / 60.0; // 1 Hz for 60 BPM
        let sine_pulse = (self.pulse_phase * 2.0 * std::f32::consts::PI).sin();
        self.pulse_phase += pulse_freq / self.sample_rate;
        if self.pulse_phase >= 1.0 {
            self.pulse_phase -= 1.0;
        }

        // Combine filtered noise and sine pulse
        let pulse_sample = (self.pulse_filter_state * 0.3 + sine_pulse * 0.2) * beat_strength;

        pulse_sample * 0.4 // Moderate volume for pulse
    }

    /// Generate ambient pad sample
    fn generate_pad_sample(&mut self, time: f64) -> f32 {
        let mut pad_sample = 0.0;

        // Update pad oscillator frequencies if needed
        if self.pad_oscillators.len() == self.pad_frequencies.len() {
            for (i, osc) in self.pad_oscillators.iter_mut().enumerate() {
                osc.set_frequency(self.pad_frequencies[i]);
                osc.update_envelope(if self.intensity > 0.0 { 1.0 } else { 0.0 });
                pad_sample += osc.generate_sample(self.sample_rate);
            }
        }

        pad_sample / self.pad_oscillators.len() as f32 // Average the oscillators
    }

    /// Generate arpeggio sample
    fn generate_arp_sample(&mut self, time: f64) -> f32 {
        let freq = Self::midi_to_frequency(self.current_arp_note);

        // Clean sine wave for arpeggios
        let arp_sample = (self.arp_phase * 2.0 * std::f32::consts::PI).sin() * 0.15;

        // Update phase
        self.arp_phase += freq / self.sample_rate;
        if self.arp_phase >= 1.0 {
            self.arp_phase -= 1.0;
        }

        arp_sample
    }

    /// Update beat timing and pattern
    fn update_beat_timing(&mut self, delta_time: f32) {
        self.beat_timer += delta_time;

        if self.beat_timer >= self.beat_duration {
            self.beat_timer = 0.0;
            self.beat_index = (self.beat_index + 1) % self.beat_pattern.len();
        }
    }

    /// Update pad chord progression
    fn update_pad_progression(&mut self, delta_time: f32) {
        self.pad_timer += delta_time;

        if self.pad_timer >= self.pad_chord_duration {
            self.pad_timer = 0.0;
            self.pad_chord_index = (self.pad_chord_index + 1) % self.pad_progression.len();
            self.pad_frequencies = Self::chord_to_frequencies(&self.pad_progression[self.pad_chord_index]);
        }
    }

    /// Update arpeggio progression
    fn update_arp_progression(&mut self, delta_time: f32) {
        self.arp_timer += delta_time;

        if self.arp_timer >= self.arp_note_duration {
            self.arp_timer = 0.0;

            // Generate next arpeggio note
            if let Some(next_note) = self.arp_chain.next(&mut self.rng) {
                self.current_arp_note = next_note;
            }

            // Vary note durations slightly for natural feel
            self.arp_note_duration = 0.2 + self.rng.gen::<f32>() * 0.1; // 0.2-0.3 seconds
        }
    }

    /// Update pattern evolution (every 4 minutes)
    fn update_pattern_evolution(&mut self, delta_time: f32) {
        self.pattern_timer += delta_time;

        if self.pattern_timer >= self.pattern_duration {
            self.pattern_timer = 0.0;
            self.current_pattern = (self.current_pattern + 1) % 4;

            // Evolve the musical elements based on pattern
            match self.current_pattern {
                0 => {
                    // Pattern 0: Moderate energy, steady rhythm
                    self.current_bpm = 120.0;
                    self.beat_pattern = vec![1.0, 0.3, 0.7, 0.3];
                    self.pulse_filter_cutoff = 800.0;
                },
                1 => {
                    // Pattern 1: Slightly more active
                    self.current_bpm = 125.0;
                    self.beat_pattern = vec![1.0, 0.4, 0.8, 0.4, 0.6, 0.3, 0.7, 0.3];
                    self.pulse_filter_cutoff = 1000.0;
                },
                2 => {
                    // Pattern 2: More complex rhythm
                    self.current_bpm = 130.0;
                    self.beat_pattern = vec![1.0, 0.2, 0.6, 0.4, 0.8, 0.3, 0.5, 0.4];
                    self.pulse_filter_cutoff = 1200.0;
                },
                3 => {
                    // Pattern 3: Return to simplicity
                    self.current_bpm = 115.0;
                    self.beat_pattern = vec![1.0, 0.4, 0.7, 0.4];
                    self.pulse_filter_cutoff = 700.0;
                },
                _ => {},
            }

            // Update beat duration based on new BPM
            self.beat_duration = 60.0 / self.current_bpm;
        }
    }

    /// Update volume envelopes
    fn update_envelopes(&mut self) {
        let target = if self.intensity > 0.0 { 1.0 } else { 0.0 };
        let rate = 0.002;

        self.beat_envelope += (target - self.beat_envelope) * rate;
        self.pad_envelope += (target - self.pad_envelope) * rate;
        self.arp_envelope += (target - self.arp_envelope) * rate;
    }
}

impl MoodGenerator for ActiveAmbientGenerator {
    fn generate_sample(&mut self, time: f64) -> f32 {
        self.time = time;

        if self.intensity <= 0.0 {
            return 0.0;
        }

        let delta_time = 1.0 / self.sample_rate;

        // Update all timing systems
        self.update_beat_timing(delta_time);
        self.update_pad_progression(delta_time);
        self.update_arp_progression(delta_time);
        self.update_pattern_evolution(delta_time);
        self.update_envelopes();

        // Generate all audio components
        let pulse_sample = self.generate_pulse_sample(time) * self.beat_envelope;
        let pad_sample = self.generate_pad_sample(time) * self.pad_envelope;
        let arp_sample = self.generate_arp_sample(time) * self.arp_envelope;

        // Mix components with appropriate balance
        let mixed = pulse_sample * 0.4 + pad_sample * 0.5 + arp_sample * 0.3;

        // Apply intensity scaling
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
        self.beat_index = 0;
        self.beat_timer = 0.0;
        self.pad_chord_index = 0;
        self.pad_timer = 0.0;
        self.arp_timer = 0.0;
        self.pattern_timer = 0.0;
        self.current_pattern = 0;

        self.pulse_phase = 0.0;
        self.pulse_filter_state = 0.0;
        self.arp_phase = 0.0;

        self.beat_envelope = 0.0;
        self.pad_envelope = 0.0;
        self.arp_envelope = 0.0;

        // Reset to initial state
        self.current_bpm = 120.0;
        self.beat_duration = 60.0 / self.current_bpm;
        self.beat_pattern = vec![1.0, 0.3, 0.7, 0.3];
        self.pulse_filter_cutoff = 800.0;

        // Reset pad oscillators
        for osc in &mut self.pad_oscillators {
            osc.phase = 0.0;
            osc.filter_state = 0.0;
            osc.envelope = 0.0;
        }

        // Reset chord progression
        self.pad_frequencies = Self::chord_to_frequencies(&self.pad_progression[0]);
        self.current_arp_note = 72;
    }

    fn update_focus_parameters(&mut self) {
        // When active ambient is dominant, make patterns more regular
        self.pad_chord_duration = 6.0 + self.rng.gen::<f32>() * 4.0; // 6-10 seconds
        self.arp_note_duration = 0.15 + self.rng.gen::<f32>() * 0.15; // 0.15-0.3 seconds
    }

    fn get_state(&self) -> GeneratorState {
        let pattern_names = ["Steady", "Active", "Complex", "Simple"];
        let current_pattern_name = pattern_names.get(self.current_pattern)
            .unwrap_or(&"Unknown")
            .to_string();

        GeneratorState {
            name: "Active Ambient".to_string(),
            intensity: self.intensity,
            is_active: self.intensity > 0.0,
            current_pattern: format!("{} - {} BPM", current_pattern_name, self.current_bpm),
            pattern_progress: self.pattern_timer / self.pattern_duration,
            cpu_usage: 0.25, // Higher CPU usage due to multiple synthesis components
        }
    }
}