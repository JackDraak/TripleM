use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{MoodGenerator, GeneratorState};
use crate::patterns::markov::MarkovChain;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// EDM-style music generator (0.75-1.0 mood range)
/// Generates high-energy electronic dance music with beats, drops, and synthesis
#[derive(Debug)]
pub struct EdmStyleGenerator {
    intensity: f32,
    sample_rate: f32,
    time: f64,

    // Beat and rhythm section
    current_bpm: f32,
    beat_timer: f32,
    beat_duration: f32,              // Duration per beat in seconds
    beat_count: usize,               // Current beat in measure
    measure_count: usize,            // Current measure (for buildups/drops)

    // Kick drum synthesis
    kick_phase: f32,
    kick_envelope: f32,
    kick_filter_state: f32,

    // Hi-hat synthesis
    hihat_timer: f32,
    hihat_duration: f32,
    hihat_envelope: f32,
    hihat_noise_state: f32,

    // Bass synthesis (wobble bass)
    bass_oscillators: Vec<BassOscillator>,
    bass_filter_cutoff: f32,
    bass_filter_state: f32,
    bass_lfo_phase: f32,
    bass_note_timer: f32,
    bass_note_duration: f32,
    current_bass_note: u8,

    // Lead synthesis (supersaw)
    lead_oscillators: Vec<LeadOscillator>,
    lead_chord_progression: Vec<Vec<u8>>,
    lead_chord_index: usize,
    lead_chord_timer: f32,
    lead_chord_duration: f32,
    lead_filter_cutoff: f32,
    lead_filter_state: f32,

    // Drop/buildup system (3-minute cycles)
    drop_timer: f32,
    drop_cycle_duration: f32,        // 3 minutes = 180 seconds
    current_section: DropSection,
    buildup_intensity: f32,

    // Bass note pattern (Markov chain)
    bass_chain: MarkovChain<u8>,

    // Random number generator
    rng: StdRng,

    // Volume envelopes for mixing
    kick_vol_envelope: f32,
    hihat_vol_envelope: f32,
    bass_vol_envelope: f32,
    lead_vol_envelope: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DropSection {
    Intro,     // 0-45s: Building energy
    Buildup,   // 45-90s: Tension building
    Drop,      // 90-135s: Full energy release
    Breakdown, // 135-180s: Cool down, prepare for next cycle
}

#[derive(Debug, Clone)]
struct BassOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    detune: f32,
}

impl BassOscillator {
    fn new(frequency: f32, detune: f32) -> Self {
        Self {
            phase: 0.0,
            frequency,
            amplitude: 0.8,
            detune,
        }
    }

    fn generate_sample(&mut self, sample_rate: f32, lfo_mod: f32) -> f32 {
        let freq = self.frequency * (1.0 + self.detune + lfo_mod * 0.1);
        let phase_increment = freq / sample_rate;
        self.phase += phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Sawtooth wave for aggressive bass
        (self.phase * 2.0 - 1.0) * self.amplitude
    }
}

#[derive(Debug, Clone)]
struct LeadOscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
    detune: f32,
}

impl LeadOscillator {
    fn new(frequency: f32, detune: f32) -> Self {
        Self {
            phase: 0.0,
            frequency,
            amplitude: 0.4,
            detune,
        }
    }

    fn generate_sample(&mut self, sample_rate: f32) -> f32 {
        let freq = self.frequency * (1.0 + self.detune);
        let phase_increment = freq / sample_rate;
        self.phase += phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Supersaw: detuned sawtooth waves
        (self.phase * 2.0 - 1.0) * self.amplitude
    }
}

impl EdmStyleGenerator {
    pub fn new(config: &MoodConfig) -> Result<Self> {
        let mut rng = StdRng::from_entropy();

        // Start with 128 BPM (classic house/EDM tempo)
        let current_bpm = 128.0;
        let beat_duration = 60.0 / current_bpm;

        // Initialize bass oscillators (3 detuned oscillators for thickness)
        let bass_oscillators = vec![
            BassOscillator::new(65.41, 0.0),    // C2
            BassOscillator::new(65.41, -0.01),  // Slightly detuned
            BassOscillator::new(65.41, 0.01),   // Slightly detuned
        ];

        // Initialize lead oscillators (supersaw: 7 detuned oscillators)
        let lead_oscillators = vec![
            LeadOscillator::new(261.63, 0.0),    // C4
            LeadOscillator::new(261.63, -0.02),
            LeadOscillator::new(261.63, 0.02),
            LeadOscillator::new(261.63, -0.04),
            LeadOscillator::new(261.63, 0.04),
            LeadOscillator::new(261.63, -0.06),
            LeadOscillator::new(261.63, 0.06),
        ];

        // Create EDM chord progression
        let lead_chord_progression = Self::create_edm_chord_progression();

        // Initialize bass pattern chain
        let bass_chain = Self::create_bass_chain();

        Ok(Self {
            intensity: 0.0,
            sample_rate: config.sample_rate as f32,
            time: 0.0,

            current_bpm,
            beat_timer: 0.0,
            beat_duration,
            beat_count: 0,
            measure_count: 0,

            kick_phase: 0.0,
            kick_envelope: 0.0,
            kick_filter_state: 0.0,

            hihat_timer: 0.0,
            hihat_duration: beat_duration * 0.25, // 16th notes
            hihat_envelope: 0.0,
            hihat_noise_state: 0.0,

            bass_oscillators,
            bass_filter_cutoff: 800.0,
            bass_filter_state: 0.0,
            bass_lfo_phase: 0.0,
            bass_note_timer: 0.0,
            bass_note_duration: beat_duration * 2.0, // Change every 2 beats
            current_bass_note: 36, // C1

            lead_oscillators,
            lead_chord_progression,
            lead_chord_index: 0,
            lead_chord_timer: 0.0,
            lead_chord_duration: beat_duration * 8.0, // Change every 2 measures
            lead_filter_cutoff: 2000.0,
            lead_filter_state: 0.0,

            drop_timer: 0.0,
            drop_cycle_duration: 180.0, // 3 minutes
            current_section: DropSection::Intro,
            buildup_intensity: 0.0,

            bass_chain,
            rng,

            kick_vol_envelope: 0.0,
            hihat_vol_envelope: 0.0,
            bass_vol_envelope: 0.0,
            lead_vol_envelope: 0.0,
        })
    }

    /// Create EDM chord progression (trance/house style)
    fn create_edm_chord_progression() -> Vec<Vec<u8>> {
        vec![
            vec![60, 64, 67],        // C major
            vec![57, 60, 64],        // A minor
            vec![65, 69, 72],        // F major
            vec![62, 65, 69],        // D minor
            vec![67, 71, 74],        // G major
            vec![64, 67, 71],        // E minor
            vec![65, 69, 72],        // F major
            vec![67, 71, 74],        // G major
        ]
    }

    /// Create bass pattern Markov chain
    fn create_bass_chain() -> MarkovChain<u8> {
        let mut chain = MarkovChain::new(2);

        // EDM bass patterns (root and fifth mostly)
        let patterns = vec![
            vec![36, 36, 43, 36, 36, 43, 36, 43],    // C1-G1 pattern
            vec![36, 41, 36, 43, 36, 41, 36, 43],    // Adding F1
            vec![38, 38, 45, 38, 38, 45, 38, 45],    // D1-A1 pattern
            vec![41, 41, 48, 41, 41, 48, 41, 48],    // F1-C2 pattern
            vec![43, 43, 50, 43, 43, 50, 43, 50],    // G1-D2 pattern
        ];

        for pattern in patterns {
            chain.add_sequence(&pattern);
        }

        chain.normalize();
        chain
    }

    /// Convert MIDI note to frequency
    fn midi_to_frequency(midi_note: u8) -> f32 {
        440.0 * 2.0_f32.powf((midi_note as f32 - 69.0) / 12.0)
    }

    /// Generate kick drum sample
    fn generate_kick_sample(&mut self, _time: f64) -> f32 {
        // Kick on every beat
        let kick_trigger = self.beat_count % 4 == 0; // 4/4 kick pattern

        if kick_trigger && self.beat_timer < 0.01 {
            self.kick_envelope = 1.0; // Trigger kick
        }

        // Generate kick sound (sine wave with pitch envelope + click)
        let kick_freq = 60.0 * (1.0 + self.kick_envelope * 2.0); // Pitch drops
        let kick_wave = (self.kick_phase * 2.0 * std::f32::consts::PI).sin();
        self.kick_phase += kick_freq / self.sample_rate;
        if self.kick_phase >= 1.0 {
            self.kick_phase -= 1.0;
        }

        // Add click for punch
        let click = if self.kick_envelope > 0.9 { 0.3 } else { 0.0 };

        // Apply envelope
        let kick_sample = (kick_wave + click) * self.kick_envelope;

        // Decay envelope
        self.kick_envelope *= 0.9995; // Fast decay

        kick_sample * 0.8
    }

    /// Generate hi-hat sample
    fn generate_hihat_sample(&mut self, _time: f64) -> f32 {
        self.hihat_timer += 1.0 / self.sample_rate;

        if self.hihat_timer >= self.hihat_duration {
            self.hihat_timer = 0.0;
            self.hihat_envelope = 1.0; // Trigger hi-hat
        }

        // Generate hi-hat (filtered noise)
        let noise = (self.rng.gen::<f32>() - 0.5) * 2.0;
        let alpha = 6000.0 / self.sample_rate; // High-pass effect
        self.hihat_noise_state += (noise - self.hihat_noise_state) * alpha;

        let hihat_sample = self.hihat_noise_state * self.hihat_envelope;

        // Decay envelope
        self.hihat_envelope *= 0.999; // Fast decay

        hihat_sample * 0.3
    }

    /// Generate bass sample (wobble bass)
    fn generate_bass_sample(&mut self, _time: f64) -> f32 {
        // Update LFO for wobble effect
        let lfo_freq = match self.current_section {
            DropSection::Intro => 2.0,
            DropSection::Buildup => 4.0,
            DropSection::Drop => 8.0,
            DropSection::Breakdown => 1.0,
        };

        self.bass_lfo_phase += lfo_freq / self.sample_rate;
        if self.bass_lfo_phase >= 1.0 {
            self.bass_lfo_phase -= 1.0;
        }

        let lfo_mod = (self.bass_lfo_phase * 2.0 * std::f32::consts::PI).sin();

        // Generate bass oscillators
        let mut bass_sample = 0.0;
        for osc in &mut self.bass_oscillators {
            bass_sample += osc.generate_sample(self.sample_rate, lfo_mod);
        }
        bass_sample /= self.bass_oscillators.len() as f32;

        // Apply filter with LFO modulation
        let cutoff = self.bass_filter_cutoff * (1.0 + lfo_mod * 0.5);
        let alpha = cutoff / self.sample_rate;
        self.bass_filter_state += (bass_sample - self.bass_filter_state) * alpha.min(1.0);

        self.bass_filter_state * 0.6
    }

    /// Generate lead sample (supersaw)
    fn generate_lead_sample(&mut self, _time: f64) -> f32 {
        let mut lead_sample = 0.0;
        for osc in &mut self.lead_oscillators {
            lead_sample += osc.generate_sample(self.sample_rate);
        }
        lead_sample /= self.lead_oscillators.len() as f32;

        // Apply filter
        let alpha = self.lead_filter_cutoff / self.sample_rate;
        self.lead_filter_state += (lead_sample - self.lead_filter_state) * alpha.min(1.0);

        self.lead_filter_state * 0.4
    }

    /// Update beat timing and patterns
    fn update_beat_timing(&mut self, delta_time: f32) {
        self.beat_timer += delta_time;

        if self.beat_timer >= self.beat_duration {
            self.beat_timer = 0.0;
            self.beat_count += 1;

            if self.beat_count % 4 == 0 {
                self.measure_count += 1;
            }
        }
    }

    /// Update bass note progression
    fn update_bass_progression(&mut self, delta_time: f32) {
        self.bass_note_timer += delta_time;

        if self.bass_note_timer >= self.bass_note_duration {
            self.bass_note_timer = 0.0;

            // Generate next bass note
            if let Some(next_note) = self.bass_chain.next(&mut self.rng) {
                self.current_bass_note = next_note;

                // Update bass oscillator frequencies
                let freq = Self::midi_to_frequency(self.current_bass_note);
                for osc in &mut self.bass_oscillators {
                    osc.frequency = freq;
                }
            }
        }
    }

    /// Update lead chord progression
    fn update_lead_progression(&mut self, delta_time: f32) {
        self.lead_chord_timer += delta_time;

        if self.lead_chord_timer >= self.lead_chord_duration {
            self.lead_chord_timer = 0.0;
            self.lead_chord_index = (self.lead_chord_index + 1) % self.lead_chord_progression.len();

            // Update lead oscillator frequencies to chord tones
            let chord = &self.lead_chord_progression[self.lead_chord_index];
            for (i, osc) in self.lead_oscillators.iter_mut().enumerate() {
                let note_index = i % chord.len();
                let octave_shift = i / chord.len();
                let midi_note = chord[note_index] + (octave_shift * 12) as u8;
                osc.frequency = Self::midi_to_frequency(midi_note);
            }
        }
    }

    /// Update drop cycle and section management
    fn update_drop_cycle(&mut self, delta_time: f32) {
        self.drop_timer += delta_time;

        let section_duration = self.drop_cycle_duration / 4.0; // 45 seconds per section
        let section_index = (self.drop_timer / section_duration) as usize % 4;

        let new_section = match section_index {
            0 => DropSection::Intro,
            1 => DropSection::Buildup,
            2 => DropSection::Drop,
            3 => DropSection::Breakdown,
            _ => DropSection::Intro,
        };

        if new_section != self.current_section {
            self.current_section = new_section;
            self.update_section_parameters();
        }

        // Update buildup intensity
        match self.current_section {
            DropSection::Intro => {
                self.buildup_intensity = (self.drop_timer % section_duration) / section_duration * 0.5;
            },
            DropSection::Buildup => {
                self.buildup_intensity = 0.5 + (self.drop_timer % section_duration) / section_duration * 0.5;
            },
            DropSection::Drop => {
                self.buildup_intensity = 1.0;
            },
            DropSection::Breakdown => {
                self.buildup_intensity = 1.0 - (self.drop_timer % section_duration) / section_duration * 0.7;
            },
        }

        if self.drop_timer >= self.drop_cycle_duration {
            self.drop_timer = 0.0;
        }
    }

    /// Update parameters based on current section
    fn update_section_parameters(&mut self) {
        match self.current_section {
            DropSection::Intro => {
                self.current_bpm = 128.0;
                self.bass_filter_cutoff = 600.0;
                self.lead_filter_cutoff = 1500.0;
            },
            DropSection::Buildup => {
                self.current_bpm = 128.0;
                self.bass_filter_cutoff = 800.0;
                self.lead_filter_cutoff = 2000.0;
            },
            DropSection::Drop => {
                self.current_bpm = 128.0;
                self.bass_filter_cutoff = 1200.0;
                self.lead_filter_cutoff = 3000.0;
            },
            DropSection::Breakdown => {
                self.current_bpm = 128.0;
                self.bass_filter_cutoff = 400.0;
                self.lead_filter_cutoff = 1000.0;
            },
        }

        self.beat_duration = 60.0 / self.current_bpm;
    }

    /// Update volume envelopes for section mixing
    fn update_envelopes(&mut self) {
        let target = if self.intensity > 0.0 { 1.0 } else { 0.0 };
        let rate = 0.002;

        // Section-based volume scaling
        let (kick_target, hihat_target, bass_target, lead_target) = match self.current_section {
            DropSection::Intro => (0.6, 0.3, 0.4, 0.2),
            DropSection::Buildup => (0.8, 0.5, 0.6, 0.4),
            DropSection::Drop => (1.0, 0.8, 1.0, 0.8),
            DropSection::Breakdown => (0.4, 0.2, 0.3, 0.6),
        };

        self.kick_vol_envelope += (target * kick_target - self.kick_vol_envelope) * rate;
        self.hihat_vol_envelope += (target * hihat_target - self.hihat_vol_envelope) * rate;
        self.bass_vol_envelope += (target * bass_target - self.bass_vol_envelope) * rate;
        self.lead_vol_envelope += (target * lead_target - self.lead_vol_envelope) * rate;
    }
}

impl MoodGenerator for EdmStyleGenerator {
    fn generate_sample(&mut self, time: f64) -> f32 {
        self.time = time;

        if self.intensity <= 0.0 {
            return 0.0;
        }

        let delta_time = 1.0 / self.sample_rate;

        // Update all timing systems
        self.update_beat_timing(delta_time);
        self.update_bass_progression(delta_time);
        self.update_lead_progression(delta_time);
        self.update_drop_cycle(delta_time);
        self.update_envelopes();

        // Generate all components
        let kick_sample = self.generate_kick_sample(time) * self.kick_vol_envelope;
        let hihat_sample = self.generate_hihat_sample(time) * self.hihat_vol_envelope;
        let bass_sample = self.generate_bass_sample(time) * self.bass_vol_envelope;
        let lead_sample = self.generate_lead_sample(time) * self.lead_vol_envelope;

        // Mix all components
        let mixed = kick_sample + hihat_sample + bass_sample + lead_sample;

        // Apply intensity and buildup intensity
        let output = mixed * self.intensity * (0.3 + 0.7 * self.buildup_intensity);
        output.clamp(-0.9, 0.9) // Aggressive limiting for EDM
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
        self.beat_timer = 0.0;
        self.beat_count = 0;
        self.measure_count = 0;
        self.drop_timer = 0.0;
        self.current_section = DropSection::Intro;
        self.buildup_intensity = 0.0;

        self.kick_phase = 0.0;
        self.kick_envelope = 0.0;
        self.kick_filter_state = 0.0;

        self.hihat_timer = 0.0;
        self.hihat_envelope = 0.0;
        self.hihat_noise_state = 0.0;

        self.bass_lfo_phase = 0.0;
        self.bass_note_timer = 0.0;
        self.bass_filter_state = 0.0;

        self.lead_chord_timer = 0.0;
        self.lead_chord_index = 0;
        self.lead_filter_state = 0.0;

        self.kick_vol_envelope = 0.0;
        self.hihat_vol_envelope = 0.0;
        self.bass_vol_envelope = 0.0;
        self.lead_vol_envelope = 0.0;

        // Reset oscillators
        for osc in &mut self.bass_oscillators {
            osc.phase = 0.0;
        }
        for osc in &mut self.lead_oscillators {
            osc.phase = 0.0;
        }

        self.update_section_parameters();
    }

    fn update_focus_parameters(&mut self) {
        // When EDM is dominant, increase tempo variation
        self.current_bpm = 126.0 + self.rng.gen::<f32>() * 6.0; // 126-132 BPM
        self.beat_duration = 60.0 / self.current_bpm;
    }

    fn get_state(&self) -> GeneratorState {
        let section_name = match self.current_section {
            DropSection::Intro => "Intro",
            DropSection::Buildup => "Buildup",
            DropSection::Drop => "DROP!",
            DropSection::Breakdown => "Breakdown",
        };

        GeneratorState {
            name: "EDM Style".to_string(),
            intensity: self.intensity,
            is_active: self.intensity > 0.0,
            current_pattern: format!("{} - {:.0} BPM - {:.0}%",
                section_name, self.current_bpm, self.buildup_intensity * 100.0),
            pattern_progress: self.drop_timer / self.drop_cycle_duration,
            cpu_usage: 0.35, // Highest CPU usage due to complex synthesis
        }
    }
}