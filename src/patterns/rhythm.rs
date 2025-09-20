use crate::patterns::{PatternGenerator, PatternParameters};
use crate::audio::NaturalVariation;
use crate::error::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::VecDeque;

/// Enhanced rhythm pattern with micro-timing and dynamics
#[derive(Debug, Clone)]
pub struct RhythmPattern {
    pub kick: bool,
    pub snare: bool,
    pub hihat: bool,
    pub percussion: bool,
    pub intensity: f32,

    // Enhanced timing information
    pub micro_timing: f32,      // Subtle timing offset (-0.1 to 0.1)
    pub velocity: f32,          // Dynamic velocity (0.0 to 1.0)
    pub groove_swing: f32,      // Groove swing amount
    pub accent: bool,           // Accent marker for emphasis
}

impl Default for RhythmPattern {
    fn default() -> Self {
        Self {
            kick: false,
            snare: false,
            hihat: false,
            percussion: false,
            intensity: 0.0,
            micro_timing: 0.0,
            velocity: 0.7,
            groove_swing: 0.0,
            accent: false,
        }
    }
}

/// Multi-scale rhythm evolution system
pub struct RhythmGenerator {
    parameters: PatternParameters,

    // Multi-scale timing
    micro_scale: MicroRhythmEvolution,
    beat_scale: BeatRhythmEvolution,
    phrase_scale: PhraseRhythmEvolution,
    section_scale: SectionRhythmEvolution,

    // Pattern state
    current_pattern: RhythmPattern,
    step_position: usize,
    beat_position: usize,
    phrase_position: usize,
    section_position: usize,

    // Pattern memory for evolution
    pattern_memory: PatternMemory,

    // Natural variation integration
    variation: NaturalVariation,

    // Euclidean rhythm generators
    euclidean_generators: Vec<EuclideanRhythm>,

    sample_rate: f32,
}

/// Micro-scale rhythm evolution (sub-beat timing variations)
#[derive(Debug, Clone)]
pub struct MicroRhythmEvolution {
    groove_template: GrooveTemplate,
    micro_timing_drift: f32,
    swing_amount: f32,
    humanization_level: f32,
}

/// Beat-scale rhythm evolution (within-measure variations)
#[derive(Debug, Clone)]
pub struct BeatRhythmEvolution {
    base_patterns: Vec<RhythmicCell>,
    current_cell_index: usize,
    variation_probability: f32,
    fill_probability: f32,
    accent_patterns: Vec<AccentPattern>,
}

/// Phrase-scale rhythm evolution (across measures)
#[derive(Debug, Clone)]
pub struct PhraseRhythmEvolution {
    phrase_length: usize,        // Length in measures
    phrase_templates: Vec<PhraseTemplate>,
    current_template: usize,
    evolution_rate: f32,
    tension_curve: TensionCurve,
}

/// Section-scale rhythm evolution (long-term changes)
#[derive(Debug, Clone)]
pub struct SectionRhythmEvolution {
    section_length: usize,       // Length in phrases
    evolution_trajectory: EvolutionTrajectory,
    complexity_envelope: ComplexityEnvelope,
    pattern_morphing: PatternMorphing,
}

/// Groove template for micro-timing
#[derive(Debug, Clone)]
pub struct GrooveTemplate {
    name: String,
    timing_offsets: Vec<f32>,    // Per-step timing offsets
    velocity_map: Vec<f32>,      // Per-step velocity multipliers
    swing_style: SwingStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum SwingStyle {
    None,
    Light,      // Subtle swing
    Medium,     // Standard swing
    Heavy,      // Pronounced swing
    Shuffle,    // Shuffle feel
    Latin,      // Latin rhythm feel
}

/// Rhythmic cell - basic building block
#[derive(Debug, Clone)]
pub struct RhythmicCell {
    pattern: Vec<RhythmPattern>,
    weight: f32,                 // Probability weight
    energy_level: f32,           // 0.0 = low energy, 1.0 = high energy
    complexity: f32,             // 0.0 = simple, 1.0 = complex
}

/// Accent pattern for emphasis
#[derive(Debug, Clone)]
pub struct AccentPattern {
    accents: Vec<bool>,          // Which beats to accent
    intensity: f32,              // Accent strength
}

/// Phrase template defining structure
#[derive(Debug, Clone)]
pub struct PhraseTemplate {
    name: String,
    measure_types: Vec<MeasureType>,
    intro_probability: f32,
    fill_positions: Vec<usize>,
    energy_arc: Vec<f32>,        // Energy curve across phrase
}

#[derive(Debug, Clone, Copy)]
pub enum MeasureType {
    Standard,    // Regular pattern
    Variant,     // Slight variation
    Fill,        // Drum fill
    Break,       // Rhythmic break
    Build,       // Building intensity
}

/// Tension curve for musical phrasing
#[derive(Debug, Clone)]
pub struct TensionCurve {
    curve_points: Vec<f32>,      // Tension values (0.0 to 1.0)
    resolution_points: Vec<usize>, // Where tension resolves
}

/// Long-term evolution trajectory
#[derive(Debug, Clone)]
pub struct EvolutionTrajectory {
    complexity_path: Vec<f32>,   // How complexity changes over time
    energy_path: Vec<f32>,       // How energy changes over time
    pattern_mutation_rate: f32,  // How fast patterns evolve
}

/// Complexity envelope over time
#[derive(Debug, Clone)]
pub struct ComplexityEnvelope {
    stages: Vec<ComplexityStage>,
    current_stage: usize,
    stage_progress: f32,
}

#[derive(Debug, Clone)]
pub struct ComplexityStage {
    target_complexity: f32,
    duration: f32,               // In phrases
    transition_style: TransitionStyle,
}

#[derive(Debug, Clone, Copy)]
pub enum TransitionStyle {
    Gradual,     // Smooth transition
    Stepped,     // Sudden changes
    Organic,     // Natural evolution
}

/// Pattern morphing system
#[derive(Debug, Clone)]
pub struct PatternMorphing {
    source_pattern: Vec<RhythmPattern>,
    target_pattern: Vec<RhythmPattern>,
    morph_position: f32,         // 0.0 = source, 1.0 = target
    morph_rate: f32,
}

/// Pattern memory system to avoid repetition
#[derive(Debug, Clone)]
pub struct PatternMemory {
    recent_patterns: VecDeque<Vec<RhythmPattern>>,
    memory_depth: usize,
    repetition_avoidance: f32,   // How much to avoid repetition
}

/// Euclidean rhythm generator
#[derive(Debug, Clone)]
pub struct EuclideanRhythm {
    steps: usize,                // Total steps
    pulses: usize,               // Number of pulses
    rotation: usize,             // Rotation offset
    instrument: RhythmInstrument,
    pattern: Vec<bool>,
}

#[derive(Debug, Clone, Copy)]
pub enum RhythmInstrument {
    Kick,
    Snare,
    HiHat,
    Percussion,
}

impl RhythmGenerator {
    pub fn new(parameters: &PatternParameters) -> Result<Self> {
        let variation = NaturalVariation::new(None);
        let sample_rate = 44100.0; // Default sample rate

        Ok(Self {
            parameters: parameters.clone(),

            micro_scale: MicroRhythmEvolution::new(),
            beat_scale: BeatRhythmEvolution::new(),
            phrase_scale: PhraseRhythmEvolution::new(),
            section_scale: SectionRhythmEvolution::new(),

            current_pattern: RhythmPattern::default(),
            step_position: 0,
            beat_position: 0,
            phrase_position: 0,
            section_position: 0,

            pattern_memory: PatternMemory::new(),
            variation,

            euclidean_generators: vec![
                EuclideanRhythm::new(16, 4, 0, RhythmInstrument::Kick),     // 4-on-the-floor
                EuclideanRhythm::new(16, 3, 8, RhythmInstrument::Snare),    // Offset snare
                EuclideanRhythm::new(16, 11, 0, RhythmInstrument::HiHat),   // Complex hihat
            ],

            sample_rate,
        })
    }

    /// Generate next rhythm pattern with multi-scale evolution
    fn generate_evolved_pattern(&mut self) -> RhythmPattern {
        // Update all evolution scales
        self.micro_scale.update();
        self.beat_scale.update(&self.parameters);
        self.phrase_scale.update();
        self.section_scale.update();

        // Generate base pattern from current beat-scale cell
        let mut pattern = self.beat_scale.generate_pattern(
            self.step_position,
            &self.parameters
        );

        // Apply phrase-scale modifications
        pattern = self.phrase_scale.modify_pattern(
            pattern,
            self.phrase_position,
            self.beat_position
        );

        // Apply section-scale evolution
        pattern = self.section_scale.evolve_pattern(
            pattern,
            self.section_position
        );

        // Add Euclidean rhythm contributions
        for euclidean in &mut self.euclidean_generators {
            euclidean.update();
            pattern = euclidean.blend_with_pattern(pattern, &self.parameters);
        }

        // Apply micro-scale groove and timing
        pattern = self.micro_scale.apply_groove(
            pattern,
            self.step_position,
            &self.variation
        );

        // Check against pattern memory and adjust if too repetitive
        pattern = self.pattern_memory.avoid_repetition(pattern);

        pattern
    }
}

impl PatternGenerator for RhythmGenerator {
    type Output = RhythmPattern;

    fn next(&mut self) -> Self::Output {
        // Update natural variation
        self.variation.update();

        // Generate evolved pattern
        self.current_pattern = self.generate_evolved_pattern();

        // Update positions across all scales
        self.step_position += 1;

        if self.step_position % 4 == 0 {
            self.beat_position += 1;
        }

        if self.beat_position % 16 == 0 {  // 4 beats per measure, 4 measures per phrase
            self.phrase_position += 1;
            self.beat_position = 0;
        }

        if self.phrase_position % 8 == 0 { // 8 phrases per section
            self.section_position += 1;
            self.phrase_position = 0;
        }

        self.current_pattern.clone()
    }

    fn reset(&mut self) {
        self.step_position = 0;
        self.beat_position = 0;
        self.phrase_position = 0;
        self.section_position = 0;
        self.pattern_memory.clear();
    }

    fn update_parameters(&mut self, params: &PatternParameters) {
        self.parameters = params.clone();

        // Update evolution systems based on new parameters
        self.beat_scale.adapt_to_parameters(params);
        self.phrase_scale.adapt_to_parameters(params);
        self.section_scale.adapt_to_parameters(params);
    }

    fn pattern_length(&self) -> usize {
        16 // 16 steps per pattern
    }

    fn is_cycle_complete(&self) -> bool {
        self.step_position % 16 == 0
    }
}

// Implementation of all the evolution components follows...
impl MicroRhythmEvolution {
    fn new() -> Self {
        Self {
            groove_template: GrooveTemplate::standard_groove(),
            micro_timing_drift: 0.0,
            swing_amount: 0.1,
            humanization_level: 0.3,
        }
    }

    fn update(&mut self) {
        // Update micro-timing drift slowly
        self.micro_timing_drift += (rand::random::<f32>() - 0.5) * 0.001;
        self.micro_timing_drift = self.micro_timing_drift.clamp(-0.05, 0.05);
    }

    fn apply_groove(&self, mut pattern: RhythmPattern, step: usize, variation: &NaturalVariation) -> RhythmPattern {
        // Apply groove timing
        let groove_offset = self.groove_template.get_timing_offset(step);
        let variation_offset = variation.get_timing_variation() * 0.02;
        pattern.micro_timing = groove_offset + self.micro_timing_drift + variation_offset;

        // Apply groove velocity
        let groove_velocity = self.groove_template.get_velocity_multiplier(step);
        let variation_velocity = variation.get_amplitude_variation();
        pattern.velocity *= groove_velocity * (1.0 + variation_velocity * 0.1);

        // Apply swing
        if step % 2 == 1 {
            pattern.micro_timing += self.swing_amount * 0.1;
        }

        pattern
    }
}

impl BeatRhythmEvolution {
    fn new() -> Self {
        Self {
            base_patterns: RhythmicCell::create_standard_cells(),
            current_cell_index: 0,
            variation_probability: 0.2,
            fill_probability: 0.05,
            accent_patterns: AccentPattern::create_standard_accents(),
        }
    }

    fn update(&mut self, params: &PatternParameters) {
        // Occasionally switch to different rhythmic cell
        if rand::random::<f32>() < self.variation_probability * params.variation {
            self.current_cell_index = rand::random::<usize>() % self.base_patterns.len();
        }
    }

    fn generate_pattern(&self, step: usize, params: &PatternParameters) -> RhythmPattern {
        let cell = &self.base_patterns[self.current_cell_index];
        let pattern_index = step % cell.pattern.len();
        let mut pattern = cell.pattern[pattern_index].clone();

        // Apply intensity scaling
        pattern.intensity = params.intensity * cell.energy_level;

        // Apply complexity-based modifications
        if params.complexity > 0.5 && rand::random::<f32>() < params.complexity {
            pattern = self.add_complexity_elements(pattern);
        }

        pattern
    }

    fn add_complexity_elements(&self, mut pattern: RhythmPattern) -> RhythmPattern {
        // Add percussion hits for complexity
        if rand::random::<f32>() < 0.3 {
            pattern.percussion = true;
        }

        // Add occasional accent
        if rand::random::<f32>() < 0.2 {
            pattern.accent = true;
            pattern.velocity *= 1.3;
        }

        pattern
    }

    fn adapt_to_parameters(&mut self, params: &PatternParameters) {
        self.variation_probability = params.variation * 0.5;
        self.fill_probability = params.complexity * 0.1;
    }
}

impl PhraseRhythmEvolution {
    fn new() -> Self {
        Self {
            phrase_length: 4, // 4 measures per phrase
            phrase_templates: PhraseTemplate::create_templates(),
            current_template: 0,
            evolution_rate: 0.1,
            tension_curve: TensionCurve::standard_curve(),
        }
    }

    fn update(&mut self) {
        // Evolve phrase template occasionally
        if rand::random::<f32>() < self.evolution_rate {
            self.current_template = (self.current_template + 1) % self.phrase_templates.len();
        }
    }

    fn modify_pattern(&self, mut pattern: RhythmPattern, phrase_pos: usize, beat_pos: usize) -> RhythmPattern {
        let template = &self.phrase_templates[self.current_template];
        let measure_in_phrase = beat_pos / 4;

        if let Some(measure_type) = template.measure_types.get(measure_in_phrase) {
            pattern = self.apply_measure_type(pattern, *measure_type);
        }

        // Apply tension curve
        let tension = self.tension_curve.get_tension_at_position(phrase_pos, beat_pos);
        pattern.intensity *= 1.0 + tension * 0.3;

        pattern
    }

    fn apply_measure_type(&self, mut pattern: RhythmPattern, measure_type: MeasureType) -> RhythmPattern {
        match measure_type {
            MeasureType::Fill => {
                pattern.percussion = true;
                pattern.velocity *= 1.2;
            },
            MeasureType::Break => {
                pattern.kick = false;
                pattern.intensity *= 0.5;
            },
            MeasureType::Build => {
                pattern.intensity *= 1.5;
                pattern.accent = true;
            },
            _ => {} // Standard and Variant handled elsewhere
        }
        pattern
    }

    fn adapt_to_parameters(&mut self, params: &PatternParameters) {
        self.evolution_rate = params.variation * 0.2;
    }
}

impl SectionRhythmEvolution {
    fn new() -> Self {
        Self {
            section_length: 8, // 8 phrases per section
            evolution_trajectory: EvolutionTrajectory::organic_growth(),
            complexity_envelope: ComplexityEnvelope::building_complexity(),
            pattern_morphing: PatternMorphing::new(),
        }
    }

    fn update(&mut self) {
        self.pattern_morphing.update();
        self.complexity_envelope.update();
    }

    fn evolve_pattern(&self, mut pattern: RhythmPattern, section_pos: usize) -> RhythmPattern {
        // Apply complexity envelope
        let complexity_factor = self.complexity_envelope.get_complexity_at_position(section_pos);
        pattern.intensity *= 1.0 + complexity_factor * 0.5;

        // Apply evolution trajectory
        let evolution_factor = self.evolution_trajectory.get_evolution_at_position(section_pos);
        if evolution_factor > 0.7 && rand::random::<f32>() < 0.1 {
            pattern = self.apply_evolutionary_mutation(pattern);
        }

        pattern
    }

    fn apply_evolutionary_mutation(&self, mut pattern: RhythmPattern) -> RhythmPattern {
        // Randomly mutate some aspect of the pattern
        match rand::random::<usize>() % 4 {
            0 => pattern.kick = !pattern.kick,
            1 => pattern.snare = !pattern.snare,
            2 => pattern.hihat = !pattern.hihat,
            3 => pattern.percussion = !pattern.percussion,
            _ => {}
        }
        pattern
    }

    fn adapt_to_parameters(&mut self, params: &PatternParameters) {
        self.evolution_trajectory.pattern_mutation_rate = params.variation;
        self.pattern_morphing.morph_rate = params.complexity * 0.1;
    }
}

// Additional implementation details for all supporting structures...
impl GrooveTemplate {
    fn standard_groove() -> Self {
        Self {
            name: "Standard".to_string(),
            timing_offsets: vec![0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02],
            velocity_map: vec![1.0, 0.8, 0.9, 0.8, 1.0, 0.8, 0.9, 0.8],
            swing_style: SwingStyle::Light,
        }
    }

    fn get_timing_offset(&self, step: usize) -> f32 {
        self.timing_offsets[step % self.timing_offsets.len()]
    }

    fn get_velocity_multiplier(&self, step: usize) -> f32 {
        self.velocity_map[step % self.velocity_map.len()]
    }
}

impl RhythmicCell {
    fn create_standard_cells() -> Vec<Self> {
        vec![
            // Basic 4/4 pattern
            Self {
                pattern: vec![
                    RhythmPattern { kick: true, snare: false, hihat: false, percussion: false, intensity: 1.0, micro_timing: 0.0, velocity: 1.0, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: false, snare: false, hihat: true, percussion: false, intensity: 0.6, micro_timing: 0.0, velocity: 0.8, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: false, snare: true, hihat: false, percussion: false, intensity: 0.9, micro_timing: 0.0, velocity: 0.9, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: false, snare: false, hihat: true, percussion: false, intensity: 0.6, micro_timing: 0.0, velocity: 0.7, groove_swing: 0.0, accent: false },
                ],
                weight: 1.0,
                energy_level: 0.7,
                complexity: 0.3,
            },
            // Syncopated pattern
            Self {
                pattern: vec![
                    RhythmPattern { kick: true, snare: false, hihat: true, percussion: false, intensity: 1.0, micro_timing: 0.0, velocity: 1.0, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: false, snare: false, hihat: false, percussion: true, intensity: 0.5, micro_timing: 0.0, velocity: 0.6, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: false, snare: true, hihat: true, percussion: false, intensity: 0.9, micro_timing: 0.0, velocity: 0.9, groove_swing: 0.0, accent: false },
                    RhythmPattern { kick: true, snare: false, hihat: false, percussion: false, intensity: 0.8, micro_timing: 0.0, velocity: 0.8, groove_swing: 0.0, accent: false },
                ],
                weight: 0.7,
                energy_level: 0.8,
                complexity: 0.6,
            },
        ]
    }
}

impl AccentPattern {
    fn create_standard_accents() -> Vec<Self> {
        vec![
            Self { accents: vec![true, false, false, false], intensity: 1.2 },
            Self { accents: vec![true, false, true, false], intensity: 1.1 },
            Self { accents: vec![true, false, false, true], intensity: 1.3 },
        ]
    }
}

impl PhraseTemplate {
    fn create_templates() -> Vec<Self> {
        vec![
            Self {
                name: "Standard".to_string(),
                measure_types: vec![MeasureType::Standard, MeasureType::Standard, MeasureType::Variant, MeasureType::Fill],
                intro_probability: 0.1,
                fill_positions: vec![3],
                energy_arc: vec![0.7, 0.8, 0.9, 1.0],
            },
            Self {
                name: "Building".to_string(),
                measure_types: vec![MeasureType::Standard, MeasureType::Build, MeasureType::Build, MeasureType::Fill],
                intro_probability: 0.2,
                fill_positions: vec![3],
                energy_arc: vec![0.6, 0.7, 0.9, 1.0],
            },
        ]
    }
}

impl TensionCurve {
    fn standard_curve() -> Self {
        Self {
            curve_points: vec![0.0, 0.3, 0.7, 1.0, 0.5, 0.2],
            resolution_points: vec![3, 5],
        }
    }

    fn get_tension_at_position(&self, phrase_pos: usize, beat_pos: usize) -> f32 {
        let total_pos = phrase_pos * 16 + beat_pos;
        let curve_index = total_pos % self.curve_points.len();
        self.curve_points[curve_index]
    }
}

impl EvolutionTrajectory {
    fn organic_growth() -> Self {
        Self {
            complexity_path: vec![0.3, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.6],
            energy_path: vec![0.5, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.7],
            pattern_mutation_rate: 0.1,
        }
    }

    fn get_evolution_at_position(&self, section_pos: usize) -> f32 {
        let index = section_pos % self.complexity_path.len();
        self.complexity_path[index]
    }
}

impl ComplexityEnvelope {
    fn building_complexity() -> Self {
        Self {
            stages: vec![
                ComplexityStage { target_complexity: 0.3, duration: 2.0, transition_style: TransitionStyle::Gradual },
                ComplexityStage { target_complexity: 0.7, duration: 4.0, transition_style: TransitionStyle::Gradual },
                ComplexityStage { target_complexity: 0.9, duration: 2.0, transition_style: TransitionStyle::Stepped },
            ],
            current_stage: 0,
            stage_progress: 0.0,
        }
    }

    fn update(&mut self) {
        self.stage_progress += 0.01;
        if self.stage_progress >= 1.0 {
            self.current_stage = (self.current_stage + 1) % self.stages.len();
            self.stage_progress = 0.0;
        }
    }

    fn get_complexity_at_position(&self, _section_pos: usize) -> f32 {
        if let Some(stage) = self.stages.get(self.current_stage) {
            stage.target_complexity * self.stage_progress
        } else {
            0.5
        }
    }
}

impl PatternMorphing {
    fn new() -> Self {
        Self {
            source_pattern: vec![],
            target_pattern: vec![],
            morph_position: 0.0,
            morph_rate: 0.01,
        }
    }

    fn update(&mut self) {
        self.morph_position += self.morph_rate;
        if self.morph_position >= 1.0 {
            self.morph_position = 0.0;
            // Switch patterns
            std::mem::swap(&mut self.source_pattern, &mut self.target_pattern);
        }
    }
}

impl PatternMemory {
    fn new() -> Self {
        Self {
            recent_patterns: VecDeque::new(),
            memory_depth: 8,
            repetition_avoidance: 0.7,
        }
    }

    fn avoid_repetition(&mut self, mut pattern: RhythmPattern) -> RhythmPattern {
        // Simple repetition avoidance - could be more sophisticated
        let pattern_signature = self.get_pattern_signature(&pattern);

        for recent in &self.recent_patterns {
            if let Some(recent_sig) = recent.first() {
                if self.patterns_too_similar(&pattern_signature, recent_sig) {
                    pattern = self.mutate_pattern(pattern);
                    break;
                }
            }
        }

        // Add to memory
        self.recent_patterns.push_back(vec![pattern.clone()]);
        if self.recent_patterns.len() > self.memory_depth {
            self.recent_patterns.pop_front();
        }

        pattern
    }

    fn get_pattern_signature(&self, pattern: &RhythmPattern) -> RhythmPattern {
        pattern.clone()
    }

    fn patterns_too_similar(&self, a: &RhythmPattern, b: &RhythmPattern) -> bool {
        a.kick == b.kick && a.snare == b.snare && a.hihat == b.hihat
    }

    fn mutate_pattern(&self, mut pattern: RhythmPattern) -> RhythmPattern {
        // Simple mutation
        if rand::random::<f32>() < 0.5 {
            pattern.hihat = !pattern.hihat;
        }
        pattern
    }

    fn clear(&mut self) {
        self.recent_patterns.clear();
    }
}

impl EuclideanRhythm {
    fn new(steps: usize, pulses: usize, rotation: usize, instrument: RhythmInstrument) -> Self {
        let mut rhythm = Self {
            steps,
            pulses,
            rotation,
            instrument,
            pattern: vec![false; steps],
        };
        rhythm.generate_pattern();
        rhythm
    }

    fn generate_pattern(&mut self) {
        // Euclidean rhythm algorithm
        self.pattern = vec![false; self.steps];

        let mut bucket = 0;
        for i in 0..self.steps {
            bucket += self.pulses;
            if bucket >= self.steps {
                bucket -= self.steps;
                let index = (i + self.rotation) % self.steps;
                self.pattern[index] = true;
            }
        }
    }

    fn update(&mut self) {
        // Occasionally evolve the euclidean pattern
        if rand::random::<f32>() < 0.01 {
            self.rotation = (self.rotation + 1) % self.steps;
            self.generate_pattern();
        }
    }

    fn blend_with_pattern(&self, mut pattern: RhythmPattern, params: &PatternParameters) -> RhythmPattern {
        let step = rand::random::<usize>() % self.steps;
        let euclidean_hit = self.pattern[step];

        if euclidean_hit && params.complexity > 0.5 {
            match self.instrument {
                RhythmInstrument::Kick => pattern.kick = true,
                RhythmInstrument::Snare => pattern.snare = true,
                RhythmInstrument::HiHat => pattern.hihat = true,
                RhythmInstrument::Percussion => pattern.percussion = true,
            }
        }

        pattern
    }
}