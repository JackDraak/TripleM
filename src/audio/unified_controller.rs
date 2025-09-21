//! Unified Control Interface for the Mood Music Module
//!
//! This module provides a comprehensive, intuitive API that exposes the powerful
//! cross-fade system and all underlying synthesis capabilities through a clean
//! interface suitable for integration with any application or control surface.

use crate::audio::{
    CrossfadeManager, CrossfadeParameter, CrossfadePriority,
    AudioPipeline, StereoFrame, VoiceCoordinator, MusicalContext,
};
use crate::audio::crossfade::CrossfadeStats;
use crate::patterns::multi_scale_rhythm::{MultiScaleRhythmSystem, ComplexityProfile};
use crate::config::MoodConfig;
use crate::error::{Result, MoodMusicError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Unified control interface exposing the cross-fade system with clean API
#[derive(Debug)]
pub struct UnifiedController {
    /// Core audio pipeline
    audio_pipeline: AudioPipeline,

    /// Cross-fade manager for seamless parameter transitions
    crossfade_manager: CrossfadeManager,

    /// Parameter state management
    parameter_state: ParameterState,

    /// Preset management system
    preset_manager: PresetManager,

    /// Real-time monitoring and feedback
    monitor: RealTimeMonitor,

    /// Multi-scale rhythm generation system
    rhythm_system: MultiScaleRhythmSystem,

    /// Voice coordination for polyphonic management
    voice_coordinator: VoiceCoordinator,

    /// Configuration
    config: MoodConfig,

    /// Controller state
    is_active: AtomicBool,
}

/// Parameter state tracking and validation
#[derive(Debug, Clone)]
pub struct ParameterState {
    /// Current parameter values (0.0-1.0)
    current_values: HashMap<ControlParameter, f32>,

    /// Target values during transitions
    target_values: HashMap<ControlParameter, f32>,

    /// Parameter constraints and mappings
    constraints: HashMap<ControlParameter, ParameterConstraints>,

    /// Change history for intelligent behavior
    change_history: Vec<ParameterChange>,
}

/// High-level control parameters exposed through the unified interface
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlParameter {
    // Main control
    /// Primary mood control (0.0-1.0): ambient → melodic → rhythmic → electronic
    MoodIntensity,

    /// Master volume control
    MasterVolume,

    // Musical parameters
    /// Tempo control (maps to BPM range based on research)
    Tempo,

    /// Overall complexity of musical elements
    MusicalComplexity,

    /// Rhythmic pattern density and complexity
    RhythmicDensity,

    /// Melodic note density and range
    MelodicDensity,

    /// Harmonic complexity (simple → complex chords)
    HarmonicComplexity,

    // Synthesis parameters
    /// Synthesis character morphing (wavetable ↔ additive)
    SynthesisCharacter,

    /// Timbre brightness and presence
    TimbralBrightness,

    /// Sound texture (smooth → granular)
    TexturalComplexity,

    // Spatial and effects
    /// Stereo field width
    StereoWidth,

    /// Reverb and ambient space
    AmbientSpace,

    /// Dynamic range and compression
    DynamicRange,

    // Advanced parameters
    /// Natural variation and humanization
    Humanization,

    /// Cross-fade behavior tuning
    TransitionSmoothing,
}

/// Parameter constraints and validation rules
#[derive(Debug, Clone)]
pub struct ParameterConstraints {
    /// Minimum allowed value
    min_value: f32,

    /// Maximum allowed value
    max_value: f32,

    /// Default value
    default_value: f32,

    /// Validation curve (linear, exponential, etc.)
    curve_type: ParameterCurve,

    /// Crossfade priority for this parameter
    crossfade_priority: CrossfadePriority,

    /// Crossfade duration override (None = use default)
    crossfade_duration: Option<f32>,
}

/// Parameter change record for intelligent behavior
#[derive(Debug, Clone)]
pub struct ParameterChange {
    /// Which parameter changed
    parameter: ControlParameter,

    /// Previous value
    old_value: f32,

    /// New value
    new_value: f32,

    /// When the change occurred
    timestamp: f64,

    /// How the change was initiated
    change_source: ChangeSource,
}

/// How parameter changes are initiated
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangeSource {
    /// User interface interaction
    UserInterface,

    /// Preset loading
    PresetLoad,

    /// Automated modulation
    Automation,

    /// Internal system adjustment
    System,
}

/// Parameter mapping curves
#[derive(Debug, Clone, Copy)]
pub enum ParameterCurve {
    /// Linear mapping
    Linear,

    /// Exponential curve (more sensitive at low values)
    Exponential,

    /// Logarithmic curve (more sensitive at high values)
    Logarithmic,

    /// S-curve for smooth transitions
    Sigmoid,
}

/// Preset management system
#[derive(Debug, Clone)]
pub struct PresetManager {
    /// Built-in presets
    builtin_presets: HashMap<String, Preset>,

    /// User-created presets
    user_presets: HashMap<String, Preset>,

    /// Currently active preset (if any)
    active_preset: Option<String>,
}

/// A complete parameter preset
#[derive(Debug, Clone)]
pub struct Preset {
    /// Preset name
    name: String,

    /// Parameter values
    parameters: HashMap<ControlParameter, f32>,

    /// Preset metadata
    metadata: PresetMetadata,
}

/// Preset metadata for organization and display
#[derive(Debug, Clone)]
pub struct PresetMetadata {
    /// Preset description
    pub description: String,

    /// Category (e.g., "Focus", "Relaxation", "Energy")
    pub category: String,

    /// Tags for searching
    pub tags: Vec<String>,

    /// When preset was created
    pub created_time: std::time::SystemTime,

    /// User rating (1-5 stars)
    pub user_rating: Option<u8>,
}

/// Real-time monitoring and feedback system
#[derive(Debug)]
pub struct RealTimeMonitor {
    /// Current CPU usage
    cpu_usage: f32,

    /// Audio output level monitoring
    output_levels: OutputLevels,

    /// Cross-fade system status
    crossfade_stats: CrossfadeStats,

    /// Parameter activity monitoring
    parameter_activity: HashMap<ControlParameter, ParameterActivity>,

    /// Performance metrics
    performance_metrics: PerformanceMetrics,
}

/// Audio output level monitoring
#[derive(Debug, Clone)]
pub struct OutputLevels {
    /// Current peak level (dB)
    peak_db: f32,

    /// RMS level (dB)
    rms_db: f32,

    /// Peak hold level
    peak_hold_db: f32,

    /// Clipping detection
    is_clipping: bool,
}

/// Parameter activity monitoring
#[derive(Debug, Clone)]
pub struct ParameterActivity {
    /// How recently this parameter changed
    last_change_time: f64,

    /// Rate of change
    change_velocity: f32,

    /// Whether parameter is currently transitioning
    is_transitioning: bool,
}

/// Performance metrics for system health monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Audio callback timing statistics
    audio_timing: TimingStats,

    /// Memory usage information
    memory_usage: MemoryUsage,

    /// System stability indicators
    stability_indicators: StabilityIndicators,
}

/// Timing statistics for performance monitoring
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Average callback duration (ms)
    avg_callback_duration: f32,

    /// Maximum callback duration (ms)
    max_callback_duration: f32,

    /// Callback timing variance
    timing_variance: f32,

    /// Number of timing violations
    timing_violations: u32,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Current memory usage (MB)
    current_mb: f32,

    /// Peak memory usage (MB)
    peak_mb: f32,

    /// Memory allocations per second
    allocations_per_sec: f32,
}

/// System stability indicators
#[derive(Debug, Clone)]
pub struct StabilityIndicators {
    /// Audio dropouts detected
    audio_dropouts: u32,

    /// Parameter validation failures
    validation_failures: u32,

    /// System health score (0.0-1.0)
    health_score: f32,
}

impl UnifiedController {
    /// Create a new unified controller
    pub fn new(config: MoodConfig) -> Result<Self> {
        let audio_pipeline = AudioPipeline::new(&config)?;
        let crossfade_manager = CrossfadeManager::new(config.sample_rate as f32)?;
        let parameter_state = ParameterState::new();
        let preset_manager = PresetManager::new();
        let monitor = RealTimeMonitor::new();
        let rhythm_system = MultiScaleRhythmSystem::new(config.sample_rate as f32)?;
        let voice_coordinator = VoiceCoordinator::new(16); // Max 16 voices

        Ok(Self {
            audio_pipeline,
            crossfade_manager,
            parameter_state,
            preset_manager,
            monitor,
            rhythm_system,
            voice_coordinator,
            config,
            is_active: AtomicBool::new(false),
        })
    }

    /// Start the audio system
    pub fn start(&self) -> Result<()> {
        self.audio_pipeline.start();
        self.is_active.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Stop the audio system
    pub fn stop(&self) {
        self.audio_pipeline.stop();
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Check if the controller is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }

    // === PRIMARY CONTROL METHODS ===

    /// Set the main mood intensity (0.0-1.0)
    /// This is the primary control that affects all other parameters
    pub fn set_mood_intensity(&mut self, intensity: f32) -> Result<()> {
        self.set_parameter_smooth(ControlParameter::MoodIntensity, intensity, ChangeSource::UserInterface)
    }

    /// Get the current mood intensity
    pub fn get_mood_intensity(&self) -> f32 {
        self.get_parameter_value(ControlParameter::MoodIntensity)
    }

    /// Set master volume (0.0-1.0)
    pub fn set_master_volume(&mut self, volume: f32) -> Result<()> {
        self.set_parameter_smooth(ControlParameter::MasterVolume, volume, ChangeSource::UserInterface)
    }

    /// Get current master volume
    pub fn get_master_volume(&self) -> f32 {
        self.get_parameter_value(ControlParameter::MasterVolume)
    }

    // === ADVANCED PARAMETER CONTROL ===

    /// Set any parameter with smooth crossfading
    pub fn set_parameter_smooth(
        &mut self,
        parameter: ControlParameter,
        value: f32,
        source: ChangeSource,
    ) -> Result<()> {
        // Validate parameter value
        let validated_value = self.validate_parameter_value(parameter, value)?;

        // Map to internal crossfade parameter
        let crossfade_param = self.map_to_crossfade_parameter(parameter);
        let priority = self.parameter_state.get_priority(parameter);

        // Record the change
        self.parameter_state.record_change(parameter, validated_value, source);

        // Request crossfade transition
        self.crossfade_manager.request_parameter_change(
            crossfade_param,
            validated_value,
            priority,
        )?;

        Ok(())
    }

    /// Set parameter immediately without crossfading
    pub fn set_parameter_immediate(&mut self, parameter: ControlParameter, value: f32) -> Result<()> {
        let validated_value = self.validate_parameter_value(parameter, value)?;
        let crossfade_param = self.map_to_crossfade_parameter(parameter);

        // Apply immediately
        self.crossfade_manager.apply_parameter_change_immediate(crossfade_param, validated_value);
        self.parameter_state.set_current_value(parameter, validated_value);

        Ok(())
    }

    /// Get current parameter value
    pub fn get_parameter_value(&self, parameter: ControlParameter) -> f32 {
        self.parameter_state.get_current_value(parameter)
    }

    /// Get all current parameter values
    pub fn get_all_parameters(&self) -> HashMap<ControlParameter, f32> {
        self.parameter_state.get_all_current_values()
    }

    // === PRESET MANAGEMENT ===

    /// Load a preset by name
    pub fn load_preset(&mut self, preset_name: &str) -> Result<()> {
        let preset_parameters = {
            let preset = self.preset_manager.get_preset(preset_name)
                .ok_or_else(|| MoodMusicError::InvalidConfiguration(format!("Preset not found: {}", preset_name)))?;
            preset.parameters.clone()
        };

        // Apply all preset parameters
        for (parameter, value) in preset_parameters {
            self.set_parameter_smooth(parameter, value, ChangeSource::PresetLoad)?;
        }

        self.preset_manager.set_active_preset(preset_name.to_string());
        Ok(())
    }

    /// Save current state as a new preset
    pub fn save_preset(&mut self, name: String, metadata: PresetMetadata) -> Result<()> {
        let preset = Preset {
            name: name.clone(),
            parameters: self.get_all_parameters(),
            metadata,
        };

        self.preset_manager.save_user_preset(name, preset);
        Ok(())
    }

    /// Get list of available presets
    pub fn list_presets(&self) -> Vec<String> {
        self.preset_manager.list_presets()
    }

    /// Get preset information
    pub fn get_preset_info(&self, preset_name: &str) -> Option<&PresetMetadata> {
        self.preset_manager.get_preset(preset_name)
            .map(|preset| &preset.metadata)
    }

    // === REAL-TIME MONITORING ===

    /// Get current system status
    pub fn get_system_status(&self) -> SystemStatus {
        SystemStatus {
            is_active: self.is_active(),
            cpu_usage: self.monitor.cpu_usage,
            output_levels: self.monitor.output_levels.clone(),
            crossfade_stats: self.monitor.crossfade_stats.clone(),
            active_parameters: self.get_active_parameters(),
            performance_metrics: self.monitor.performance_metrics.clone(),
        }
    }

    /// Get real-time audio output levels
    pub fn get_output_levels(&self) -> &OutputLevels {
        &self.monitor.output_levels
    }

    /// Get crossfade system statistics
    pub fn get_crossfade_stats(&self) -> &CrossfadeStats {
        &self.monitor.crossfade_stats
    }

    // === AUDIO GENERATION ===

    /// Get next audio sample (mono)
    pub fn get_next_sample(&mut self) -> f32 {
        if !self.is_active() {
            return 0.0;
        }

        // Update crossfade manager
        let delta_time = 1.0 / self.config.sample_rate as f32;
        self.crossfade_manager.update(delta_time);

        // Update parameter state with crossfaded values
        self.update_parameter_state_from_crossfades();

        // Generate rhythm patterns and events
        self.update_rhythm_system(delta_time);

        // Generate audio sample
        let sample = self.audio_pipeline.get_next_sample();

        // Update monitoring
        self.monitor.update_with_sample(sample);

        sample
    }

    /// Get next stereo audio sample
    pub fn get_next_stereo_sample(&mut self) -> StereoFrame {
        if !self.is_active() {
            return StereoFrame::silence();
        }

        // Update crossfade manager
        let delta_time = 1.0 / self.config.sample_rate as f32;
        self.crossfade_manager.update(delta_time);

        // Update parameter state with crossfaded values
        self.update_parameter_state_from_crossfades();

        // Generate rhythm patterns and events
        self.update_rhythm_system(delta_time);

        // Generate stereo sample
        let sample = self.audio_pipeline.get_next_stereo_sample();

        // Update monitoring
        self.monitor.update_with_stereo_sample(sample);

        sample
    }

    /// Fill audio buffer (more efficient for batch processing)
    pub fn fill_buffer(&mut self, buffer: &mut [f32]) {
        if !self.is_active() {
            buffer.fill(0.0);
            return;
        }

        // Update systems for entire buffer
        let delta_time = buffer.len() as f32 / self.config.sample_rate as f32;
        self.crossfade_manager.update(delta_time);
        self.update_parameter_state_from_crossfades();

        // Fill buffer
        self.audio_pipeline.fill_buffer(buffer);

        // Update monitoring
        self.monitor.update_with_buffer(buffer);
    }

    /// Fill stereo audio buffer
    pub fn fill_stereo_buffer(&mut self, buffer: &mut [StereoFrame]) {
        if !self.is_active() {
            buffer.fill(StereoFrame::silence());
            return;
        }

        // Update systems for entire buffer
        let delta_time = buffer.len() as f32 / self.config.sample_rate as f32;
        self.crossfade_manager.update(delta_time);
        self.update_parameter_state_from_crossfades();

        // Fill buffer
        self.audio_pipeline.fill_stereo_buffer(buffer);

        // Update monitoring
        self.monitor.update_with_stereo_buffer(buffer);
    }

    // === CONFIGURATION AND DIAGNOSTICS ===

    /// Get current configuration
    pub fn config(&self) -> &MoodConfig {
        &self.config
    }

    /// Get diagnostic information
    pub fn get_diagnostics(&self) -> ControllerDiagnostics {
        ControllerDiagnostics {
            system_status: self.get_system_status(),
            parameter_state: self.parameter_state.clone(),
            crossfade_manager_stats: self.crossfade_manager.get_crossfade_stats(),
            audio_pipeline_diagnostics: self.audio_pipeline.get_diagnostics(),
        }
    }

    // === INTERNAL HELPER METHODS ===

    /// Validate parameter value against constraints
    fn validate_parameter_value(&self, parameter: ControlParameter, value: f32) -> Result<f32> {
        let constraints = self.parameter_state.get_constraints(parameter);

        if value < constraints.min_value || value > constraints.max_value {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Parameter {:?} value {} out of range [{}, {}]",
                    parameter, value, constraints.min_value, constraints.max_value)
            ));
        }

        // Apply parameter curve
        Ok(self.apply_parameter_curve(value, constraints.curve_type))
    }

    /// Apply parameter curve transformation
    fn apply_parameter_curve(&self, value: f32, curve: ParameterCurve) -> f32 {
        match curve {
            ParameterCurve::Linear => value,
            ParameterCurve::Exponential => value * value,
            ParameterCurve::Logarithmic => value.sqrt(),
            ParameterCurve::Sigmoid => {
                // S-curve: smooth transitions
                let x = (value - 0.5) * 6.0; // Scale to -3 to +3
                1.0 / (1.0 + (-x).exp())
            }
        }
    }

    /// Map control parameter to internal crossfade parameter
    fn map_to_crossfade_parameter(&self, parameter: ControlParameter) -> CrossfadeParameter {
        match parameter {
            ControlParameter::MoodIntensity => CrossfadeParameter::InputValue,
            ControlParameter::MasterVolume => CrossfadeParameter::MasterVolume,
            ControlParameter::Tempo => CrossfadeParameter::Tempo,
            ControlParameter::RhythmicDensity => CrossfadeParameter::RhythmComplexity,
            ControlParameter::MelodicDensity => CrossfadeParameter::NoteDensity,
            ControlParameter::HarmonicComplexity => CrossfadeParameter::ChordComplexity,
            ControlParameter::SynthesisCharacter => CrossfadeParameter::SynthesisMorph,
            ControlParameter::StereoWidth => CrossfadeParameter::StereoWidth,
            _ => CrossfadeParameter::Custom(parameter as u32),
        }
    }

    /// Update parameter state with current crossfade values
    fn update_parameter_state_from_crossfades(&mut self) {
        for parameter in [
            ControlParameter::MoodIntensity,
            ControlParameter::MasterVolume,
            ControlParameter::Tempo,
            ControlParameter::RhythmicDensity,
            ControlParameter::MelodicDensity,
            ControlParameter::HarmonicComplexity,
            ControlParameter::SynthesisCharacter,
            ControlParameter::StereoWidth,
        ] {
            let crossfade_param = self.map_to_crossfade_parameter(parameter);
            let current_value = self.crossfade_manager.get_parameter_value(crossfade_param);
            self.parameter_state.set_current_value(parameter, current_value);
        }
    }

    /// Get list of currently active/transitioning parameters
    fn get_active_parameters(&self) -> Vec<ControlParameter> {
        self.parameter_state.get_active_parameters()
    }

    /// Update rhythm system and generate events
    fn update_rhythm_system(&mut self, delta_time: f32) {
        // Update rhythm system with current mood intensity
        let input_value = self.get_parameter_value(ControlParameter::MoodIntensity);
        self.rhythm_system.set_input_value(input_value);

        // Generate rhythm patterns
        if let Ok(pattern) = self.rhythm_system.generate_pattern() {
            // Process the multi-scale rhythm pattern
            self.process_rhythm_pattern(pattern);
        }
    }

    /// Process a multi-scale rhythm pattern through the voice coordination system
    fn process_rhythm_pattern(&mut self, _pattern: crate::patterns::multi_scale_rhythm::MultiScaleRhythmPattern) {
        // For now, this is a placeholder - in a full implementation, this would:
        // 1. Extract patterns from all scales (micro, meso, macro)
        // 2. Convert to AudioEvents with proper timing
        // 3. Route through voice coordinator for polyphonic allocation
        // 4. Send to appropriate generators based on pattern characteristics

        // This integration point represents where sophisticated multi-scale rhythm
        // patterns become actual audio through the generator system.
        // The MultiScaleRhythmPattern contains:
        // - micro_scale_events: High-frequency pattern details
        // - meso_scale_events: Mid-level rhythmic structure
        // - macro_scale_events: Large-scale musical architecture
        // - polyrhythmic_layers: Overlapping rhythmic patterns
        // - euclidean_patterns: Mathematical rhythm distributions
        // - phrase_context: Musical phrase awareness
    }
}

/// Complete system status information
#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub is_active: bool,
    pub cpu_usage: f32,
    pub output_levels: OutputLevels,
    pub crossfade_stats: CrossfadeStats,
    pub active_parameters: Vec<ControlParameter>,
    pub performance_metrics: PerformanceMetrics,
}

/// Complete diagnostic information
#[derive(Debug, Clone)]
pub struct ControllerDiagnostics {
    pub system_status: SystemStatus,
    pub parameter_state: ParameterState,
    pub crossfade_manager_stats: CrossfadeStats,
    pub audio_pipeline_diagnostics: crate::audio::pipeline::PipelineDiagnostics,
}

// Implementation stubs for supporting structures
impl ParameterState {
    fn new() -> Self {
        Self {
            current_values: HashMap::new(),
            target_values: HashMap::new(),
            constraints: Self::create_default_constraints(),
            change_history: Vec::new(),
        }
    }

    fn create_default_constraints() -> HashMap<ControlParameter, ParameterConstraints> {
        // This would be populated with sensible defaults for each parameter
        HashMap::new()
    }

    fn get_current_value(&self, parameter: ControlParameter) -> f32 {
        self.current_values.get(&parameter).copied().unwrap_or(0.0)
    }

    fn set_current_value(&mut self, parameter: ControlParameter, value: f32) {
        self.current_values.insert(parameter, value);
    }

    fn get_all_current_values(&self) -> HashMap<ControlParameter, f32> {
        self.current_values.clone()
    }

    fn get_constraints(&self, parameter: ControlParameter) -> &ParameterConstraints {
        self.constraints.get(&parameter).unwrap_or_else(|| {
            static DEFAULT: ParameterConstraints = ParameterConstraints {
                min_value: 0.0,
                max_value: 1.0,
                default_value: 0.0,
                curve_type: ParameterCurve::Linear,
                crossfade_priority: CrossfadePriority::Normal,
                crossfade_duration: None,
            };
            &DEFAULT
        })
    }

    fn get_priority(&self, parameter: ControlParameter) -> CrossfadePriority {
        self.get_constraints(parameter).crossfade_priority
    }

    fn record_change(&mut self, parameter: ControlParameter, value: f32, source: ChangeSource) {
        let old_value = self.get_current_value(parameter);
        self.change_history.push(ParameterChange {
            parameter,
            old_value,
            new_value: value,
            timestamp: 0.0, // Would use actual timestamp
            change_source: source,
        });
        self.set_current_value(parameter, value);
    }

    fn get_active_parameters(&self) -> Vec<ControlParameter> {
        // Return parameters that are currently transitioning
        self.current_values.keys().cloned().collect()
    }
}

impl ParameterConstraints {
    fn default() -> Self {
        Self {
            min_value: 0.0,
            max_value: 1.0,
            default_value: 0.0,
            curve_type: ParameterCurve::Linear,
            crossfade_priority: CrossfadePriority::Normal,
            crossfade_duration: None,
        }
    }
}

impl PresetManager {
    fn new() -> Self {
        Self {
            builtin_presets: Self::create_builtin_presets(),
            user_presets: HashMap::new(),
            active_preset: None,
        }
    }

    fn create_builtin_presets() -> HashMap<String, Preset> {
        // Would create default presets like "Focus", "Relaxation", "Energy", etc.
        HashMap::new()
    }

    fn get_preset(&self, name: &str) -> Option<&Preset> {
        self.builtin_presets.get(name).or_else(|| self.user_presets.get(name))
    }

    fn save_user_preset(&mut self, name: String, preset: Preset) {
        self.user_presets.insert(name, preset);
    }

    fn set_active_preset(&mut self, name: String) {
        self.active_preset = Some(name);
    }

    fn list_presets(&self) -> Vec<String> {
        let mut presets: Vec<String> = self.builtin_presets.keys().cloned().collect();
        presets.extend(self.user_presets.keys().cloned());
        presets.sort();
        presets
    }
}

impl RealTimeMonitor {
    fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            output_levels: OutputLevels::default(),
            crossfade_stats: CrossfadeStats {
                active_crossfades: 0,
                parameters_being_crossfaded: 0,
                average_crossfade_duration: 0.0,
                crossfade_cpu_usage: 0.0,
            },
            parameter_activity: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
        }
    }

    fn update_with_sample(&mut self, _sample: f32) {
        // Update monitoring with audio sample
    }

    fn update_with_stereo_sample(&mut self, _sample: StereoFrame) {
        // Update monitoring with stereo sample
    }

    fn update_with_buffer(&mut self, _buffer: &[f32]) {
        // Update monitoring with audio buffer
    }

    fn update_with_stereo_buffer(&mut self, _buffer: &[StereoFrame]) {
        // Update monitoring with stereo buffer
    }
}

impl Default for OutputLevels {
    fn default() -> Self {
        Self {
            peak_db: -96.0,
            rms_db: -96.0,
            peak_hold_db: -96.0,
            is_clipping: false,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            audio_timing: TimingStats::default(),
            memory_usage: MemoryUsage::default(),
            stability_indicators: StabilityIndicators::default(),
        }
    }
}

impl Default for TimingStats {
    fn default() -> Self {
        Self {
            avg_callback_duration: 0.0,
            max_callback_duration: 0.0,
            timing_variance: 0.0,
            timing_violations: 0,
        }
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            current_mb: 0.0,
            peak_mb: 0.0,
            allocations_per_sec: 0.0,
        }
    }
}

impl Default for StabilityIndicators {
    fn default() -> Self {
        Self {
            audio_dropouts: 0,
            validation_failures: 0,
            health_score: 1.0,
        }
    }
}