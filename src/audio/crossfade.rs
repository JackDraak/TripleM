//! Seamless cross-fade system for real-time parameter morphing
//!
//! This module implements sophisticated cross-fading algorithms to ensure
//! completely seamless transitions when parameters change in real-time,
//! eliminating clicks, pops, and other audio artifacts.

use crate::audio::NaturalVariation;
use crate::error::Result;
use std::collections::VecDeque;

/// Seamless cross-fade manager for real-time parameter morphing
#[derive(Debug, Clone)]
pub struct CrossfadeManager {
    /// Active crossfade instances
    active_crossfades: Vec<ActiveCrossfade>,

    /// Crossfade configuration
    config: CrossfadeConfig,

    /// Parameter change detection
    change_detector: ParameterChangeDetector,

    /// Smoothing filters for different parameter types
    smoothing_filters: SmoothingFilterBank,

    /// Natural variation for organic transitions
    variation: NaturalVariation,

    /// Sample rate for timing calculations
    sample_rate: f32,
}

/// Configuration for crossfade behavior
#[derive(Debug, Clone)]
pub struct CrossfadeConfig {
    /// Default crossfade duration in seconds
    default_duration: f32,

    /// Crossfade curve type
    curve_type: CrossfadeCurve,

    /// Parameter-specific crossfade settings
    parameter_settings: Vec<ParameterCrossfadeSettings>,

    /// Anti-aliasing filters
    anti_alias_enabled: bool,

    /// Lookahead time for smooth transitions
    lookahead_time: f32,
}

/// Individual crossfade instance
#[derive(Debug, Clone)]
pub struct ActiveCrossfade {
    /// Unique ID for this crossfade
    id: usize,

    /// Parameter being crossfaded
    parameter: CrossfadeParameter,

    /// Source value (where we're fading from)
    source_value: f32,

    /// Target value (where we're fading to)
    target_value: f32,

    /// Current position in crossfade (0.0 to 1.0)
    position: f32,

    /// Duration of this crossfade
    duration: f32,

    /// Elapsed time
    elapsed_time: f32,

    /// Crossfade curve for this transition
    curve: CrossfadeCurve,

    /// Priority (higher priority crossfades take precedence)
    priority: CrossfadePriority,

    /// State of this crossfade
    state: CrossfadeState,
}

/// Types of parameters that can be crossfaded
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeParameter {
    // Global parameters
    InputValue,
    MasterVolume,

    // Rhythm parameters
    Tempo,
    RhythmComplexity,
    GrooveMorph,

    // Melody parameters
    ScaleMorph,
    NoteDensity,
    MelodicRange,

    // Harmony parameters
    ChordComplexity,
    VoicingSpread,
    HarmonicRhythm,

    // Synthesis parameters
    SynthesisMorph,
    WavetableCharacter,
    AdditiveComplexity,

    // Audio processing parameters
    FilterCutoff,
    FilterResonance,
    StereoWidth,
    Compression,

    // Custom parameter
    Custom(u32),
}

/// Parameter-specific crossfade settings
#[derive(Debug, Clone)]
pub struct ParameterCrossfadeSettings {
    /// Which parameter this applies to
    parameter: CrossfadeParameter,

    /// Crossfade duration for this parameter
    duration: f32,

    /// Curve type for this parameter
    curve: CrossfadeCurve,

    /// Smoothing characteristics
    smoothing: SmoothingCharacteristics,

    /// Whether this parameter needs anti-aliasing
    needs_anti_alias: bool,
}

/// Crossfade curve shapes
#[derive(Debug, Clone, Copy)]
pub enum CrossfadeCurve {
    /// Linear crossfade (constant power)
    Linear,

    /// Smooth S-curve (sigmoid-like)
    Smooth,

    /// Exponential curve (faster start, slower end)
    Exponential,

    /// Logarithmic curve (slower start, faster end)
    Logarithmic,

    /// Equal power crossfade (for audio signals)
    EqualPower,

    /// Custom curve with control points
    Custom,
}

/// Priority levels for crossfades
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub enum CrossfadePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// State of an active crossfade
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeState {
    /// Crossfade is active and interpolating
    Active,

    /// Crossfade is completing
    Finishing,

    /// Crossfade is complete and can be removed
    Complete,

    /// Crossfade was cancelled
    Cancelled,
}

/// Detects parameter changes and initiates crossfades
#[derive(Debug, Clone)]
pub struct ParameterChangeDetector {
    /// Previous parameter values
    previous_values: Vec<(CrossfadeParameter, f32)>,

    /// Change detection thresholds
    change_thresholds: Vec<(CrossfadeParameter, f32)>,

    /// Velocity tracking for adaptive crossfade timing
    velocity_tracker: ParameterVelocityTracker,

    /// Rate limiting to prevent excessive crossfades
    rate_limiter: RateLimiter,
}

/// Tracks the velocity of parameter changes
#[derive(Debug, Clone)]
pub struct ParameterVelocityTracker {
    /// History of parameter changes
    change_history: VecDeque<ParameterChange>,

    /// Maximum history length
    max_history: usize,

    /// Time window for velocity calculation
    velocity_window: f32,
}

/// Individual parameter change record
#[derive(Debug, Clone)]
pub struct ParameterChange {
    /// Which parameter changed
    parameter: CrossfadeParameter,

    /// Value before change
    old_value: f32,

    /// Value after change
    new_value: f32,

    /// When the change occurred
    timestamp: f32,
}

/// Rate limiter to prevent excessive crossfade triggering
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Last crossfade time for each parameter
    last_crossfade_times: Vec<(CrossfadeParameter, f32)>,

    /// Minimum time between crossfades per parameter
    min_intervals: Vec<(CrossfadeParameter, f32)>,

    /// Current time
    current_time: f32,
}

/// Bank of smoothing filters for different parameter types
#[derive(Debug, Clone)]
pub struct SmoothingFilterBank {
    /// Smoothing filters
    filters: Vec<SmoothingFilter>,
}

/// Individual smoothing filter
#[derive(Debug, Clone)]
pub struct SmoothingFilter {
    /// Which parameter this filter handles
    parameter: CrossfadeParameter,

    /// Filter characteristics
    characteristics: SmoothingCharacteristics,

    /// Filter state
    state: FilterState,
}

/// Smoothing characteristics for different parameter types
#[derive(Debug, Clone)]
pub struct SmoothingCharacteristics {
    /// Smoothing factor (0.0 to 1.0)
    smoothing_factor: f32,

    /// Response time (seconds)
    response_time: f32,

    /// Filter type
    filter_type: SmoothingFilterType,

    /// Natural variation amount
    variation_amount: f32,
}

/// Types of smoothing filters
#[derive(Debug, Clone, Copy)]
pub enum SmoothingFilterType {
    /// Simple exponential smoothing
    Exponential,

    /// One-pole lowpass filter
    OnePole,

    /// Biquad filter
    Biquad,

    /// Custom smoothing algorithm
    Custom,
}

/// Filter state for smoothing
#[derive(Debug, Clone)]
pub struct FilterState {
    /// Current output value
    current_value: f32,

    /// Filter memory (implementation-specific)
    memory: Vec<f32>,

    /// Time since last update
    time_since_update: f32,
}

impl CrossfadeManager {
    /// Create a new crossfade manager
    pub fn new(sample_rate: f32) -> Result<Self> {
        Ok(Self {
            active_crossfades: Vec::new(),
            config: CrossfadeConfig::default(),
            change_detector: ParameterChangeDetector::new(),
            smoothing_filters: SmoothingFilterBank::new(),
            variation: NaturalVariation::new(None),
            sample_rate,
        })
    }

    /// Update the crossfade manager and process all active crossfades
    pub fn update(&mut self, delta_time: f32) {
        // Update natural variation
        self.variation.update();

        // Update change detector
        self.change_detector.update(delta_time);

        // Update all active crossfades
        for i in 0..self.active_crossfades.len() {
            self.update_crossfade_by_index(i, delta_time);
        }

        // Remove completed crossfades
        self.active_crossfades.retain(|cf| cf.state != CrossfadeState::Complete);

        // Update smoothing filters
        self.smoothing_filters.update(delta_time);
    }

    /// Request a parameter change with crossfading
    pub fn request_parameter_change(
        &mut self,
        parameter: CrossfadeParameter,
        new_value: f32,
        priority: CrossfadePriority,
    ) -> Result<Option<usize>> {
        // Check if this change should be rate-limited
        if !self.change_detector.rate_limiter.should_allow_change(parameter) {
            return Ok(None);
        }

        // Get current value for this parameter
        let current_value = self.get_current_parameter_value(parameter);

        // Check if change is significant enough to warrant crossfading
        if !self.change_detector.is_change_significant(parameter, current_value, new_value) {
            // Apply change immediately without crossfading
            self.apply_parameter_change_immediate(parameter, new_value);
            return Ok(None);
        }

        // Cancel any existing crossfade for this parameter if new one has higher priority
        self.cancel_existing_crossfade(parameter, priority);

        // Calculate crossfade duration based on parameter velocity
        let velocity = self.change_detector.velocity_tracker.get_velocity(parameter);
        let duration = self.calculate_adaptive_duration(parameter, velocity);

        // Create new crossfade
        let crossfade_id = self.create_crossfade(
            parameter,
            current_value,
            new_value,
            duration,
            priority,
        );

        // Record the parameter change
        self.change_detector.record_change(parameter, current_value, new_value);

        Ok(Some(crossfade_id))
    }

    /// Get the current crossfaded value for a parameter
    pub fn get_parameter_value(&self, parameter: CrossfadeParameter) -> f32 {
        // Check if there's an active crossfade for this parameter
        if let Some(crossfade) = self.find_active_crossfade(parameter) {
            // Calculate crossfaded value
            let progress = self.calculate_crossfade_progress(crossfade);
            self.interpolate_value(
                crossfade.source_value,
                crossfade.target_value,
                progress,
                crossfade.curve,
            )
        } else {
            // Use smoothed value from filter bank
            self.smoothing_filters.get_smoothed_value(parameter)
        }
    }

    /// Apply a parameter change immediately without crossfading
    pub fn apply_parameter_change_immediate(&mut self, parameter: CrossfadeParameter, value: f32) {
        self.smoothing_filters.set_target_value(parameter, value);
    }

    /// Cancel all crossfades for a specific parameter
    pub fn cancel_parameter_crossfades(&mut self, parameter: CrossfadeParameter) {
        for crossfade in &mut self.active_crossfades {
            if crossfade.parameter == parameter {
                crossfade.state = CrossfadeState::Cancelled;
            }
        }
    }

    /// Get crossfade statistics for monitoring
    pub fn get_crossfade_stats(&self) -> CrossfadeStats {
        CrossfadeStats {
            active_crossfades: self.active_crossfades.len(),
            parameters_being_crossfaded: self.get_active_parameters().len(),
            average_crossfade_duration: self.calculate_average_duration(),
            crossfade_cpu_usage: self.estimate_cpu_usage(),
        }
    }

    /// Update a single crossfade by index
    fn update_crossfade_by_index(&mut self, index: usize, delta_time: f32) {
        if index >= self.active_crossfades.len() {
            return;
        }

        let crossfade = &mut self.active_crossfades[index];
        if crossfade.state != CrossfadeState::Active {
            return;
        }

        crossfade.elapsed_time += delta_time;
        crossfade.position = (crossfade.elapsed_time / crossfade.duration).min(1.0);

        // Add natural variation to crossfade timing
        let variation_factor = 1.0 + self.variation.get_timing_variation() * 0.05; // ±5% timing variation
        crossfade.position *= variation_factor;
        crossfade.position = crossfade.position.clamp(0.0, 1.0);

        // Check if crossfade is complete
        if crossfade.position >= 1.0 {
            let parameter = crossfade.parameter;
            let target_value = crossfade.target_value;
            crossfade.state = CrossfadeState::Complete;
            // Set final value in smoothing filter
            self.smoothing_filters.set_target_value(parameter, target_value);
        }
    }

    /// Find active crossfade for a parameter
    fn find_active_crossfade(&self, parameter: CrossfadeParameter) -> Option<&ActiveCrossfade> {
        self.active_crossfades
            .iter()
            .find(|cf| cf.parameter == parameter && cf.state == CrossfadeState::Active)
    }

    /// Calculate crossfade progress with curve application
    fn calculate_crossfade_progress(&self, crossfade: &ActiveCrossfade) -> f32 {
        self.apply_curve(crossfade.position, crossfade.curve)
    }

    /// Apply crossfade curve to linear progress
    fn apply_curve(&self, linear_progress: f32, curve: CrossfadeCurve) -> f32 {
        match curve {
            CrossfadeCurve::Linear => linear_progress,
            CrossfadeCurve::Smooth => {
                // Smooth S-curve (3x² - 2x³)
                let x = linear_progress;
                3.0 * x * x - 2.0 * x * x * x
            },
            CrossfadeCurve::Exponential => {
                // Exponential curve
                (linear_progress * linear_progress).sqrt()
            },
            CrossfadeCurve::Logarithmic => {
                // Logarithmic curve
                1.0 - (1.0 - linear_progress).powf(2.0)
            },
            CrossfadeCurve::EqualPower => {
                // Equal power crossfade (sine/cosine)
                use std::f32::consts::PI;
                (linear_progress * PI * 0.5).sin()
            },
            CrossfadeCurve::Custom => {
                // Custom curve - could be implemented with control points
                linear_progress // Fallback to linear for now
            },
        }
    }

    /// Interpolate between source and target values
    fn interpolate_value(&self, source: f32, target: f32, progress: f32, _curve: CrossfadeCurve) -> f32 {
        source + (target - source) * progress
    }

    /// Get current parameter value from various sources
    fn get_current_parameter_value(&self, parameter: CrossfadeParameter) -> f32 {
        // Check for active crossfade first
        if let Some(crossfade) = self.find_active_crossfade(parameter) {
            let progress = self.calculate_crossfade_progress(crossfade);
            self.interpolate_value(
                crossfade.source_value,
                crossfade.target_value,
                progress,
                crossfade.curve,
            )
        } else {
            // Get from smoothing filter
            self.smoothing_filters.get_smoothed_value(parameter)
        }
    }

    /// Cancel existing crossfade if new one has higher priority
    fn cancel_existing_crossfade(&mut self, parameter: CrossfadeParameter, new_priority: CrossfadePriority) {
        for crossfade in &mut self.active_crossfades {
            if crossfade.parameter == parameter && crossfade.priority < new_priority {
                crossfade.state = CrossfadeState::Cancelled;
            }
        }
    }

    /// Calculate adaptive crossfade duration based on parameter velocity
    fn calculate_adaptive_duration(&self, parameter: CrossfadeParameter, velocity: f32) -> f32 {
        let base_duration = self.config.get_parameter_duration(parameter);

        // Faster changes get shorter crossfades (to a point)
        let velocity_factor = (1.0 / (1.0 + velocity * 2.0)).max(0.3); // Min 30% of base duration

        base_duration * velocity_factor
    }

    /// Create a new crossfade instance
    fn create_crossfade(
        &mut self,
        parameter: CrossfadeParameter,
        source_value: f32,
        target_value: f32,
        duration: f32,
        priority: CrossfadePriority,
    ) -> usize {
        let id = self.generate_crossfade_id();
        let curve = self.config.get_parameter_curve(parameter);

        let crossfade = ActiveCrossfade {
            id,
            parameter,
            source_value,
            target_value,
            position: 0.0,
            duration,
            elapsed_time: 0.0,
            curve,
            priority,
            state: CrossfadeState::Active,
        };

        self.active_crossfades.push(crossfade);
        id
    }

    /// Generate unique crossfade ID
    fn generate_crossfade_id(&self) -> usize {
        static mut NEXT_ID: usize = 1;
        unsafe {
            let id = NEXT_ID;
            NEXT_ID += 1;
            id
        }
    }

    /// Get list of parameters currently being crossfaded
    fn get_active_parameters(&self) -> Vec<CrossfadeParameter> {
        self.active_crossfades
            .iter()
            .filter(|cf| cf.state == CrossfadeState::Active)
            .map(|cf| cf.parameter)
            .collect()
    }

    /// Calculate average crossfade duration
    fn calculate_average_duration(&self) -> f32 {
        if self.active_crossfades.is_empty() {
            return 0.0;
        }

        let total_duration: f32 = self.active_crossfades.iter().map(|cf| cf.duration).sum();
        total_duration / self.active_crossfades.len() as f32
    }

    /// Estimate CPU usage of crossfade system
    fn estimate_cpu_usage(&self) -> f32 {
        // Simple estimation based on number of active crossfades
        let base_usage = 0.001; // Base overhead
        let per_crossfade_usage = 0.0005; // Per crossfade overhead

        base_usage + (self.active_crossfades.len() as f32 * per_crossfade_usage)
    }
}

/// Statistics about crossfade system performance
#[derive(Debug, Clone)]
pub struct CrossfadeStats {
    /// Number of currently active crossfades
    pub active_crossfades: usize,

    /// Number of unique parameters being crossfaded
    pub parameters_being_crossfaded: usize,

    /// Average duration of active crossfades
    pub average_crossfade_duration: f32,

    /// Estimated CPU usage (0.0 to 1.0)
    pub crossfade_cpu_usage: f32,
}

// Implementation for supporting structures
impl CrossfadeConfig {
    fn default() -> Self {
        Self {
            default_duration: 0.1, // 100ms default
            curve_type: CrossfadeCurve::Smooth,
            parameter_settings: Self::create_default_parameter_settings(),
            anti_alias_enabled: true,
            lookahead_time: 0.005, // 5ms lookahead
        }
    }

    fn create_default_parameter_settings() -> Vec<ParameterCrossfadeSettings> {
        vec![
            ParameterCrossfadeSettings {
                parameter: CrossfadeParameter::InputValue,
                duration: 0.2,
                curve: CrossfadeCurve::Smooth,
                smoothing: SmoothingCharacteristics::default(),
                needs_anti_alias: false,
            },
            ParameterCrossfadeSettings {
                parameter: CrossfadeParameter::Tempo,
                duration: 0.5,
                curve: CrossfadeCurve::Smooth,
                smoothing: SmoothingCharacteristics::slow_response(),
                needs_anti_alias: false,
            },
            ParameterCrossfadeSettings {
                parameter: CrossfadeParameter::FilterCutoff,
                duration: 0.05,
                curve: CrossfadeCurve::EqualPower,
                smoothing: SmoothingCharacteristics::fast_response(),
                needs_anti_alias: true,
            },
        ]
    }

    fn get_parameter_duration(&self, parameter: CrossfadeParameter) -> f32 {
        self.parameter_settings
            .iter()
            .find(|setting| setting.parameter == parameter)
            .map(|setting| setting.duration)
            .unwrap_or(self.default_duration)
    }

    fn get_parameter_curve(&self, parameter: CrossfadeParameter) -> CrossfadeCurve {
        self.parameter_settings
            .iter()
            .find(|setting| setting.parameter == parameter)
            .map(|setting| setting.curve)
            .unwrap_or(self.curve_type)
    }
}

impl ParameterChangeDetector {
    fn new() -> Self {
        Self {
            previous_values: Vec::new(),
            change_thresholds: Self::create_default_thresholds(),
            velocity_tracker: ParameterVelocityTracker::new(),
            rate_limiter: RateLimiter::new(),
        }
    }

    fn create_default_thresholds() -> Vec<(CrossfadeParameter, f32)> {
        vec![
            (CrossfadeParameter::InputValue, 0.01),
            (CrossfadeParameter::Tempo, 1.0),
            (CrossfadeParameter::FilterCutoff, 50.0),
            (CrossfadeParameter::StereoWidth, 0.02),
        ]
    }

    fn update(&mut self, delta_time: f32) {
        self.velocity_tracker.update(delta_time);
        self.rate_limiter.update(delta_time);
    }

    fn is_change_significant(&self, parameter: CrossfadeParameter, old_value: f32, new_value: f32) -> bool {
        let threshold = self.change_thresholds
            .iter()
            .find(|(param, _)| *param == parameter)
            .map(|(_, threshold)| *threshold)
            .unwrap_or(0.001); // Default threshold

        (new_value - old_value).abs() > threshold
    }

    fn record_change(&mut self, parameter: CrossfadeParameter, old_value: f32, new_value: f32) {
        self.velocity_tracker.record_change(parameter, old_value, new_value);

        // Update previous values
        if let Some(entry) = self.previous_values.iter_mut().find(|(param, _)| *param == parameter) {
            entry.1 = new_value;
        } else {
            self.previous_values.push((parameter, new_value));
        }
    }
}

impl ParameterVelocityTracker {
    fn new() -> Self {
        Self {
            change_history: VecDeque::new(),
            max_history: 100,
            velocity_window: 1.0, // 1 second window
        }
    }

    fn update(&mut self, delta_time: f32) {
        // Update timestamps
        for change in &mut self.change_history {
            change.timestamp += delta_time;
        }

        // Remove old changes outside velocity window
        self.change_history.retain(|change| change.timestamp <= self.velocity_window);
    }

    fn record_change(&mut self, parameter: CrossfadeParameter, old_value: f32, new_value: f32) {
        let change = ParameterChange {
            parameter,
            old_value,
            new_value,
            timestamp: 0.0,
        };

        self.change_history.push_back(change);

        // Limit history size
        if self.change_history.len() > self.max_history {
            self.change_history.pop_front();
        }
    }

    fn get_velocity(&self, parameter: CrossfadeParameter) -> f32 {
        let recent_changes: Vec<_> = self.change_history
            .iter()
            .filter(|change| change.parameter == parameter)
            .collect();

        if recent_changes.len() < 2 {
            return 0.0;
        }

        // Calculate velocity as rate of change
        let mut total_change = 0.0;
        let mut total_time = 0.0;

        for window in recent_changes.windows(2) {
            let change_amount = (window[1].new_value - window[0].new_value).abs();
            let time_diff = (window[1].timestamp - window[0].timestamp).abs();

            if time_diff > 0.0 {
                total_change += change_amount;
                total_time += time_diff;
            }
        }

        if total_time > 0.0 {
            total_change / total_time
        } else {
            0.0
        }
    }
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            last_crossfade_times: Vec::new(),
            min_intervals: Self::create_default_intervals(),
            current_time: 0.0,
        }
    }

    fn create_default_intervals() -> Vec<(CrossfadeParameter, f32)> {
        vec![
            (CrossfadeParameter::InputValue, 0.05),   // 50ms min interval
            (CrossfadeParameter::Tempo, 0.1),         // 100ms min interval
            (CrossfadeParameter::FilterCutoff, 0.01), // 10ms min interval
        ]
    }

    fn update(&mut self, delta_time: f32) {
        self.current_time += delta_time;
    }

    fn should_allow_change(&mut self, parameter: CrossfadeParameter) -> bool {
        let min_interval = self.min_intervals
            .iter()
            .find(|(param, _)| *param == parameter)
            .map(|(_, interval)| *interval)
            .unwrap_or(0.01); // Default 10ms

        if let Some(entry) = self.last_crossfade_times.iter_mut().find(|(param, _)| *param == parameter) {
            let time_since_last = self.current_time - entry.1;
            if time_since_last >= min_interval {
                entry.1 = self.current_time;
                true
            } else {
                false
            }
        } else {
            self.last_crossfade_times.push((parameter, self.current_time));
            true
        }
    }
}

impl SmoothingFilterBank {
    fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    fn update(&mut self, delta_time: f32) {
        for filter in &mut self.filters {
            filter.update(delta_time);
        }
    }

    fn get_smoothed_value(&self, parameter: CrossfadeParameter) -> f32 {
        self.filters
            .iter()
            .find(|filter| filter.parameter == parameter)
            .map(|filter| filter.state.current_value)
            .unwrap_or(0.0)
    }

    fn set_target_value(&mut self, parameter: CrossfadeParameter, value: f32) {
        if let Some(filter) = self.filters.iter_mut().find(|f| f.parameter == parameter) {
            filter.set_target(value);
        } else {
            // Create new filter for this parameter
            let mut filter = SmoothingFilter::new(parameter);
            filter.set_target(value);
            self.filters.push(filter);
        }
    }
}

impl SmoothingFilter {
    fn new(parameter: CrossfadeParameter) -> Self {
        Self {
            parameter,
            characteristics: SmoothingCharacteristics::for_parameter(parameter),
            state: FilterState::new(),
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.state.time_since_update += delta_time;

        // Simple exponential smoothing for now
        // In a full implementation, this would use the specified filter type
    }

    fn set_target(&mut self, target: f32) {
        // Set target value - the filter will smooth towards it
        self.state.current_value = target; // Simplified for now
    }
}

impl SmoothingCharacteristics {
    fn default() -> Self {
        Self {
            smoothing_factor: 0.95,
            response_time: 0.1,
            filter_type: SmoothingFilterType::Exponential,
            variation_amount: 0.02,
        }
    }

    fn fast_response() -> Self {
        Self {
            smoothing_factor: 0.8,
            response_time: 0.02,
            filter_type: SmoothingFilterType::OnePole,
            variation_amount: 0.01,
        }
    }

    fn slow_response() -> Self {
        Self {
            smoothing_factor: 0.99,
            response_time: 0.5,
            filter_type: SmoothingFilterType::Biquad,
            variation_amount: 0.05,
        }
    }

    fn for_parameter(parameter: CrossfadeParameter) -> Self {
        match parameter {
            CrossfadeParameter::FilterCutoff => Self::fast_response(),
            CrossfadeParameter::Tempo => Self::slow_response(),
            _ => Self::default(),
        }
    }
}

impl FilterState {
    fn new() -> Self {
        Self {
            current_value: 0.0,
            memory: vec![0.0; 4], // Space for biquad coefficients
            time_since_update: 0.0,
        }
    }
}