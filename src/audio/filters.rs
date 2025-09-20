//! Advanced filtering system for sophisticated synthesis
//!
//! This module provides various filter types with resonance support,
//! essential for creating rich timbres and dynamic sound shaping.

use std::f32::consts::PI;

/// Filter types available in the system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
}

/// State-variable filter implementation
/// This is a versatile filter that can produce multiple filter types simultaneously
#[derive(Debug, Clone)]
pub struct StateVariableFilter {
    sample_rate: f32,
    cutoff: f32,
    resonance: f32,

    // Filter state variables
    low: f32,
    high: f32,
    band: f32,
    notch: f32,

    // Filter coefficients
    f: f32,  // Frequency coefficient
    q: f32,  // Q (resonance) coefficient
}

impl StateVariableFilter {
    /// Create a new state variable filter
    pub fn new(sample_rate: f32, cutoff: f32, resonance: f32) -> Self {
        let mut filter = Self {
            sample_rate,
            cutoff: cutoff.clamp(20.0, sample_rate * 0.45),
            resonance: resonance.clamp(0.1, 10.0),
            low: 0.0,
            high: 0.0,
            band: 0.0,
            notch: 0.0,
            f: 0.0,
            q: 0.0,
        };

        filter.update_coefficients();
        filter
    }

    /// Set the cutoff frequency
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.clamp(20.0, self.sample_rate * 0.45);
        self.update_coefficients();
    }

    /// Set the resonance (Q factor)
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.1, 10.0);
        self.update_coefficients();
    }

    /// Get the current cutoff frequency
    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }

    /// Get the current resonance
    pub fn resonance(&self) -> f32 {
        self.resonance
    }

    /// Update internal filter coefficients
    fn update_coefficients(&mut self) {
        self.f = 2.0 * (PI * self.cutoff / self.sample_rate).sin();
        self.q = 1.0 / self.resonance;
    }

    /// Process a single sample through the filter
    pub fn process(&mut self, input: f32) -> FilterOutput {
        // State variable filter algorithm
        self.low += self.f * self.band;
        self.high = input - self.low - self.q * self.band;
        self.band += self.f * self.high;
        self.notch = self.high + self.low;

        FilterOutput {
            lowpass: self.low,
            highpass: self.high,
            bandpass: self.band,
            notch: self.notch,
        }
    }

    /// Process a sample and return only the specified filter type
    pub fn process_type(&mut self, input: f32, filter_type: FilterType) -> f32 {
        let output = self.process(input);
        match filter_type {
            FilterType::Lowpass => output.lowpass,
            FilterType::Highpass => output.highpass,
            FilterType::Bandpass => output.bandpass,
            FilterType::Notch => output.notch,
        }
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.low = 0.0;
        self.high = 0.0;
        self.band = 0.0;
        self.notch = 0.0;
    }
}

/// Output from state variable filter containing all filter types
#[derive(Debug, Clone, Copy)]
pub struct FilterOutput {
    pub lowpass: f32,
    pub highpass: f32,
    pub bandpass: f32,
    pub notch: f32,
}

/// Simple one-pole lowpass filter for basic filtering needs
#[derive(Debug, Clone)]
pub struct OnePoleFilter {
    cutoff: f32,
    sample_rate: f32,
    state: f32,
    coefficient: f32,
}

impl OnePoleFilter {
    /// Create a new one-pole filter
    pub fn new(sample_rate: f32, cutoff: f32) -> Self {
        let mut filter = Self {
            cutoff: cutoff.clamp(20.0, sample_rate * 0.45),
            sample_rate,
            state: 0.0,
            coefficient: 0.0,
        };
        filter.update_coefficient();
        filter
    }

    /// Set the cutoff frequency
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.clamp(20.0, self.sample_rate * 0.45);
        self.update_coefficient();
    }

    /// Update the filter coefficient
    fn update_coefficient(&mut self) {
        let rc = 1.0 / (2.0 * PI * self.cutoff);
        let dt = 1.0 / self.sample_rate;
        self.coefficient = dt / (dt + rc);
    }

    /// Process a single sample
    pub fn process(&mut self, input: f32) -> f32 {
        self.state += (input - self.state) * self.coefficient;
        self.state
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

/// Multi-mode filter that can smoothly morph between filter types
#[derive(Debug, Clone)]
pub struct MorphingFilter {
    svf: StateVariableFilter,
    morph: f32, // 0.0 = lowpass, 0.33 = bandpass, 0.66 = highpass, 1.0 = notch
}

impl MorphingFilter {
    /// Create a new morphing filter
    pub fn new(sample_rate: f32, cutoff: f32, resonance: f32) -> Self {
        Self {
            svf: StateVariableFilter::new(sample_rate, cutoff, resonance),
            morph: 0.0,
        }
    }

    /// Set the morph parameter (0.0 to 1.0)
    /// 0.0 = lowpass, 0.33 = bandpass, 0.66 = highpass, 1.0 = notch
    pub fn set_morph(&mut self, morph: f32) {
        self.morph = morph.clamp(0.0, 1.0);
    }

    /// Set cutoff frequency
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.svf.set_cutoff(cutoff);
    }

    /// Set resonance
    pub fn set_resonance(&mut self, resonance: f32) {
        self.svf.set_resonance(resonance);
    }

    /// Process a sample with smooth morphing between filter types
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.svf.process(input);

        if self.morph <= 0.33 {
            // Morph between lowpass and bandpass
            let t = self.morph / 0.33;
            output.lowpass * (1.0 - t) + output.bandpass * t
        } else if self.morph <= 0.66 {
            // Morph between bandpass and highpass
            let t = (self.morph - 0.33) / 0.33;
            output.bandpass * (1.0 - t) + output.highpass * t
        } else {
            // Morph between highpass and notch
            let t = (self.morph - 0.66) / 0.34;
            output.highpass * (1.0 - t) + output.notch * t
        }
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.svf.reset();
    }
}

/// Biquad filter implementation for more precise frequency response
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    // Coefficients
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,

    // State
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter
    pub fn new() -> Self {
        Self {
            a0: 1.0,
            a1: 0.0,
            a2: 0.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Configure as lowpass filter
    pub fn lowpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let mut filter = Self::new();
        filter.set_lowpass(sample_rate, cutoff, q);
        filter
    }

    /// Configure as highpass filter
    pub fn highpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let mut filter = Self::new();
        filter.set_highpass(sample_rate, cutoff, q);
        filter
    }

    /// Configure as bandpass filter
    pub fn bandpass(sample_rate: f32, center: f32, q: f32) -> Self {
        let mut filter = Self::new();
        filter.set_bandpass(sample_rate, center, q);
        filter
    }

    /// Set lowpass coefficients
    pub fn set_lowpass(&mut self, sample_rate: f32, cutoff: f32, q: f32) {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize coefficients
        self.a0 = b0 / a0;
        self.a1 = b1 / a0;
        self.a2 = b2 / a0;
        self.b1 = a1 / a0;
        self.b2 = a2 / a0;
    }

    /// Set highpass coefficients
    pub fn set_highpass(&mut self, sample_rate: f32, cutoff: f32, q: f32) {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize coefficients
        self.a0 = b0 / a0;
        self.a1 = b1 / a0;
        self.a2 = b2 / a0;
        self.b1 = a1 / a0;
        self.b2 = a2 / a0;
    }

    /// Set bandpass coefficients
    pub fn set_bandpass(&mut self, sample_rate: f32, center: f32, q: f32) {
        let omega = 2.0 * PI * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = q * alpha;
        let b1 = 0.0;
        let b2 = -q * alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Normalize coefficients
        self.a0 = b0 / a0;
        self.a1 = b1 / a0;
        self.a2 = b2 / a0;
        self.b1 = a1 / a0;
        self.b2 = a2 / a0;
    }

    /// Process a single sample
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.a0 * input + self.a1 * self.x1 + self.a2 * self.x2
                   - self.b1 * self.y1 - self.b2 * self.y2;

        // Update state
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Filter bank for parallel processing with multiple filters
#[derive(Debug, Clone)]
pub struct FilterBank {
    filters: Vec<StateVariableFilter>,
    mix_levels: Vec<f32>,
}

impl FilterBank {
    /// Create a new filter bank
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            mix_levels: Vec::new(),
        }
    }

    /// Add a filter to the bank
    pub fn add_filter(&mut self, filter: StateVariableFilter, mix_level: f32) {
        self.filters.push(filter);
        self.mix_levels.push(mix_level.clamp(0.0, 1.0));
    }

    /// Process a sample through all filters and mix the results
    pub fn process(&mut self, input: f32, filter_types: &[FilterType]) -> f32 {
        let mut output = 0.0;
        let mut total_mix = 0.0;

        for (i, filter) in self.filters.iter_mut().enumerate() {
            if i < filter_types.len() && i < self.mix_levels.len() {
                let filtered = filter.process_type(input, filter_types[i]);
                output += filtered * self.mix_levels[i];
                total_mix += self.mix_levels[i];
            }
        }

        if total_mix > 0.0 {
            output / total_mix
        } else {
            input
        }
    }

    /// Reset all filters in the bank
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

/// Convenience functions for creating common filter configurations
pub mod presets {
    use super::*;

    /// Create a warm lowpass filter suitable for ambient pads
    pub fn warm_lowpass(sample_rate: f32) -> StateVariableFilter {
        StateVariableFilter::new(sample_rate, 800.0, 0.7)
    }

    /// Create a bright highpass filter for percussion
    pub fn bright_highpass(sample_rate: f32) -> StateVariableFilter {
        StateVariableFilter::new(sample_rate, 2000.0, 1.2)
    }

    /// Create a resonant bandpass for lead synthesis
    pub fn resonant_bandpass(sample_rate: f32) -> StateVariableFilter {
        StateVariableFilter::new(sample_rate, 1000.0, 2.0)
    }

    /// Create a notch filter for removing specific frequencies
    pub fn notch_filter(sample_rate: f32, frequency: f32) -> StateVariableFilter {
        StateVariableFilter::new(sample_rate, frequency, 5.0)
    }

    /// Create a morphing filter setup for electronic music
    pub fn electronic_morph(sample_rate: f32) -> MorphingFilter {
        MorphingFilter::new(sample_rate, 1500.0, 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_variable_filter() {
        let mut filter = StateVariableFilter::new(44100.0, 1000.0, 1.0);

        // Test with impulse
        let output = filter.process(1.0);
        assert!(output.lowpass.abs() > 0.0);
        assert!(output.highpass.abs() > 0.0);
        assert!(output.bandpass.abs() > 0.0);
        assert!(output.notch.abs() > 0.0);
    }

    #[test]
    fn test_one_pole_filter() {
        let mut filter = OnePoleFilter::new(44100.0, 1000.0);

        // Test basic functionality
        let output1 = filter.process(1.0);
        let output2 = filter.process(0.0);

        assert!(output1 > 0.0);
        assert!(output2 < output1); // Should decay towards 0
    }

    #[test]
    fn test_morphing_filter() {
        let mut filter = MorphingFilter::new(44100.0, 1000.0, 1.0);

        // Test morphing
        filter.set_morph(0.0); // Lowpass
        let lp_output = filter.process(1.0);

        filter.set_morph(0.66); // Highpass
        let hp_output = filter.process(1.0);

        assert_ne!(lp_output, hp_output);
    }

    #[test]
    fn test_biquad_filter() {
        let mut filter = BiquadFilter::lowpass(44100.0, 1000.0, 1.0);

        let output = filter.process(1.0);
        assert!(output.is_finite());
    }

    #[test]
    fn test_filter_presets() {
        let sample_rate = 44100.0;

        let _warm = presets::warm_lowpass(sample_rate);
        let _bright = presets::bright_highpass(sample_rate);
        let _resonant = presets::resonant_bandpass(sample_rate);
        let _notch = presets::notch_filter(sample_rate, 60.0);
        let _morph = presets::electronic_morph(sample_rate);

        // Just test that they can be created without panicking
    }

    #[test]
    fn test_filter_bank() {
        let mut bank = FilterBank::new();

        let filter1 = StateVariableFilter::new(44100.0, 500.0, 1.0);
        let filter2 = StateVariableFilter::new(44100.0, 2000.0, 1.0);

        bank.add_filter(filter1, 0.5);
        bank.add_filter(filter2, 0.5);

        let output = bank.process(1.0, &[FilterType::Lowpass, FilterType::Highpass]);
        assert!(output.is_finite());
    }
}