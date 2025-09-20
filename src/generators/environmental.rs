use crate::config::MoodConfig;
use crate::error::Result;
use crate::generators::{MoodGenerator, GeneratorState};
use crate::audio::{utils, StereoFrame};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Environmental sounds generator (0.0-0.25 mood range)
/// Generates ambient environmental sounds like ocean, wind, rain, etc.
#[derive(Debug)]
pub struct EnvironmentalGenerator {
    intensity: f32,
    sample_rate: f32,
    time: f64,

    // Pink noise state for natural environmental sounds
    pink_noise_state: [f32; 7],

    // Random number generator
    rng: StdRng,

    // Wave simulation parameters
    wave_phase: f32,
    wave_frequency: f32,
    wind_phase: f32,
    wind_frequency: f32,

    // Pattern cycling
    current_soundscape: usize,
    soundscape_timer: f32,
    soundscape_duration: f32,

    // Enhanced stereo spatial parameters
    stereo_pan: f32,           // Current pan position (-1.0 to 1.0)
    stereo_width: f32,         // Stereo width factor (0.0 to 1.0)
    pan_phase: f32,            // Phase for automatic panning
    pan_frequency: f32,        // How fast the automatic panning moves

    // Multi-layer stereo positioning
    left_layer_pan: f32,       // Independent left layer panning
    right_layer_pan: f32,      // Independent right layer panning
    left_layer_phase: f32,     // Left layer phase for movement
    right_layer_phase: f32,    // Right layer phase for movement

    // Enhanced ocean wave parameters
    wave_layers: [f32; 4],     // Multiple wave layer phases
    wave_speeds: [f32; 4],     // Different wave speeds for realism
}

impl EnvironmentalGenerator {
    pub fn new(config: &MoodConfig) -> Result<Self> {
        let mut rng = StdRng::from_entropy();
        let wave_frequency = rng.gen_range(0.1..0.3);
        let wind_frequency = rng.gen_range(0.05..0.15);
        let pan_frequency = rng.gen_range(0.008..0.02); // Much slower, realistic ocean rhythm

        Ok(Self {
            intensity: 0.0,
            sample_rate: config.sample_rate as f32,
            time: 0.0,
            pink_noise_state: [0.0; 7],
            rng,
            wave_phase: 0.0,
            wave_frequency, // Slow wave movement
            wind_phase: 0.0,
            wind_frequency, // Even slower wind
            current_soundscape: 0,
            soundscape_timer: 0.0,
            soundscape_duration: config.pattern_cycle_lengths[0], // 3 minutes default
            stereo_pan: 0.0,
            stereo_width: 1.0, // Maximum stereo width for dramatic effect
            pan_phase: 0.0,
            pan_frequency,

            // Initialize multi-layer stereo
            left_layer_pan: -0.7,  // Start left layer more to the left
            right_layer_pan: 0.7,  // Start right layer more to the right
            left_layer_phase: 0.0,
            right_layer_phase: std::f32::consts::PI, // Start out of phase

            // Initialize ocean wave layers with realistic speeds
            wave_layers: [0.0, 0.0, 0.0, 0.0],
            wave_speeds: [0.15, 0.25, 0.35, 0.6], // Much slower, more realistic ocean rhythm
        })
    }

    /// Generate ocean wave sounds
    fn generate_ocean_waves(&mut self, _time: f64) -> f32 {
        let base_wave = (self.wave_phase * 2.0 * std::f32::consts::PI).sin();
        let wave_variation = (self.wave_phase * 0.7 * 2.0 * std::f32::consts::PI).sin() * 0.3;

        // Add some pink noise for texture
        let white_noise = utils::white_noise(&mut self.rng);
        let pink_noise = utils::pink_noise_simple(white_noise, &mut self.pink_noise_state);

        let wave_sound = (base_wave + wave_variation) * 0.8 + pink_noise * 0.4;

        // Update wave phase
        self.wave_phase += self.wave_frequency / self.sample_rate;
        if self.wave_phase >= 1.0 {
            self.wave_phase -= 1.0;
        }

        wave_sound
    }

    /// Generate wind sounds
    fn generate_wind(&mut self, _time: f64) -> f32 {
        let white_noise = utils::white_noise(&mut self.rng);
        let pink_noise = utils::pink_noise_simple(white_noise, &mut self.pink_noise_state);

        // Low frequency modulation for wind gusts
        let wind_modulation = (self.wind_phase * 2.0 * std::f32::consts::PI).sin();
        let wind_intensity = 0.3 + wind_modulation * 0.2;

        self.wind_phase += self.wind_frequency / self.sample_rate;
        if self.wind_phase >= 1.0 {
            self.wind_phase -= 1.0;
        }

        pink_noise * wind_intensity * 1.0
    }

    /// Generate forest/nature sounds
    fn generate_forest(&mut self, time: f64) -> f32 {
        let white_noise = utils::white_noise(&mut self.rng);
        let pink_noise = utils::pink_noise_simple(white_noise, &mut self.pink_noise_state);

        // Occasional bird-like chirps (very subtle)
        let chirp_trigger = self.rng.gen::<f32>();
        let chirp = if chirp_trigger < 0.001 { // Very rare
            let chirp_freq = self.rng.gen_range(1000.0..3000.0);
            let chirp_phase = time * chirp_freq * 2.0 * std::f64::consts::PI;
            (chirp_phase.sin() as f32) * 0.1 * (-time as f32 * 10.0).exp()
        } else {
            0.0
        };

        pink_noise * 0.6 + chirp
    }

    /// Generate rain sounds
    fn generate_rain(&mut self, time: f64) -> f32 {
        let white_noise = utils::white_noise(&mut self.rng);

        // Rain is close to white noise but with some filtering
        let rain_base = white_noise * 0.4;

        // Add occasional raindrop impacts
        let drop_trigger = self.rng.gen::<f32>();
        let drop = if drop_trigger < 0.005 { // Occasional drops
            let drop_freq = self.rng.gen_range(200.0..800.0);
            let drop_phase = time * drop_freq * 2.0 * std::f64::consts::PI;
            (drop_phase.sin() as f32) * 0.05 * (-time as f32 * 20.0).exp()
        } else {
            0.0
        };

        (rain_base + drop) * 1.5
    }

    /// Update soundscape cycling
    fn update_soundscape(&mut self, delta_time: f32) {
        self.soundscape_timer += delta_time;

        if self.soundscape_timer >= self.soundscape_duration {
            self.soundscape_timer = 0.0;
            self.current_soundscape = (self.current_soundscape + 1) % 4;

            // Randomize some parameters for the new soundscape
            self.wave_frequency = self.rng.gen_range(0.1..0.3);
            self.wind_frequency = self.rng.gen_range(0.05..0.15);
        }
    }

    /// Generate realistic ocean waves with natural rhythm and dynamics
    fn generate_ocean_waves_stereo(&mut self, _time: f64) -> StereoFrame {
        let delta_time = 1.0 / self.sample_rate;

        // Update wave layers with realistic, slow speeds
        for i in 0..4 {
            self.wave_layers[i] += self.wave_speeds[i] * delta_time;
        }

        // Calculate wave intensity using realistic ocean patterns
        // Main wave cycle (very slow, 20-40 second periods)
        let main_wave_cycle = (self.wave_layers[0] * 2.0 * std::f32::consts::PI).sin();

        // Secondary wave pattern (medium period, 8-15 seconds)
        let secondary_wave = (self.wave_layers[1] * 2.0 * std::f32::consts::PI).sin();

        // Create realistic wave envelope: mostly calm with brief crescendos
        let base_intensity = 0.15; // Calm baseline
        let wave_build = (main_wave_cycle * 0.5 + 0.5).powf(3.0); // Exponential build
        let crescendo_factor = if wave_build > 0.7 {
            // Brief, sharp crescendo like real waves
            ((wave_build - 0.7) / 0.3).powf(2.0) * 0.8
        } else {
            0.0
        };

        let current_intensity = base_intensity + crescendo_factor;

        // Generate base ocean noise
        let base_noise = utils::white_noise(&mut self.rng);

        // Generate multi-layered stereo ocean sound
        let mut left_sample = 0.0;
        let mut right_sample = 0.0;

        // Layer 1: Deep ocean swell (centered, constant)
        let deep_swell = base_noise * 0.3 * current_intensity * 0.8;
        left_sample += deep_swell;
        right_sample += deep_swell;

        // Layer 2: Mid-frequency waves (slight left bias)
        let mid_waves = base_noise * 0.4 * current_intensity;
        let mid_envelope = (secondary_wave * 0.3 + 0.7); // Gentle variation
        left_sample += mid_waves * mid_envelope * 0.8;
        right_sample += mid_waves * mid_envelope * 0.6;

        // Layer 3: Surface activity (slight right bias)
        let surface_noise = utils::white_noise(&mut self.rng) * 0.35;
        let surface_envelope = (self.wave_layers[2] * 2.0 * std::f32::consts::PI).sin() * 0.2 + 0.8;
        left_sample += surface_noise * surface_envelope * current_intensity * 0.6;
        right_sample += surface_noise * surface_envelope * current_intensity * 0.8;

        // Layer 4: Foam and splash during crescendos only
        if crescendo_factor > 0.2 {
            let foam_noise = utils::white_noise(&mut self.rng) * 0.6;
            let splash_pan = (self.wave_layers[3] * 2.0 * std::f32::consts::PI).sin() * 0.6;
            let left_splash = foam_noise * crescendo_factor * (1.0 - splash_pan.abs());
            let right_splash = foam_noise * crescendo_factor * (splash_pan.abs());

            if splash_pan < 0.0 {
                left_sample += left_splash * 0.9;
                right_sample += right_splash * 0.3;
            } else {
                left_sample += left_splash * 0.3;
                right_sample += right_splash * 0.9;
            }
        }

        // Apply pink noise filtering for natural character
        let pink_left = utils::pink_noise_simple(left_sample, &mut self.pink_noise_state);
        let pink_right = utils::pink_noise_simple(right_sample, &mut self.pink_noise_state);

        // Subtle cross-channel bleeding for realism
        let final_left = pink_left * 0.9 + pink_right * 0.1;
        let final_right = pink_right * 0.9 + pink_left * 0.1;

        StereoFrame::new(final_left * self.intensity, final_right * self.intensity)
    }

    /// Generate wind with stereo enhancement
    fn generate_wind_stereo(&mut self, time: f64) -> f32 {
        self.generate_wind(time)
    }

    /// Generate forest sounds with stereo enhancement
    fn generate_forest_stereo(&mut self, time: f64) -> f32 {
        self.generate_forest(time)
    }

    /// Generate rain with stereo enhancement
    fn generate_rain_stereo(&mut self, time: f64) -> f32 {
        self.generate_rain(time)
    }

    /// Apply dramatic stereo positioning to a mono sample
    fn apply_stereo_positioning(&self, mono_sample: f32) -> StereoFrame {
        // Use the left and right layer pans for much more dynamic positioning
        let left_pan = self.left_layer_pan.clamp(-1.0, 1.0);
        let right_pan = self.right_layer_pan.clamp(-1.0, 1.0);

        // Calculate dramatic left and right gains
        let left_radians = (left_pan + 1.0) * 0.5 * std::f32::consts::PI * 0.5;
        let right_radians = (right_pan + 1.0) * 0.5 * std::f32::consts::PI * 0.5;

        // Create dual-layer stereo effect
        let left_layer_gain = left_radians.cos() * self.stereo_width;
        let right_layer_gain = right_radians.sin() * self.stereo_width;

        // Apply cross-mixing for fuller stereo field
        let left_sample = mono_sample * (left_layer_gain * 0.8 + right_layer_gain * 0.2) * self.intensity;
        let right_sample = mono_sample * (right_layer_gain * 0.8 + left_layer_gain * 0.2) * self.intensity;

        StereoFrame::new(left_sample, right_sample)
    }
}

impl MoodGenerator for EnvironmentalGenerator {
    fn generate_sample(&mut self, time: f64) -> f32 {
        self.time = time;
        let delta_time = 1.0 / self.sample_rate;
        self.update_soundscape(delta_time);

        if self.intensity <= 0.0 {
            return 0.0;
        }

        let sample = match self.current_soundscape {
            0 => self.generate_ocean_waves(time),
            1 => self.generate_wind(time),
            2 => self.generate_forest(time),
            3 => self.generate_rain(time),
            _ => 0.0,
        };

        // Apply intensity scaling
        sample * self.intensity
    }

    fn generate_batch(&mut self, output: &mut [f32], start_time: f64) {
        let sample_duration = 1.0 / self.sample_rate as f64;

        for (i, sample) in output.iter_mut().enumerate() {
            let time = start_time + i as f64 * sample_duration;
            *sample = self.generate_sample(time);
        }
    }

    fn generate_stereo_sample(&mut self, time: f64) -> StereoFrame {
        self.time = time;
        let delta_time = 1.0 / self.sample_rate;
        self.update_soundscape(delta_time);

        if self.intensity <= 0.0 {
            return StereoFrame::silence();
        }

        // Update gentle, realistic stereo pan automation
        self.pan_phase += self.pan_frequency * delta_time;
        self.left_layer_phase += self.pan_frequency * 0.8 * delta_time;
        self.right_layer_phase += self.pan_frequency * 1.2 * delta_time;

        // Generate subtle, natural stereo positioning
        self.stereo_pan = (self.pan_phase * 2.0 * std::f32::consts::PI).sin() * 0.4; // Gentle panning
        self.left_layer_pan = -0.5 + (self.left_layer_phase * 2.0 * std::f32::consts::PI).sin() * 0.3;
        self.right_layer_pan = 0.5 + (self.right_layer_phase * 2.0 * std::f32::consts::PI).sin() * 0.3;

        // Generate stereo samples based on soundscape
        match self.current_soundscape {
            0 => self.generate_ocean_waves_stereo(time), // Direct stereo generation
            1 => {
                let sample = self.generate_wind_stereo(time);
                self.apply_stereo_positioning(sample)
            },
            2 => {
                let sample = self.generate_forest_stereo(time);
                self.apply_stereo_positioning(sample)
            },
            3 => {
                let sample = self.generate_rain_stereo(time);
                self.apply_stereo_positioning(sample)
            },
            _ => StereoFrame::silence(),
        }
    }

    fn generate_stereo_batch(&mut self, output: &mut [StereoFrame], start_time: f64) {
        let sample_duration = 1.0 / self.sample_rate as f64;

        for (i, frame) in output.iter_mut().enumerate() {
            let time = start_time + i as f64 * sample_duration;
            *frame = self.generate_stereo_sample(time);
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
        self.pink_noise_state = [0.0; 7];
        self.wave_phase = 0.0;
        self.wind_phase = 0.0;
        self.current_soundscape = 0;
        self.soundscape_timer = 0.0;
        self.rng = StdRng::from_entropy();
    }

    fn update_focus_parameters(&mut self) {
        // When environmental is the focus, enhance natural variation
        self.wave_frequency *= self.rng.gen_range(0.8..1.2);
        self.wind_frequency *= self.rng.gen_range(0.9..1.1);
    }

    fn get_state(&self) -> GeneratorState {
        let soundscape_names = ["Ocean Waves", "Wind", "Forest", "Rain"];
        let current_name = soundscape_names.get(self.current_soundscape)
            .unwrap_or(&"Unknown")
            .to_string();

        GeneratorState {
            name: "Environmental".to_string(),
            intensity: self.intensity,
            is_active: self.intensity > 0.0,
            current_pattern: current_name,
            pattern_progress: self.soundscape_timer / self.soundscape_duration,
            cpu_usage: 0.1, // Relatively low CPU usage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MoodConfig;

    #[test]
    fn test_environmental_generator_creation() {
        let config = MoodConfig::default();
        let generator = EnvironmentalGenerator::new(&config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_sample_generation() {
        let config = MoodConfig::default();
        let mut generator = EnvironmentalGenerator::new(&config).unwrap();

        generator.set_intensity(1.0);

        let sample = generator.generate_sample(0.0);
        assert!(sample.is_finite());

        // Should be relatively quiet environmental sound
        assert!(sample.abs() <= 1.0);
    }

    #[test]
    fn test_batch_generation() {
        let config = MoodConfig::default();
        let mut generator = EnvironmentalGenerator::new(&config).unwrap();

        generator.set_intensity(0.5);

        let mut buffer = vec![0.0; 512];
        generator.generate_batch(&mut buffer, 0.0);

        // Should have generated audio
        assert!(buffer.iter().any(|&x| x != 0.0));
        assert!(buffer.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_intensity_control() {
        let config = MoodConfig::default();
        let mut generator = EnvironmentalGenerator::new(&config).unwrap();

        generator.set_intensity(0.0);
        let quiet_sample = generator.generate_sample(0.0);
        assert_eq!(quiet_sample, 0.0);

        generator.set_intensity(1.0);
        let loud_sample = generator.generate_sample(0.0);
        assert!(loud_sample.abs() > 0.0);
    }

    #[test]
    fn test_soundscape_cycling() {
        let config = MoodConfig::default();
        let mut generator = EnvironmentalGenerator::new(&config).unwrap();

        generator.set_intensity(1.0);

        let initial_soundscape = generator.current_soundscape;

        // Manually advance timer to trigger soundscape change
        generator.soundscape_timer = generator.soundscape_duration + 1.0;
        generator.update_soundscape(1.0);

        assert_ne!(generator.current_soundscape, initial_soundscape);
    }
}