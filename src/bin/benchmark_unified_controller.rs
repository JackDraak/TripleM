//! Performance benchmark for the unified controller interface
//!
//! This benchmark tests the performance characteristics of the unified controller
//! including audio generation, parameter changes, and cross-fade performance.

use mood_music_module::{
    UnifiedController, MoodConfig, ControlParameter, ChangeSource,
    StereoFrame,
};
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 44100;
const BUFFER_SIZE: usize = 512;
const BENCHMARK_DURATION_SECONDS: u64 = 10;
const PARAMETER_CHANGE_INTERVAL_MS: u64 = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Unified Controller Performance Benchmark");
    println!("============================================");

    // Setup
    let config = MoodConfig::default_with_sample_rate(SAMPLE_RATE);
    let mut controller = UnifiedController::new(config)?;
    controller.start()?;

    println!("📊 Configuration:");
    println!("   Sample Rate: {} Hz", SAMPLE_RATE);
    println!("   Buffer Size: {} samples", BUFFER_SIZE);
    println!("   Benchmark Duration: {} seconds", BENCHMARK_DURATION_SECONDS);
    println!("   Parameter Change Interval: {} ms", PARAMETER_CHANGE_INTERVAL_MS);

    // Prepare benchmark data
    let total_samples = SAMPLE_RATE as u64 * BENCHMARK_DURATION_SECONDS;
    let total_buffers = total_samples / BUFFER_SIZE as u64;
    let parameter_change_interval = Duration::from_millis(PARAMETER_CHANGE_INTERVAL_MS);

    let mut buffer = vec![0.0; BUFFER_SIZE];
    let mut stereo_buffer = vec![StereoFrame::silence(); BUFFER_SIZE];

    println!("\n🎛️ Starting benchmark...");

    // Benchmark 1: Pure audio generation
    println!("\n📈 Benchmark 1: Pure Audio Generation");
    let start_time = Instant::now();
    let mut samples_generated = 0u64;

    for _ in 0..total_buffers {
        controller.fill_buffer(&mut buffer);
        samples_generated += BUFFER_SIZE as u64;
    }

    let pure_audio_duration = start_time.elapsed();
    let pure_audio_rate = samples_generated as f64 / pure_audio_duration.as_secs_f64();
    let realtime_ratio = pure_audio_rate / SAMPLE_RATE as f64;

    println!("   Generated {} samples in {:?}", samples_generated, pure_audio_duration);
    println!("   Rate: {:.0} samples/sec ({:.1}x realtime)", pure_audio_rate, realtime_ratio);

    // Benchmark 2: Audio generation with parameter changes
    println!("\n🎛️ Benchmark 2: Audio + Parameter Changes");
    let start_time = Instant::now();
    let mut last_param_change = Instant::now();
    let mut param_changes = 0;
    samples_generated = 0;

    let test_parameters = [
        ControlParameter::MoodIntensity,
        ControlParameter::RhythmicDensity,
        ControlParameter::MelodicDensity,
        ControlParameter::HarmonicComplexity,
        ControlParameter::SynthesisCharacter,
    ];

    for buffer_idx in 0..total_buffers {
        // Change parameters periodically
        if last_param_change.elapsed() >= parameter_change_interval {
            let param = test_parameters[param_changes % test_parameters.len()];
            let value = (buffer_idx as f32 / total_buffers as f32).sin().abs();

            if let Err(e) = controller.set_parameter_smooth(param, value, ChangeSource::UserInterface) {
                eprintln!("Parameter change error: {}", e);
            }

            param_changes += 1;
            last_param_change = Instant::now();
        }

        controller.fill_buffer(&mut buffer);
        samples_generated += BUFFER_SIZE as u64;
    }

    let param_change_duration = start_time.elapsed();
    let param_change_rate = samples_generated as f64 / param_change_duration.as_secs_f64();
    let param_realtime_ratio = param_change_rate / SAMPLE_RATE as f64;

    println!("   Generated {} samples with {} parameter changes in {:?}",
             samples_generated, param_changes, param_change_duration);
    println!("   Rate: {:.0} samples/sec ({:.1}x realtime)", param_change_rate, param_realtime_ratio);

    // Benchmark 3: Stereo generation
    println!("\n🎵 Benchmark 3: Stereo Audio Generation");
    let start_time = Instant::now();
    samples_generated = 0;

    for _ in 0..total_buffers {
        controller.fill_stereo_buffer(&mut stereo_buffer);
        samples_generated += BUFFER_SIZE as u64;
    }

    let stereo_duration = start_time.elapsed();
    let stereo_rate = samples_generated as f64 / stereo_duration.as_secs_f64();
    let stereo_realtime_ratio = stereo_rate / SAMPLE_RATE as f64;

    println!("   Generated {} stereo samples in {:?}", samples_generated, stereo_duration);
    println!("   Rate: {:.0} samples/sec ({:.1}x realtime)", stereo_rate, stereo_realtime_ratio);

    // Benchmark 4: Cross-fade stress test
    println!("\n🌊 Benchmark 4: Cross-fade Stress Test");
    let start_time = Instant::now();
    let mut rapid_param_changes = 0;
    samples_generated = 0;

    for buffer_idx in 0..total_buffers / 4 { // Shorter test for stress
        // Rapid parameter changes to stress cross-fade system
        for (i, &param) in test_parameters.iter().enumerate() {
            let value = ((buffer_idx as f32 + i as f32) * 0.1).sin().abs();
            if let Err(e) = controller.set_parameter_smooth(param, value, ChangeSource::UserInterface) {
                eprintln!("Rapid parameter change error: {}", e);
            }
            rapid_param_changes += 1;
        }

        controller.fill_buffer(&mut buffer);
        samples_generated += BUFFER_SIZE as u64;
    }

    let stress_duration = start_time.elapsed();
    let stress_rate = samples_generated as f64 / stress_duration.as_secs_f64();
    let stress_realtime_ratio = stress_rate / SAMPLE_RATE as f64;

    println!("   Generated {} samples with {} rapid parameter changes in {:?}",
             samples_generated, rapid_param_changes, stress_duration);
    println!("   Rate: {:.0} samples/sec ({:.1}x realtime)", stress_rate, stress_realtime_ratio);

    // Audio quality check
    println!("\n🎵 Audio Quality Check:");
    controller.fill_buffer(&mut buffer);
    let max_amplitude = buffer.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let rms = (buffer.iter().map(|&x| x * x).sum::<f32>() / buffer.len() as f32).sqrt();
    let has_audio = buffer.iter().any(|&x| x.abs() > 0.001);

    // System status and statistics
    println!("\n📊 Final System Status:");
    let status = controller.get_system_status();
    let stats = controller.get_crossfade_stats();

    println!("   CPU Usage: {:.1}%", status.cpu_usage * 100.0);
    println!("   Active Parameters: {}", status.active_parameters.len());
    println!("   Active Crossfades: {}", stats.active_crossfades);
    println!("   Parameters Being Crossfaded: {}", stats.parameters_being_crossfaded);
    println!("   Average Crossfade Duration: {:.1}ms", stats.average_crossfade_duration * 1000.0);
    println!("   Crossfade CPU Usage: {:.3}%", stats.crossfade_cpu_usage * 100.0);

    println!("   Audio Present: {}", has_audio);
    println!("   Max Amplitude: {:.4}", max_amplitude);
    println!("   RMS Level: {:.4}", rms);
    println!("   RMS dB: {:.1}", 20.0 * rms.log10());

    // Performance summary
    println!("\n🏆 Performance Summary:");
    println!("   Pure Audio:    {:.1}x realtime", realtime_ratio);
    println!("   With Params:   {:.1}x realtime", param_realtime_ratio);
    println!("   Stereo Audio:  {:.1}x realtime", stereo_realtime_ratio);
    println!("   Stress Test:   {:.1}x realtime", stress_realtime_ratio);

    let min_performance = realtime_ratio.min(param_realtime_ratio).min(stereo_realtime_ratio).min(stress_realtime_ratio);

    if min_performance > 10.0 {
        println!("\n✅ EXCELLENT: System can handle 10+ times realtime performance");
    } else if min_performance > 5.0 {
        println!("\n✅ GOOD: System can handle 5+ times realtime performance");
    } else if min_performance > 2.0 {
        println!("\n✅ ACCEPTABLE: System can handle 2+ times realtime performance");
    } else if min_performance > 1.0 {
        println!("\n⚠️  MARGINAL: System barely meets realtime requirements");
    } else {
        println!("\n❌ POOR: System cannot meet realtime requirements");
    }

    println!("\n🎯 Memory efficiency:");
    let crossfade_overhead = stats.crossfade_cpu_usage / status.cpu_usage;
    println!("   Crossfade overhead: {:.1}% of total CPU", crossfade_overhead * 100.0);

    // Stop the system
    controller.stop();

    println!("\n🎉 Benchmark completed successfully!");
    Ok(())
}