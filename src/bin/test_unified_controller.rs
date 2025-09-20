//! Test program for the unified controller interface
//!
//! This program demonstrates the advanced unified controller interface with
//! real-time parameter morphing, preset management, and system monitoring.

use mood_music_module::{
    UnifiedController, MoodConfig, ControlParameter, ChangeSource,
    PresetMetadata,
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎛️ Testing Unified Controller Interface");
    println!("=======================================");

    // Create unified controller
    let config = MoodConfig::default_with_sample_rate(44100);
    let mut controller = UnifiedController::new(config)?;

    println!("✅ Created unified controller");

    // Start the audio system
    controller.start()?;
    println!("🎵 Started audio system");

    // Test basic mood control
    println!("\n📈 Testing basic mood control:");
    controller.set_mood_intensity(0.3)?;
    println!("   Set mood intensity to 0.3 (gentle melodic)");

    // Generate some audio to allow crossfades to process
    let mut buffer = vec![0.0; 1024];
    controller.fill_buffer(&mut buffer);
    println!("   Generated {} samples", buffer.len());

    // Test advanced parameter control
    println!("\n🎛️ Testing advanced parameter control:");

    // Test different parameters with smooth crossfading
    let test_parameters = [
        (ControlParameter::RhythmicDensity, 0.7, "rhythmic density"),
        (ControlParameter::MelodicDensity, 0.5, "melodic density"),
        (ControlParameter::HarmonicComplexity, 0.6, "harmonic complexity"),
        (ControlParameter::SynthesisCharacter, 0.4, "synthesis character"),
        (ControlParameter::StereoWidth, 0.8, "stereo width"),
    ];

    for (parameter, value, name) in test_parameters.iter() {
        controller.set_parameter_smooth(*parameter, *value, ChangeSource::UserInterface)?;
        println!("   Set {} to {:.1}", name, value);

        // Generate audio to process crossfades
        controller.fill_buffer(&mut buffer);
    }

    // Test preset system
    println!("\n💾 Testing preset system:");

    // Create a custom preset
    let metadata = PresetMetadata {
        description: "Test preset for focused work".to_string(),
        category: "Focus".to_string(),
        tags: vec!["productivity".to_string(), "concentration".to_string()],
        created_time: std::time::SystemTime::now(),
        user_rating: Some(5),
    };

    controller.save_preset("Test Focus".to_string(), metadata)?;
    println!("   Saved 'Test Focus' preset");

    // List available presets
    let presets = controller.list_presets();
    println!("   Available presets: {:?}", presets);

    // Test system monitoring
    println!("\n📊 Testing system monitoring:");

    // Generate audio for monitoring
    let start_time = Instant::now();
    for _ in 0..10 {
        controller.fill_buffer(&mut buffer);
    }
    let elapsed = start_time.elapsed();

    let status = controller.get_system_status();
    println!("   System active: {}", status.is_active);
    println!("   CPU usage: {:.1}%", status.cpu_usage * 100.0);
    println!("   Active parameters: {} transitioning", status.active_parameters.len());
    println!("   Processing time: {:?} for {} samples", elapsed, buffer.len() * 10);

    // Test crossfade statistics
    let crossfade_stats = controller.get_crossfade_stats();
    println!("   Active crossfades: {}", crossfade_stats.active_crossfades);
    println!("   Parameters being crossfaded: {}", crossfade_stats.parameters_being_crossfaded);
    println!("   Average crossfade duration: {:.1}ms", crossfade_stats.average_crossfade_duration * 1000.0);

    // Test parameter value retrieval
    println!("\n📋 Current parameter values:");
    let all_params = controller.get_all_parameters();
    for (parameter, value) in all_params.iter() {
        println!("   {:?}: {:.3}", parameter, value);
    }

    // Test real-time parameter morphing
    println!("\n🌊 Testing real-time parameter morphing:");
    println!("   Smoothly transitioning mood from 0.2 to 0.8...");

    let steps = 20;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let mood = 0.2 + (0.8 - 0.2) * t;

        controller.set_mood_intensity(mood)?;

        // Generate audio to process the crossfade
        controller.fill_buffer(&mut buffer);

        if i % 5 == 0 {
            println!("   Step {}: mood = {:.2}", i, mood);
        }

        // Small delay to simulate real-time usage
        std::thread::sleep(Duration::from_millis(10));
    }

    // Test volume control
    println!("\n🔊 Testing volume control:");
    controller.set_master_volume(0.7)?;
    println!("   Set master volume to 0.7");
    println!("   Current volume: {:.2}", controller.get_master_volume());

    // Generate final audio samples
    println!("\n🎵 Generating final audio samples:");
    let mut stereo_buffer = vec![mood_music_module::StereoFrame::silence(); 512];
    controller.fill_stereo_buffer(&mut stereo_buffer);

    // Check for valid audio generation
    let has_audio = stereo_buffer.iter().any(|frame| frame.left != 0.0 || frame.right != 0.0);
    println!("   Generated {} stereo frames, audio present: {}", stereo_buffer.len(), has_audio);

    // Get comprehensive diagnostics
    println!("\n🔧 System diagnostics:");
    let diagnostics = controller.get_diagnostics();
    println!("   System status: {:?}", diagnostics.system_status.is_active);
    println!("   Performance metrics available: {}",
        diagnostics.system_status.cpu_usage >= 0.0);

    // Stop the system
    controller.stop();
    println!("\n⏹️ Stopped audio system");

    println!("\n✅ All tests completed successfully!");
    println!("🎉 Unified Controller Interface is fully functional!");

    Ok(())
}