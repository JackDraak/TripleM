use mood_music_module::{MoodMusicModule, MoodConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Audio Test - Checking Output Levels");
    println!("======================================");

    let config = MoodConfig::default();
    let mut module = MoodMusicModule::with_config(config)?;

    module.start();
    module.set_mood(0.1); // Environmental sounds

    println!("Testing audio generation for 5 seconds...");
    println!("Generating samples and showing levels:");

    let mut max_amplitude = 0.0_f32;
    let mut min_amplitude = 0.0_f32;
    let mut sample_count = 0;
    let mut sum_squared = 0.0_f32;

    // Test for 5 seconds at 44.1kHz
    for i in 0..(44100 * 5) {
        let sample = module.get_next_sample();

        max_amplitude = max_amplitude.max(sample);
        min_amplitude = min_amplitude.min(sample);
        sum_squared += sample * sample;
        sample_count += 1;

        // Show progress every second
        if i % 44100 == 0 {
            let rms = (sum_squared / sample_count as f32).sqrt();
            println!("Second {}: Max: {:.4}, Min: {:.4}, RMS: {:.6}",
                     i / 44100 + 1, max_amplitude, min_amplitude, rms);
        }
    }

    let final_rms = (sum_squared / sample_count as f32).sqrt();

    println!();
    println!("📊 Final Statistics:");
    println!("  Samples Generated: {}", sample_count);
    println!("  Max Amplitude: {:.6}", max_amplitude);
    println!("  Min Amplitude: {:.6}", min_amplitude);
    println!("  RMS Level: {:.6}", final_rms);
    println!("  Peak-to-Peak: {:.6}", max_amplitude - min_amplitude);

    if final_rms < 0.001 {
        println!("⚠️  WARNING: Audio levels very low (RMS < 0.001)");
        println!("   This might explain why you can't hear anything!");
    } else if final_rms > 0.1 {
        println!("✅ Audio levels look good (RMS > 0.1)");
    } else {
        println!("🔊 Audio levels moderate (0.001 < RMS < 0.1)");
    }

    if max_amplitude.abs() < 0.0001 {
        println!("❌ ERROR: No audio output detected!");
        println!("   The generator might not be working correctly.");
    } else {
        println!("✅ Audio output detected");
    }

    Ok(())
}