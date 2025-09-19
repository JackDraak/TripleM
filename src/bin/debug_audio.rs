use mood_music_module::{MoodMusicModule, MoodConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug Audio Test");
    println!("==================");

    let config = MoodConfig::default();
    let mut module = MoodMusicModule::with_config(config.clone())?;

    println!("Module created, starting...");
    module.start();
    println!("Module started: {}", module.is_running());

    println!("Setting mood to 0.35...");
    module.set_mood(0.35);
    println!("Current mood: {:.3}", module.get_mood());

    // Let the transition system update by generating a few samples
    println!("Letting transition system settle...");
    for _ in 0..1000 {
        let _ = module.get_next_sample();
    }

    println!();
    println!("Testing direct generator access...");

    // Let's test the environmental generator directly
    use mood_music_module::generators::{EnvironmentalGenerator, MoodGenerator};

    let mut env_gen = EnvironmentalGenerator::new(&config)?;
    env_gen.set_intensity(1.0);

    println!("Direct environmental generator test:");
    for i in 0..10 {
        let sample = env_gen.generate_sample(i as f64 / 44100.0);
        println!("  Sample {}: {:.6}", i, sample);
    }

    println!();
    println!("Testing full module pipeline:");
    for i in 0..10 {
        let sample = module.get_next_sample();
        println!("  Module sample {}: {:.6}", i, sample);
    }

    Ok(())
}