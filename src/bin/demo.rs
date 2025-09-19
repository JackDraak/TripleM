use mood_music_module::{MoodMusicModule, MoodConfig};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Mood Music Module Demo 🎵");
    println!("================================");
    println!();

    // Create the mood music module
    println!("Initializing mood music module...");
    let config = MoodConfig::default();
    println!("Sample Rate: {} Hz", config.sample_rate);
    println!("Buffer Size: {} samples", config.buffer_size);
    println!("Transition Duration: {:.1}s", config.transition_duration);
    println!();

    match MoodMusicModule::with_config(config) {
        Ok(mut module) => {
            println!("✅ Module initialized successfully!");
            run_demo(&mut module)?;
        }
        Err(e) => {
            println!("❌ Failed to initialize module: {}", e);
            println!();
            println!("Note: This is expected as the full generator implementation");
            println!("is still in progress. The basic structure is working!");
        }
    }

    Ok(())
}

fn run_demo(module: &mut MoodMusicModule) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("Demo Controls:");
    println!("- Enter a mood value between 0.0 and 1.0");
    println!("- Enter 'q' to quit");
    println!("- Enter 's' to show status");
    println!("- Enter 'r' to reset");
    println!();

    module.start();
    println!("🎵 Audio generation started!");

    loop {
        print!("Enter mood (0.0-1.0) or command: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "q" | "quit" => {
                println!("Stopping...");
                module.stop();
                break;
            }
            "s" | "status" => {
                show_status(module);
            }
            "r" | "reset" => {
                println!("Resetting module...");
                module.stop();
                thread::sleep(Duration::from_millis(100));
                module.start();
                module.set_mood(0.0);
                println!("✅ Reset complete");
            }
            "test" => {
                run_test_sequence(module);
            }
            _ => {
                if let Ok(mood) = input.parse::<f32>() {
                    if mood < 0.0 {
                        println!("🛑 Stopping module (mood < 0.0)");
                        module.set_mood(mood);
                    } else if mood > 1.0 {
                        println!("⚠️  Clamping mood to 1.0");
                        module.set_mood(1.0);
                    } else {
                        module.set_mood(mood);
                        println!("🎵 Set mood to {:.2} ({})", mood, mood_description(mood));
                    }
                } else {
                    println!("❌ Invalid input. Enter a number between 0.0 and 1.0");
                }
            }
        }
    }

    Ok(())
}

fn show_status(module: &MoodMusicModule) {
    println!();
    println!("📊 Module Status:");
    println!("  Running: {}", module.is_running());
    println!("  Current Mood: {:.3}", module.get_mood());
    println!("  Mood Type: {}", mood_description(module.get_mood()));
    println!("  Sample Rate: {} Hz", module.sample_rate());
    println!();
}

fn mood_description(mood: f32) -> &'static str {
    match mood {
        m if m < 0.0 => "Stopped",
        m if m <= 0.25 => "Environmental (0.0-0.25)",
        m if m <= 0.5 => "Gentle Melodic (0.25-0.5)",
        m if m <= 0.75 => "Active Ambient (0.5-0.75)",
        _ => "EDM Style (0.75-1.0)",
    }
}

fn run_test_sequence(module: &mut MoodMusicModule) {
    println!();
    println!("🧪 Running test sequence...");

    let test_moods = [0.0, 0.25, 0.5, 0.75, 1.0];

    for &mood in &test_moods {
        println!("  Setting mood to {:.2} ({})", mood, mood_description(mood));
        module.set_mood(mood);

        // Simulate some audio generation
        for _ in 0..1000 {
            let _sample = module.get_next_sample();
        }

        thread::sleep(Duration::from_millis(500));
    }

    println!("✅ Test sequence complete");
    println!();
}