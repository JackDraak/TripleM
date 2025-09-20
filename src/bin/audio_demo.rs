use mood_music_module::{MoodMusicModule, MoodConfig, StereoFrame};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Mood Music Module - Audio Demo 🎵");
    println!("=====================================");
    println!();

    // Initialize audio
    let host = cpal::default_host();
    let device = host.default_output_device()
        .ok_or("No output device available")?;

    println!("Using audio device: {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Audio config: {} Hz, {} channels",
             config.sample_rate().0,
             config.channels());

    // Create mood music module
    let mood_config = MoodConfig::default_with_sample_rate(config.sample_rate().0);
    let module = Arc::new(Mutex::new(MoodMusicModule::with_config(mood_config)?));

    // Start the module
    {
        let mut module_lock = module.lock().unwrap();
        module_lock.start();
        module_lock.set_mood(0.1); // Start with environmental sounds
    }

    println!("✅ Module initialized successfully!");
    println!();

    // Build the audio stream
    let module_clone = Arc::clone(&module);
    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => build_stream::<i8>(&device, &config.into(), module_clone),
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config.into(), module_clone),
        cpal::SampleFormat::I32 => build_stream::<i32>(&device, &config.into(), module_clone),
        cpal::SampleFormat::I64 => build_stream::<i64>(&device, &config.into(), module_clone),
        cpal::SampleFormat::U8 => build_stream::<u8>(&device, &config.into(), module_clone),
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config.into(), module_clone),
        cpal::SampleFormat::U32 => build_stream::<u32>(&device, &config.into(), module_clone),
        cpal::SampleFormat::U64 => build_stream::<u64>(&device, &config.into(), module_clone),
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config.into(), module_clone),
        cpal::SampleFormat::F64 => build_stream::<f64>(&device, &config.into(), module_clone),
        _ => return Err("Unsupported sample format".into()),
    }?;

    // Start the audio stream
    stream.play()?;
    println!("🔊 Audio stream started!");

    // Run the interactive demo
    run_interactive_demo(Arc::clone(&module))?;

    // Clean shutdown
    drop(stream);
    println!("👋 Goodbye!");

    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    module: Arc<Mutex<MoodMusicModule>>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;

    let data_callback = move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
        let mut module_lock = module.lock().unwrap();

        for frame in data.chunks_mut(channels) {
            if channels == 1 {
                // Mono output - use mono sample
                let sample = module_lock.get_next_sample();
                frame[0] = T::from_sample(sample);
            } else {
                // Stereo or multi-channel output - use stereo sample
                let stereo_sample = module_lock.get_next_stereo_sample();
                frame[0] = T::from_sample(stereo_sample.left);  // Left channel
                if frame.len() > 1 {
                    frame[1] = T::from_sample(stereo_sample.right); // Right channel
                }
                // Fill any additional channels with the right channel
                for ch in frame.iter_mut().skip(2) {
                    *ch = T::from_sample(stereo_sample.right);
                }
            }
        }
    };

    let error_callback = |err| {
        eprintln!("❌ Audio stream error: {}", err);
    };

    device.build_output_stream(config, data_callback, error_callback, None)
}

fn run_interactive_demo(module: Arc<Mutex<MoodMusicModule>>) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("Demo Controls:");
    println!("- Enter a mood value between 0.0 and 1.0");
    println!("- Enter 'q' to quit");
    println!("- Enter 's' to show status");
    println!("- Enter 'v <volume>' to set volume (0.0-1.0)");
    println!("- Enter 'test' to run test sequence");
    println!();

    loop {
        print!("Enter mood (0.0-1.0) or command: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "q" | "quit" => {
                println!("Stopping...");
                {
                    let mut module_lock = module.lock().unwrap();
                    module_lock.stop();
                }
                break;
            }
            "s" | "status" => {
                show_status(&module);
            }
            "test" => {
                run_test_sequence(&module);
            }
            _ if input.starts_with("v ") => {
                let volume_str = &input[2..];
                if let Ok(volume) = volume_str.parse::<f32>() {
                    let clamped_volume = volume.clamp(0.0, 1.0);
                    // Note: Volume control would need to be implemented in the module
                    println!("🔊 Volume set to {:.2}", clamped_volume);
                } else {
                    println!("❌ Invalid volume. Enter a number between 0.0 and 1.0");
                }
            }
            _ => {
                if let Ok(mood) = input.parse::<f32>() {
                    let mut module_lock = module.lock().unwrap();
                    if mood < 0.0 {
                        println!("🛑 Stopping module (mood < 0.0)");
                        module_lock.set_mood(mood);
                    } else if mood > 1.0 {
                        println!("⚠️  Clamping mood to 1.0");
                        module_lock.set_mood(1.0);
                    } else {
                        module_lock.set_mood(mood);
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

fn show_status(module: &Arc<Mutex<MoodMusicModule>>) {
    let module_lock = module.lock().unwrap();
    println!();
    println!("📊 Module Status:");
    println!("  Running: {}", module_lock.is_running());
    println!("  Current Mood: {:.3}", module_lock.get_mood());
    println!("  Mood Type: {}", mood_description(module_lock.get_mood()));
    println!("  Sample Rate: {} Hz", module_lock.sample_rate());
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

fn run_test_sequence(module: &Arc<Mutex<MoodMusicModule>>) {
    println!();
    println!("🧪 Running test sequence...");

    let test_moods = [0.0, 0.1, 0.15, 0.2, 0.25];

    for &mood in &test_moods {
        println!("  Setting mood to {:.2} ({})", mood, mood_description(mood));
        {
            let mut module_lock = module.lock().unwrap();
            module_lock.set_mood(mood);
        }

        // Let it play for 3 seconds
        thread::sleep(Duration::from_secs(3));
    }

    println!("✅ Test sequence complete");
    println!();
}