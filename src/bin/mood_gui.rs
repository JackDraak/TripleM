use std::sync::{Arc, Mutex};
use std::time::Duration;
use eframe::egui;
use mood_music_module::{MoodMusicModule, MoodConfig};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_title("🎵 Mood Music Controller")
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "Mood Music Controller",
        options,
        Box::new(|_cc| Box::new(MoodMusicApp::new())),
    )
}

struct MoodMusicApp {
    mood_value: f32,
    volume: f32,
    is_playing: bool,
    current_mood_name: String,
    audio_module: Option<Arc<Mutex<MoodMusicModule>>>,
    audio_thread: Option<std::thread::JoinHandle<()>>,
    error_message: Option<String>,

    // Audio stats
    current_rms: f32,
    generator_states: Vec<GeneratorDisplayState>,
}

#[derive(Clone)]
struct GeneratorDisplayState {
    name: String,
    intensity: f32,
    is_active: bool,
    current_pattern: String,
    pattern_progress: f32,
}

impl MoodMusicApp {
    fn new() -> Self {
        Self {
            mood_value: 0.35,
            volume: 0.8,
            is_playing: false,
            current_mood_name: "Gentle Melodic".to_string(),
            audio_module: None,
            audio_thread: None,
            error_message: None,
            current_rms: 0.0,
            generator_states: Vec::new(),
        }
    }

    fn start_audio(&mut self) {
        if self.audio_module.is_some() {
            return; // Already running
        }

        match self.initialize_audio() {
            Ok(module) => {
                self.audio_module = Some(module.clone());
                self.start_audio_thread(module);
                self.is_playing = true;
                self.error_message = None;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to start audio: {}", e));
                self.is_playing = false;
            }
        }
    }

    fn stop_audio(&mut self) {
        if let Some(module) = &self.audio_module {
            let module_lock = module.lock().unwrap();
            module_lock.stop();
        }

        self.audio_module = None;
        if let Some(handle) = self.audio_thread.take() {
            let _ = handle.join();
        }
        self.is_playing = false;
    }

    fn initialize_audio(&self) -> Result<Arc<Mutex<MoodMusicModule>>, Box<dyn std::error::Error>> {
        let config = MoodConfig::default();
        let mut module = MoodMusicModule::with_config(config)?;

        module.start();
        module.set_mood(self.mood_value);
        module.set_volume(self.volume);

        Ok(Arc::new(Mutex::new(module)))
    }

    fn start_audio_thread(&mut self, module: Arc<Mutex<MoodMusicModule>>) {
        let module_clone = module.clone();
        let module_for_thread = module.clone();

        self.audio_thread = Some(std::thread::spawn(move || {
            // Audio playback using CPAL
            use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

            let host = cpal::default_host();
            let device = host.default_output_device().expect("No output device available");
            let config = device.default_output_config().expect("No default output config");

            let _sample_rate = config.sample_rate().0;
            let channels = config.channels() as usize;

            let stream = device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut module_lock = module_clone.lock().unwrap();

                    for frame in data.chunks_mut(channels) {
                        let sample = module_lock.get_next_sample();
                        for channel in frame.iter_mut() {
                            *channel = sample;
                        }
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            ).expect("Failed to build audio stream");

            stream.play().expect("Failed to play stream");

            // Keep the stream alive
            loop {
                std::thread::sleep(Duration::from_millis(100));

                // Check if module is still valid (stop condition)
                if Arc::strong_count(&module_for_thread) <= 1 {
                    break;
                }
            }
        }));
    }

    fn update_mood(&mut self, new_mood: f32) {
        self.mood_value = new_mood;
        self.current_mood_name = self.get_mood_name(new_mood);

        if let Some(module) = &self.audio_module {
            let mut module_lock = module.lock().unwrap();
            module_lock.set_mood(new_mood);
        }
    }

    fn update_volume(&mut self, new_volume: f32) {
        self.volume = new_volume;

        if let Some(module) = &self.audio_module {
            let mut module_lock = module.lock().unwrap();
            module_lock.set_volume(new_volume);
        }
    }

    fn get_mood_name(&self, mood: f32) -> String {
        match mood {
            m if m <= 0.25 => "Environmental",
            m if m <= 0.5 => "Gentle Melodic",
            m if m <= 0.75 => "Active Ambient",
            _ => "EDM Style",
        }.to_string()
    }

    fn get_mood_description(&self, mood: f32) -> String {
        match mood {
            m if m <= 0.25 => "Natural sounds: ocean, wind, forest, rain",
            m if m <= 0.5 => "Relaxing spa music with gentle melodies",
            m if m <= 0.75 => "Productivity-focused ambient with rhythm",
            _ => "High-energy electronic dance music",
        }.to_string()
    }

    fn update_generator_states(&mut self) {
        if let Some(module) = &self.audio_module {
            if let Ok(_module_lock) = module.try_lock() {
                // Update RMS and other stats if available
                self.current_rms = 0.05; // Placeholder - would need actual RMS calculation

                // Update generator states (simplified)
                self.generator_states = vec![
                    GeneratorDisplayState {
                        name: "Environmental".to_string(),
                        intensity: if self.mood_value <= 0.25 { 1.0 } else { 0.2 },
                        is_active: self.mood_value <= 0.4,
                        current_pattern: "Ocean Waves".to_string(),
                        pattern_progress: 0.3,
                    },
                    GeneratorDisplayState {
                        name: "Gentle Melodic".to_string(),
                        intensity: if self.mood_value > 0.25 && self.mood_value <= 0.5 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.2 && self.mood_value <= 0.6,
                        current_pattern: "C Major - I chord".to_string(),
                        pattern_progress: 0.6,
                    },
                    GeneratorDisplayState {
                        name: "Active Ambient".to_string(),
                        intensity: if self.mood_value > 0.5 && self.mood_value <= 0.75 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.4 && self.mood_value <= 0.8,
                        current_pattern: "Steady - 120 BPM".to_string(),
                        pattern_progress: 0.8,
                    },
                    GeneratorDisplayState {
                        name: "EDM Style".to_string(),
                        intensity: if self.mood_value > 0.75 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.7,
                        current_pattern: "Intro - 128 BPM - 25%".to_string(),
                        pattern_progress: 0.25,
                    },
                ];
            }
        }
    }
}

impl eframe::App for MoodMusicApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update generator states periodically
        self.update_generator_states();

        // Request repaint for smooth animations
        ctx.request_repaint_after(Duration::from_millis(100));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🎵 Mood Music Controller");
            ui.separator();

            // Error display
            if let Some(error) = &self.error_message {
                ui.colored_label(egui::Color32::RED, format!("❌ {}", error));
                ui.separator();
            }

            // Play/Stop button
            ui.horizontal(|ui| {
                let button_text = if self.is_playing { "⏹ Stop" } else { "▶ Play" };
                if ui.button(button_text).clicked() {
                    if self.is_playing {
                        self.stop_audio();
                    } else {
                        self.start_audio();
                    }
                }

                ui.label(if self.is_playing {
                    "🔊 Playing"
                } else {
                    "🔇 Stopped"
                });
            });

            ui.separator();

            // Mood slider
            ui.vertical(|ui| {
                ui.label("🎭 Mood");
                ui.horizontal(|ui| {
                    ui.label("0.0");
                    let mood_response = ui.add(
                        egui::Slider::new(&mut self.mood_value, 0.0..=1.0)
                            .step_by(0.01)
                            .show_value(false)
                    );
                    ui.label("1.0");

                    if mood_response.changed() {
                        self.update_mood(self.mood_value);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(format!("Current: {:.2}", self.mood_value));
                    ui.separator();
                    ui.strong(&self.current_mood_name);
                });

                ui.label(self.get_mood_description(self.mood_value));
            });

            ui.separator();

            // Volume slider
            ui.vertical(|ui| {
                ui.label("🔊 Volume");
                ui.horizontal(|ui| {
                    ui.label("🔇");
                    let volume_response = ui.add(
                        egui::Slider::new(&mut self.volume, 0.0..=1.0)
                            .step_by(0.01)
                            .show_value(false)
                    );
                    ui.label("🔊");

                    if volume_response.changed() {
                        self.update_volume(self.volume);
                    }
                });

                ui.label(format!("Level: {:.0}%", self.volume * 100.0));
            });

            ui.separator();

            // Generator status display
            if self.is_playing && !self.generator_states.is_empty() {
                ui.label("🎹 Generator Status");

                for state in &self.generator_states {
                    ui.horizontal(|ui| {
                        let color = if state.is_active {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::GRAY
                        };

                        ui.colored_label(color, &state.name);
                        ui.separator();

                        // Intensity bar
                        let bar_width = 60.0;
                        let bar_height = 8.0;
                        let (rect, _) = ui.allocate_exact_size(
                            egui::Vec2::new(bar_width, bar_height),
                            egui::Sense::hover()
                        );

                        ui.painter().rect_filled(
                            rect,
                            2.0,
                            egui::Color32::DARK_GRAY
                        );

                        let fill_width = rect.width() * state.intensity;
                        let fill_rect = egui::Rect::from_min_size(
                            rect.min,
                            egui::Vec2::new(fill_width, rect.height())
                        );

                        ui.painter().rect_filled(
                            fill_rect,
                            2.0,
                            if state.is_active { egui::Color32::GREEN } else { egui::Color32::GRAY }
                        );

                        ui.label(format!("{:.0}%", state.intensity * 100.0));
                    });
                }
            }

            ui.separator();

            // Instructions
            ui.small("🎵 Adjust mood (0.0-1.0) to switch between music styles:");
            ui.small("   • 0.0-0.25: Environmental sounds");
            ui.small("   • 0.25-0.5: Gentle melodic music");
            ui.small("   • 0.5-0.75: Active ambient music");
            ui.small("   • 0.75-1.0: EDM-style music");
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_audio();
    }
}