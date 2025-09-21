use std::sync::{Arc, Mutex};
use std::time::Duration;
use eframe::egui;
use mood_music_module::{UnifiedController, MoodConfig, StereoFrame, ControlParameter, ChangeSource};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([450.0, 250.0])
            .with_min_inner_size([400.0, 220.0])
            .with_max_inner_size([600.0, 400.0])
            .with_title("🎵 Mood Music Controller")
            .with_resizable(true),
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
    audio_module: Option<Arc<Mutex<UnifiedController>>>,
    audio_thread: Option<std::thread::JoinHandle<()>>,
    audio_shutdown: Option<Arc<std::sync::atomic::AtomicBool>>,
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
            audio_shutdown: None,
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
        println!("🔇 Stopping audio...");

        // Signal shutdown to audio thread
        if let Some(shutdown_flag) = &self.audio_shutdown {
            shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Stop the module
        if let Some(module) = &self.audio_module {
            if let Ok(module_lock) = module.lock() {
                module_lock.stop();
            }
        }

        // Wait for audio thread to finish with timeout
        if let Some(handle) = self.audio_thread.take() {
            println!("⏳ Waiting for audio thread to finish...");
            match handle.join() {
                Ok(_) => println!("✅ Audio thread finished cleanly"),
                Err(_) => println!("⚠️ Audio thread finished with error"),
            }
        }

        self.audio_module = None;
        self.audio_shutdown = None;
        self.is_playing = false;
        println!("🔇 Audio stopped");
    }

    fn initialize_audio(&self) -> Result<Arc<Mutex<UnifiedController>>, Box<dyn std::error::Error>> {
        let config = MoodConfig::default();
        let mut controller = UnifiedController::new(config)?;

        controller.start()?;
        controller.set_mood_intensity(self.mood_value)?;
        controller.set_master_volume(self.volume)?;

        Ok(Arc::new(Mutex::new(controller)))
    }

    fn start_audio_thread(&mut self, module: Arc<Mutex<UnifiedController>>) {
        let module_clone = module.clone();
        let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let shutdown_flag_for_callback = shutdown_flag.clone();
        let shutdown_flag_for_loop = shutdown_flag.clone();

        self.audio_shutdown = Some(shutdown_flag);

        self.audio_thread = Some(std::thread::spawn(move || {
            println!("🎵 Audio thread starting...");

            // Audio playback using CPAL
            use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

            let host = cpal::default_host();
            let device = host.default_output_device().expect("No output device available");
            let config = device.default_output_config().expect("No default output config");

            let _sample_rate = config.sample_rate().0;
            let channels = config.channels() as usize;

            let stream_result = device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Check for shutdown signal
                    if shutdown_flag_for_callback.load(std::sync::atomic::Ordering::Relaxed) {
                        // Fill with silence and return
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    if let Ok(mut module_lock) = module_clone.try_lock() {
                        if channels == 1 {
                            // Mono output - use unified controller's buffer fill
                            module_lock.fill_buffer(data);
                        } else {
                            // Stereo or multi-channel output
                            // Convert interleaved buffer to stereo frames
                            let frame_count = data.len() / channels;
                            let mut stereo_buffer: Vec<StereoFrame> = vec![StereoFrame::silence(); frame_count];

                            // Fill stereo buffer using UnifiedController
                            module_lock.fill_stereo_buffer(&mut stereo_buffer);

                            // Convert back to interleaved format
                            for (i, stereo_frame) in stereo_buffer.iter().enumerate() {
                                let base_idx = i * channels;
                                if base_idx + 1 < data.len() {
                                    data[base_idx] = stereo_frame.left;      // Left channel
                                    data[base_idx + 1] = stereo_frame.right; // Right channel

                                    // Fill any additional channels with the right channel
                                    for ch_idx in 2..channels {
                                        if base_idx + ch_idx < data.len() {
                                            data[base_idx + ch_idx] = stereo_frame.right;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // If we can't lock, fill with silence
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            );

            match stream_result {
                Ok(stream) => {
                    if let Err(e) = stream.play() {
                        eprintln!("Failed to play stream: {}", e);
                        return;
                    }

                    // Keep the stream alive until shutdown
                    while !shutdown_flag_for_loop.load(std::sync::atomic::Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10));
                    }

                    println!("🎵 Audio thread received shutdown signal");
                    drop(stream); // Explicitly drop the stream
                }
                Err(e) => {
                    eprintln!("Failed to build audio stream: {}", e);
                }
            }

            println!("🎵 Audio thread finishing...");
        }));
    }

    fn update_mood(&mut self, new_mood: f32) {
        self.mood_value = new_mood;
        self.current_mood_name = self.get_mood_name(new_mood);

        if let Some(module) = &self.audio_module {
            if let Ok(mut module_lock) = module.lock() {
                if let Err(e) = module_lock.set_parameter_smooth(ControlParameter::MoodIntensity, new_mood, ChangeSource::UserInterface) {
                    self.error_message = Some(format!("Failed to update mood: {}", e));
                }
            }
        }
    }

    fn update_volume(&mut self, new_volume: f32) {
        self.volume = new_volume;

        if let Some(module) = &self.audio_module {
            if let Ok(mut module_lock) = module.lock() {
                if let Err(e) = module_lock.set_parameter_smooth(ControlParameter::MasterVolume, new_volume, ChangeSource::UserInterface) {
                    self.error_message = Some(format!("Failed to update volume: {}", e));
                }
            }
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
            if let Ok(module_lock) = module.try_lock() {
                // Get actual system status from UnifiedController
                let status = module_lock.get_system_status();
                self.current_rms = if status.is_active { self.mood_value * 0.1 } else { 0.0 }; // Simple approximation

                // Update generator states based on actual controller state
                let mood_active = status.active_parameters.contains(&ControlParameter::MoodIntensity);
                let tempo_value = if mood_active { 120.0 + (self.mood_value * 40.0) } else { 120.0 };

                self.generator_states = vec![
                    GeneratorDisplayState {
                        name: "Environmental".to_string(),
                        intensity: self.mood_value * if self.mood_value <= 0.25 { 1.0 } else { 0.2 },
                        is_active: self.mood_value <= 0.4 && status.is_active,
                        current_pattern: "Ocean Waves".to_string(),
                        pattern_progress: 0.3,
                    },
                    GeneratorDisplayState {
                        name: "Gentle Melodic".to_string(),
                        intensity: self.mood_value * if self.mood_value > 0.25 && self.mood_value <= 0.5 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.2 && self.mood_value <= 0.6 && status.is_active,
                        current_pattern: "C Major - I chord".to_string(),
                        pattern_progress: 0.6,
                    },
                    GeneratorDisplayState {
                        name: "Active Ambient".to_string(),
                        intensity: self.mood_value * if self.mood_value > 0.5 && self.mood_value <= 0.75 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.4 && self.mood_value <= 0.8 && status.is_active,
                        current_pattern: format!("Steady - {:.0} BPM", tempo_value),
                        pattern_progress: 0.8,
                    },
                    GeneratorDisplayState {
                        name: "EDM Style".to_string(),
                        intensity: self.mood_value * if self.mood_value > 0.75 { 1.0 } else { 0.2 },
                        is_active: self.mood_value > 0.7 && status.is_active,
                        current_pattern: format!("Intro - {:.0} BPM - {:.0}%", tempo_value + 8.0, 25.0),
                        pattern_progress: 0.25,
                    },
                ];
            } else {
                // Fallback to simplified display if status unavailable
                self.current_rms = 0.05;
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
            // Compact header
            ui.horizontal(|ui| {
                ui.heading("🎵 Mood Music");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Exit button
                    if ui.small_button("❌").on_hover_text("Close application").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }

                    // Play/Stop button
                    let button_text = if self.is_playing { "⏹" } else { "▶" };
                    let button_color = if self.is_playing {
                        egui::Color32::from_rgb(220, 100, 100)
                    } else {
                        egui::Color32::from_rgb(100, 220, 100)
                    };

                    if ui.add(egui::Button::new(button_text).fill(button_color))
                        .on_hover_text(if self.is_playing { "Stop playback" } else { "Start playback" })
                        .clicked()
                    {
                        if self.is_playing {
                            self.stop_audio();
                        } else {
                            self.start_audio();
                        }
                    }
                });
            });

            // Error display (compact)
            if let Some(error) = &self.error_message {
                ui.small(format!("❌ {}", error));
            }

            ui.add_space(5.0);

            // Main content in scrollable area
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Mood slider (compact)
                ui.horizontal(|ui| {
                    ui.label("🎭");
                    ui.vertical(|ui| {
                        let mood_response = ui.add(
                            egui::Slider::new(&mut self.mood_value, 0.0..=1.0)
                                .step_by(0.01)
                                .show_value(false)
                                .text("Mood")
                        );

                        if mood_response.changed() {
                            self.update_mood(self.mood_value);
                        }

                        ui.horizontal(|ui| {
                            ui.small(format!("{:.2}", self.mood_value));
                            ui.separator();
                            ui.small(&self.current_mood_name);
                        });
                    });
                });

                ui.add_space(3.0);

                // Volume slider (compact)
                ui.horizontal(|ui| {
                    ui.label("🔊");
                    ui.vertical(|ui| {
                        let volume_response = ui.add(
                            egui::Slider::new(&mut self.volume, 0.0..=1.0)
                                .step_by(0.01)
                                .show_value(false)
                                .text("Volume")
                        );

                        if volume_response.changed() {
                            self.update_volume(self.volume);
                        }

                        ui.small(format!("{:.0}%", self.volume * 100.0));
                    });
                });

                ui.add_space(5.0);

                // Generator status display (compact)
                if self.is_playing && !self.generator_states.is_empty() {
                    ui.small("🎹 Generators:");

                    for state in &self.generator_states {
                        ui.horizontal(|ui| {
                            let color = if state.is_active {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::GRAY
                            };

                            ui.small_button(&state.name[..3]).on_hover_text(&state.name); // Show 3-letter abbreviation

                            // Mini intensity bar
                            let bar_width = 40.0;
                            let bar_height = 6.0;
                            let (rect, _) = ui.allocate_exact_size(
                                egui::Vec2::new(bar_width, bar_height),
                                egui::Sense::hover()
                            );

                            ui.painter().rect_filled(rect, 1.0, egui::Color32::DARK_GRAY);

                            let fill_width = rect.width() * state.intensity;
                            let fill_rect = egui::Rect::from_min_size(
                                rect.min,
                                egui::Vec2::new(fill_width, rect.height())
                            );

                            ui.painter().rect_filled(fill_rect, 1.0, color);
                            ui.small(format!("{:.0}%", state.intensity * 100.0));
                        });
                    }
                }

                ui.add_space(5.0);

                // Compact instructions
                ui.small("🎵 Mood ranges:");
                ui.horizontal(|ui| {
                    ui.small("0.0-0.25: Environmental");
                    ui.separator();
                    ui.small("0.25-0.5: Melodic");
                });
                ui.horizontal(|ui| {
                    ui.small("0.5-0.75: Ambient");
                    ui.separator();
                    ui.small("0.75-1.0: EDM");
                });
            }); // End of ScrollArea
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        println!("🎵 Shutting down Mood Music Controller...");
        self.stop_audio();
        println!("✅ Mood Music Controller shut down cleanly");
    }
}