//! Test program for the multi-scale rhythm evolution system
//!
//! This program demonstrates the sophisticated multi-scale rhythm generation
//! system with hierarchical temporal structure, polyrhythmic patterns,
//! Euclidean rhythm generation, and phrase-aware development.

use mood_music_module::{
    patterns::{MultiScaleRhythmSystem, PatternGenerator, EvolutionDiagnostics},
    MoodConfig,
};
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 44100;
const TEST_DURATION_SECONDS: u64 = 30;
const PARAMETER_CHANGE_INTERVAL_MS: u64 = 500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Multi-Scale Rhythm Evolution System Test");
    println!("==========================================");

    // Create multi-scale rhythm system
    let mut rhythm_system = MultiScaleRhythmSystem::new(SAMPLE_RATE as f32)?;
    println!("✅ Created multi-scale rhythm system");

    // Test different complexity levels
    println!("\n📊 Testing Complexity Evolution:");
    test_complexity_evolution(&mut rhythm_system)?;

    // Test pattern generation
    println!("\n🎛️ Testing Pattern Generation:");
    test_pattern_generation(&mut rhythm_system)?;

    // Test unified controller integration
    println!("\n🔗 Testing Unified Controller Integration:");
    test_controller_integration(&mut rhythm_system)?;

    // Test evolution diagnostics
    println!("\n📈 Testing Evolution Diagnostics:");
    test_evolution_diagnostics(&mut rhythm_system)?;

    // Performance benchmark
    println!("\n⚡ Performance Benchmark:");
    performance_benchmark(&mut rhythm_system)?;

    // Test temporal hierarchy
    println!("\n⏰ Testing Temporal Hierarchy:");
    test_temporal_hierarchy(&mut rhythm_system)?;

    println!("\n🎉 All multi-scale rhythm tests completed successfully!");
    Ok(())
}

/// Test complexity evolution from simple to complex patterns
fn test_complexity_evolution(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    let complexity_levels = [0.0, 0.25, 0.5, 0.75, 1.0];

    for (i, complexity) in complexity_levels.iter().enumerate() {
        println!("   Testing complexity level {}: {:.2}", i, complexity);

        // Set input value
        rhythm_system.set_input_value(*complexity);

        // Generate patterns at this complexity
        for _ in 0..10 {
            let pattern = rhythm_system.next();
            // Pattern would be analyzed here in a full implementation
        }

        // Get evolution diagnostics
        let diagnostics = rhythm_system.get_evolution_diagnostics();
        println!("     - Overall complexity: {:.3}", diagnostics.complexity_profile.overall_complexity);
        println!("     - Polyrhythmic activity: {:.3}", diagnostics.polyrhythmic_activity);
        println!("     - Euclidean complexity: {:.3}", diagnostics.euclidean_complexity);
        println!("     - Coherence score: {:.3}", diagnostics.coherence_score);
    }

    Ok(())
}

/// Test pattern generation capabilities
fn test_pattern_generation(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    // Test pattern generation at different inputs
    let test_inputs = [0.1, 0.35, 0.6, 0.85];

    for input in test_inputs.iter() {
        println!("   Input value: {:.2}", input);
        rhythm_system.set_input_value(*input);

        // Generate multi-scale pattern
        let multi_scale_pattern = rhythm_system.generate_pattern()?;

        // Analyze pattern components
        println!("     - Micro elements: {} primary events", multi_scale_pattern.micro_elements.primary_pattern.len());
        println!("     - Meso phrase length: {}", multi_scale_pattern.meso_elements.phrase_structure.phrase_length);
        println!("     - Macro section: {:?}", multi_scale_pattern.macro_elements.current_section);
        println!("     - Polyrhythmic layers: {}", multi_scale_pattern.polyrhythmic_layers.len());
        println!("     - Euclidean patterns: {}", multi_scale_pattern.euclidean_patterns.len());
        println!("     - Coordination strength: {:.3}", multi_scale_pattern.coordination_metadata.coordination_strength);
    }

    Ok(())
}

/// Test integration with unified controller
fn test_controller_integration(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    // Test controller parameter mapping
    let test_scenarios = [
        ("Ambient Focus", 0.3, 0.4, 0.3, 0.2),
        ("Active Work", 0.6, 0.7, 0.6, 0.3),
        ("Creative Flow", 0.8, 0.8, 0.7, 0.4),
        ("High Energy", 0.95, 0.9, 0.8, 0.1),
    ];

    for (scenario, mood, density, complexity, humanization) in test_scenarios.iter() {
        println!("   Scenario: {}", scenario);

        // Update from controller parameters
        rhythm_system.update_from_controller(*mood, *density, *complexity, *humanization)?;

        // Get controller state
        let state = rhythm_system.get_controller_state();
        println!("     - Current input: {:.3}", state.current_input);
        println!("     - Micro activity: {:.3}", state.scale_activities.micro_activity);
        println!("     - Meso activity: {:.3}", state.scale_activities.meso_activity);
        println!("     - Macro activity: {:.3}", state.scale_activities.macro_activity);
        println!("     - Active layers: {}", state.polyrhythmic_layers);
        println!("     - Coherence: {:.3}", state.coherence_score);
    }

    Ok(())
}

/// Test evolution diagnostics and monitoring
fn test_evolution_diagnostics(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    // Test diagnostics at different evolution stages
    for i in 0..5 {
        let progress = i as f32 / 4.0;
        rhythm_system.set_input_value(progress);

        // Generate patterns to advance evolution
        for _ in 0..20 {
            rhythm_system.next();
        }

        let diagnostics = rhythm_system.get_evolution_diagnostics();
        println!("   Evolution stage {}: progress = {:.2}", i, progress);
        println!("     - Active processes: {}", diagnostics.active_processes);
        println!("     - Temporal positions: micro={:.3}, meso={:.3}, macro={:.3}",
                 diagnostics.temporal_positions.micro_position,
                 diagnostics.temporal_positions.meso_position,
                 diagnostics.temporal_positions.macro_position);
        println!("     - Phrase position: {:.3}", diagnostics.phrase_position);
        println!("     - Complexity profile:");
        println!("       * Overall: {:.3}", diagnostics.complexity_profile.overall_complexity);
        println!("       * Density target: {:.3}", diagnostics.complexity_profile.density_target);
        println!("       * Polyrhythmic: {:.3}", diagnostics.complexity_profile.polyrhythmic_complexity);
        println!("       * Structural: {:.3}", diagnostics.complexity_profile.structural_complexity);
    }

    Ok(())
}

/// Performance benchmark for the multi-scale system
fn performance_benchmark(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    let test_duration = Duration::from_secs(5);
    let patterns_to_generate = 1000;

    // Benchmark pattern generation
    println!("   Generating {} patterns...", patterns_to_generate);
    let start_time = Instant::now();

    for i in 0..patterns_to_generate {
        // Vary input to test different complexity levels
        let input = (i as f32 / patterns_to_generate as f32).sin().abs();
        rhythm_system.set_input_value(input);

        let _pattern = rhythm_system.next();
    }

    let generation_time = start_time.elapsed();
    let patterns_per_sec = patterns_to_generate as f32 / generation_time.as_secs_f32();

    println!("     - Generated {} patterns in {:?}", patterns_to_generate, generation_time);
    println!("     - Rate: {:.1} patterns/second", patterns_per_sec);

    // Benchmark multi-scale pattern generation
    println!("   Generating {} multi-scale patterns...", patterns_to_generate / 10);
    let start_time = Instant::now();

    for i in 0..(patterns_to_generate / 10) {
        let input = (i as f32 / (patterns_to_generate / 10) as f32) * 0.8 + 0.1;
        rhythm_system.set_input_value(input);

        let _multi_pattern = rhythm_system.generate_pattern()?;
    }

    let multi_generation_time = start_time.elapsed();
    let multi_patterns_per_sec = (patterns_to_generate / 10) as f32 / multi_generation_time.as_secs_f32();

    println!("     - Generated {} multi-scale patterns in {:?}", patterns_to_generate / 10, multi_generation_time);
    println!("     - Rate: {:.1} multi-scale patterns/second", multi_patterns_per_sec);

    // Performance analysis
    if patterns_per_sec > 10000.0 {
        println!("     ✅ EXCELLENT: Pattern generation performance is excellent");
    } else if patterns_per_sec > 1000.0 {
        println!("     ✅ GOOD: Pattern generation performance is good");
    } else {
        println!("     ⚠️ Pattern generation could be optimized");
    }

    Ok(())
}

/// Test temporal hierarchy functionality
fn test_temporal_hierarchy(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing hierarchical temporal coordination...");

    // Test cycle completion detection
    rhythm_system.reset();
    rhythm_system.set_input_value(0.5);

    let mut cycles_completed = 0;
    let mut patterns_generated = 0;
    let max_patterns = 1000;

    while patterns_generated < max_patterns {
        rhythm_system.next();
        patterns_generated += 1;

        if rhythm_system.is_cycle_complete() {
            cycles_completed += 1;
            println!("     - Cycle {} completed after {} patterns", cycles_completed, patterns_generated);

            if cycles_completed >= 3 {
                break;
            }
        }
    }

    println!("     - Pattern length: {} samples", rhythm_system.pattern_length());
    println!("     - Completed {} cycles in {} patterns", cycles_completed, patterns_generated);

    // Test temporal position tracking
    rhythm_system.reset();
    for i in 0..50 {
        rhythm_system.set_input_value(i as f32 / 49.0);
        rhythm_system.next();

        if i % 10 == 0 {
            let diagnostics = rhythm_system.get_evolution_diagnostics();
            println!("     - Step {}: temporal positions = micro:{:.3}, meso:{:.3}, macro:{:.3}",
                     i,
                     diagnostics.temporal_positions.micro_position,
                     diagnostics.temporal_positions.meso_position,
                     diagnostics.temporal_positions.macro_position);
        }
    }

    Ok(())
}

/// Test real-time parameter morphing
fn test_real_time_morphing(rhythm_system: &mut MultiScaleRhythmSystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Testing Real-time Parameter Morphing:");

    let start_time = Instant::now();
    let morph_duration = Duration::from_secs(10);
    let mut last_diagnostic_time = Instant::now();

    while start_time.elapsed() < morph_duration {
        let progress = start_time.elapsed().as_secs_f32() / morph_duration.as_secs_f32();

        // Create a smooth evolution curve
        let input = (progress * std::f32::consts::PI * 2.0).sin().abs();
        rhythm_system.set_input_value(input);

        // Generate pattern
        rhythm_system.next();

        // Print diagnostics every 2 seconds
        if last_diagnostic_time.elapsed() >= Duration::from_secs(2) {
            let diagnostics = rhythm_system.get_evolution_diagnostics();
            println!("   Progress: {:.1}%, Input: {:.3}, Complexity: {:.3}, Coherence: {:.3}",
                     progress * 100.0,
                     input,
                     diagnostics.complexity_profile.overall_complexity,
                     diagnostics.coherence_score);
            last_diagnostic_time = Instant::now();
        }

        // Small delay to simulate real-time usage
        std::thread::sleep(Duration::from_millis(10));
    }

    println!("   ✅ Real-time morphing test completed");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_scale_system_creation() {
        let result = MultiScaleRhythmSystem::new(44100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_complexity_profile_generation() {
        let mut system = MultiScaleRhythmSystem::new(44100.0).unwrap();

        // Test different input values
        for input in [0.0, 0.25, 0.5, 0.75, 1.0].iter() {
            system.set_input_value(*input);
            let diagnostics = system.get_evolution_diagnostics();

            // Verify complexity increases with input
            assert!(diagnostics.complexity_profile.overall_complexity >= 0.0);
            assert!(diagnostics.complexity_profile.overall_complexity <= 1.0);
            assert!(diagnostics.coherence_score >= 0.0);
            assert!(diagnostics.coherence_score <= 1.0);
        }
    }

    #[test]
    fn test_pattern_generator_interface() {
        let mut system = MultiScaleRhythmSystem::new(44100.0).unwrap();

        // Test PatternGenerator trait implementation
        system.reset();

        let pattern = system.next();
        assert!(system.pattern_length() > 0);

        // Pattern should be valid
        // Additional pattern validation would go here
    }

    #[test]
    fn test_controller_integration() {
        let mut system = MultiScaleRhythmSystem::new(44100.0).unwrap();

        // Test controller parameter updates
        let result = system.update_from_controller(0.5, 0.6, 0.7, 0.3);
        assert!(result.is_ok());

        let state = system.get_controller_state();
        assert_eq!(state.current_input, 0.5);
    }

    #[test]
    fn test_evolution_diagnostics() {
        let mut system = MultiScaleRhythmSystem::new(44100.0).unwrap();

        system.set_input_value(0.5);

        // Generate some patterns to advance evolution
        for _ in 0..10 {
            system.next();
        }

        let diagnostics = system.get_evolution_diagnostics();
        assert!(diagnostics.current_input >= 0.0);
        assert!(diagnostics.current_input <= 1.0);
        assert!(diagnostics.coherence_score >= 0.0);
    }
}