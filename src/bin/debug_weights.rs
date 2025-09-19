use mood_music_module::audio::mixer::MoodWeights;

fn main() {
    println!("Testing mood weight calculations:");

    for mood in [0.1, 0.35, 0.6, 0.85] {
        let weights = MoodWeights::from_mood_value(mood);
        let normalized = weights.normalized();

        println!("\nMood {:.2}:", mood);
        println!("  Raw weights: env={:.3}, gentle={:.3}, active={:.3}, edm={:.3}",
                 weights.environmental, weights.gentle_melodic, weights.active_ambient, weights.edm_style);
        println!("  Sum: {:.3}", weights.environmental + weights.gentle_melodic + weights.active_ambient + weights.edm_style);
        println!("  Normalized: env={:.3}, gentle={:.3}, active={:.3}, edm={:.3}",
                 normalized.environmental, normalized.gentle_melodic, normalized.active_ambient, normalized.edm_style);
        println!("  Norm sum: {:.3}", normalized.environmental + normalized.gentle_melodic + normalized.active_ambient + normalized.edm_style);
    }
}