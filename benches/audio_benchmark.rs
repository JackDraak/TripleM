use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mood_music_module::{MoodMusicModule, MoodConfig};

fn benchmark_sample_generation(c: &mut Criterion) {
    let config = MoodConfig::default();
    let mut module = MoodMusicModule::with_config(config).unwrap();
    module.start();
    module.set_mood(0.5);

    c.bench_function("generate_single_sample", |b| {
        b.iter(|| {
            black_box(module.get_next_sample())
        })
    });
}

fn benchmark_buffer_fill(c: &mut Criterion) {
    let config = MoodConfig::default();
    let mut module = MoodMusicModule::with_config(config).unwrap();
    module.start();
    module.set_mood(0.5);

    let mut buffer = vec![0.0; 512];

    c.bench_function("fill_buffer_512", |b| {
        b.iter(|| {
            module.fill_buffer(black_box(&mut buffer))
        })
    });
}

criterion_group!(benches, benchmark_sample_generation, benchmark_buffer_fill);
criterion_main!(benches);