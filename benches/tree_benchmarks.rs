use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kyt::tree::loss_fn::{Logit, ScoringFunction};
use kyt::tree::{Tree, TreeConfig};
use std::collections::HashMap;
use std::time::Instant;
use std::usize;

fn create_sample_data(size: usize) -> (HashMap<String, Vec<f64>>, Vec<bool>) {
    let mut data = HashMap::new();
    let mut target = Vec::with_capacity(size);

    let feature_name = "F1";
    let values: Vec<f64> = (0..size).map(|x| x as f64 * 0.1).collect();
    data.insert(feature_name.to_string(), values);

    // Create target vector
    for i in 0..size {
        target.push(i < size / 2);
    }

    (data, target)
}

fn bench_tree_fit_size_10000(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tree::");
    group.warm_up_time(std::time::Duration::from_secs(30));

    let tree_config = TreeConfig { max_depth: 3 };
    let score_fn = ScoringFunction::Logit(Logit::new(0.5));
    group.bench_function("size_10000", |b| {
        b.iter_custom(|iters| {
            let mut elapsed_time = std::time::Duration::new(0, 0);

            for _ in 0..iters {
                // Create data and target before timing
                let (data, target) = create_sample_data(10000);
                // Time only the part you want to benchmark
                let start = Instant::now();
                let _ = Tree::fit(
                    black_box(&data),
                    black_box(&target),
                    black_box(&tree_config),
                    black_box(&score_fn),
                );
                elapsed_time += start.elapsed();
            }
            elapsed_time
        });
    });

    group.finish();
}

criterion_group!(benches, bench_tree_fit_size_10000);
criterion_main!(benches);
