//! Benchmark mínimo de matmul cuadrado en CPU (con hilos).
//!
//! ```bash
//! cargo run --release --example matmul_bench -- 1024 10
//! ```

use mini_gpt::tensor::Tensor;
use std::time::Instant;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let size: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);
    let iterations: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    let left = Tensor::new(
        [size, size],
        (0..size * size)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.01)
            .collect(),
    )
    .expect("tensor izquierdo");
    let right = Tensor::new(
        [size, size],
        (0..size * size)
            .map(|index| ((index % 89) as f32 - 44.0) * 0.02)
            .collect(),
    )
    .expect("tensor derecho");

    let _ = left.matmul(&right).expect("calentamiento");
    let start = Instant::now();
    for _ in 0..iterations {
        let result = left.matmul(&right).expect("matmul");
        std::hint::black_box(result);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let flops = 2.0 * (size as f64).powi(3) * iterations as f64;
    println!(
        "backend={} tamano={size} iteraciones={iterations} tiempo={:.3}s {:.1} GFLOP/s",
        "CPU",
        elapsed,
        flops / elapsed / 1e9
    );
}
