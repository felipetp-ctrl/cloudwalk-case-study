use std::fs;
use std::time::Instant;
use tract_onnx::prelude::*;

#[derive(serde::Deserialize)]
struct SampleInput {
    feature_names: Vec<String>,
    values: Vec<f32>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64)
    }
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    // Skip args that look like a program path (contain '/' or end in '.wasm')
    let positional: Vec<&str> = args.iter()
        .skip(1)
        .filter(|a| !a.ends_with(".wasm"))
        .map(|s| s.as_str())
        .collect();
    let model_path = positional.first().copied()
        .unwrap_or("model/model_full.onnx").to_string();
    let input_path = positional.get(1).copied()
        .unwrap_or("model/sample_input.json").to_string();
    let n_iterations: usize = positional.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let input_bytes = fs::read(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read '{}': {}", input_path, e));
    let input_json = String::from_utf8(input_bytes)
        .unwrap_or_else(|_| panic!("'{}' is not valid UTF-8 (got model binary?)", input_path));
    let sample: SampleInput = serde_json::from_str(&input_json)
        .expect("Failed to parse sample input JSON");

    let model_file_size = fs::metadata(&model_path)
        .expect("Failed to read model file metadata")
        .len();

    let n_features = sample.values.len();
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .with_input_fact(0, f32::fact([1, n_features]).into())?
        .into_optimized()?
        .into_runnable()?;

    let input_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, n_features),
        sample.values.clone(),
    )
    .expect("Failed to create input tensor")
    .into();

    // Warmup
    for _ in 0..100 {
        let _ = model.run(tvec!(input_tensor.clone().into()))?;
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(n_iterations);
    for _ in 0..n_iterations {
        let start = Instant::now();
        let _ = model.run(tvec!(input_tensor.clone().into()))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(elapsed);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let throughput = 1000.0 / mean;

    println!("Model: {}", model_path);
    println!("Features: {} ({})", n_features, sample.feature_names.join(", "));
    println!("Model file size: {:.1} KB", model_file_size as f64 / 1024.0);
    println!("Iterations: {}", n_iterations);
    println!();
    println!("Latency:");
    println!("  p50:  {:.4} ms", percentile(&latencies, 50.0));
    println!("  p95:  {:.4} ms", percentile(&latencies, 95.0));
    println!("  p99:  {:.4} ms", percentile(&latencies, 99.0));
    println!("  mean: {:.4} ms", mean);
    println!("  min:  {:.4} ms", latencies.first().unwrap());
    println!("  max:  {:.4} ms", latencies.last().unwrap());
    println!();
    println!("Throughput: {:.0} inferences/sec (single-thread)", throughput);
    println!();

    let threads_50k = 50_000.0 * (percentile(&latencies, 99.0) / 1000.0);
    println!("Threads needed for 50k req/s (at p99): {:.1}", threads_50k);
    if percentile(&latencies, 99.0) < 5.0 {
        println!("PASS: p99 latency under 5ms budget");
    } else {
        println!("FAIL: p99 latency exceeds 5ms budget");
    }
    if threads_50k <= 8.0 {
        println!("PASS: fits in <=8 cores for 50k req/s");
    } else {
        println!("FAIL: needs more than 8 cores for 50k req/s");
    }

    Ok(())
}
