use anyhow::Result;
use deep_filter::tract::{DfParams, DfTract, RuntimeParams};
use hound;
use ndarray::prelude::*;
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.wav> <output.wav> [model.tar.gz]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let model_path = args.get(3).map(|s| PathBuf::from(s));

    // Load specified model
    let dfp = if let Some(path) = model_path {
        DfParams::new(path)?
    } else {
        // Try to find default model
        let models_dir = PathBuf::from("models");
        let model_tar = models_dir.join("DeepFilterNet3.tar.gz");
        if model_tar.exists() {
            DfParams::new(model_tar)?
        } else {
            anyhow::bail!("No model provided and default model not found");
        }
    };

    let rp = RuntimeParams::default_with_ch(1);
    let mut df = DfTract::new(dfp, &rp)?;

    // Load input WAV
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();

    // Convert to ndarray
    let hop_size = df.hop_size;
    let num_frames = (samples.len() + hop_size - 1) / hop_size;

    let mut noisy = Array2::zeros((1, hop_size));
    let mut enhanced = Array2::zeros((1, hop_size));
    let mut output_samples = Vec::with_capacity(samples.len());

    let start = Instant::now();
    for i in 0..num_frames {
        let start_idx = i * hop_size;
        let end_idx = (start_idx + hop_size).min(samples.len());
        let frame_len = end_idx - start_idx;

        noisy.fill(0.0);
        for j in 0..frame_len {
            noisy[[0, j]] = samples[start_idx + j];
        }

        df.process(noisy.view(), enhanced.view_mut())?;

        for j in 0..frame_len {
            output_samples.push(enhanced[[0, j]]);
        }
    }
    let duration = start.elapsed();

    println!("Processed {} frames in {:.2?}", num_frames, duration);
    println!(
        "RTF: {:.3}",
        duration.as_secs_f32() / (samples.len() as f32 / spec.sample_rate as f32)
    );

    // Write output WAV
    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for s in output_samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    println!("Wrote output to: {}", output_path);
    Ok(())
}
