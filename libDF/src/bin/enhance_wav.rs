//! Process WAV files with a DeepFilterNet model.
//!
//! Usage: `enhance-wav <model.tar.gz> <output_dir> <input.wav> ...`
//!
//! Inputs must be 16-bit 48 kHz WAV. Stereo files are processed per-channel
//! (the model is mono-only) and re-interleaved on write. Output preserves the
//! input's channel layout and sample rate.

use std::{path::PathBuf, process::exit, time::Instant};

use anyhow::Result;
use deep_filter::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{Array2, Axis};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <model.tar.gz> <output_dir> <input.wav> [input2.wav ...]",
            args[0]
        );
        exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    let files: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();

    let df_params = match DfParams::new(model_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error opening model: {e}");
            exit(1)
        }
    };

    let r_params = RuntimeParams::default();
    let hop = DfTract::new(df_params.clone(), &r_params)?.hop_size;

    if !output_dir.is_dir() {
        std::fs::create_dir_all(&output_dir)?;
    }

    for file in &files {
        let mut reader = hound::WavReader::open(file)?;
        let spec = reader.spec();
        let n_ch = spec.channels as usize;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;

        let n_frames = samples.len() / n_ch;
        let t0 = Instant::now();

        let mut channels: Vec<Vec<f32>> = Vec::with_capacity(n_ch);
        for ch in 0..n_ch {
            let mut model = DfTract::new(df_params.clone(), &r_params)?;
            let mut ch_noisy = Array2::<f32>::zeros((1, n_frames));
            for i in 0..n_frames {
                ch_noisy[[0, i]] = samples[i * n_ch + ch];
            }
            let mut ch_enh = Array2::<f32>::zeros((1, n_frames));
            for (ns_f, enh_f) in ch_noisy
                .view()
                .axis_chunks_iter(Axis(1), hop)
                .zip(ch_enh.view_mut().axis_chunks_iter_mut(Axis(1), hop))
            {
                if ns_f.len_of(Axis(1)) < hop {
                    break;
                }
                model.process(ns_f, enh_f)?;
            }
            channels.push(ch_enh.row(0).to_vec());
        }

        let elapsed = t0.elapsed().as_secs_f32();
        eprintln!(
            "Enhanced {} in {:.2}s (RTF: {:.3})",
            file.display(),
            elapsed,
            elapsed / (n_frames as f32 / 48000.0)
        );

        let mut out_path = output_dir.clone();
        out_path.push(file.file_name().unwrap());
        let mut writer = hound::WavWriter::create(&out_path, spec)?;
        for i in 0..n_frames {
            for ch in 0..n_ch {
                writer.write_sample(channels[ch][i] as i16)?;
            }
        }
        writer.finalize()?;
        eprintln!("Wrote: {}", out_path.display());
    }

    Ok(())
}
