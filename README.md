# deep_filter

DeepFilterNet3 noise suppression for 48 kHz mono audio, running on the CPU with
[tract](https://github.com/sonos/tract).

This is the inference crate (`libDF`) of [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
by Hendrik Schröter, without the training, Python, LADSPA and C API code.

## Features

- `default-model`: embeds DeepFilterNet3 as `DfParams::default_model()` (30 ms delay).

## License

Code and model weights are dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
