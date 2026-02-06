# VideoEra

A structured knowledge base for audio-visual ML research, containing documentation, tutorials, and projects related to audio processing and AV synchronization.

## Repository Structure

```
videoEra/
├── docs/                       # Documentation
│   ├── audio_fundamentals/     # Core audio concepts
│   │   ├── sound_physics_and_perception.md
│   │   ├── fourier_intuition.md
│   │   ├── aliasing.md
│   │   └── sound_reverb.md
│   ├── audio_codecs/           # Codec documentation
│   │   ├── README.md           # Audio codecs overview
│   │   └── descript_audio_codec.md
│   └── papers/                 # Research papers
│       ├── hubert.pdf
│       ├── wav2vec2.pdf
│       ├── whisper.pdf
│       └── syncnet.pdf
│
├── assets/                     # Image assets
│   ├── audio_fundamentals/     # Sound concept diagrams
│   ├── audio_codecs/           # Codec diagrams
│   └── syncnet/                # SyncNet architecture diagrams
│
├── notebooks/                  # Jupyter notebooks
│   ├── spectral_analysis/      # Frequency domain analysis
│   │   ├── audio/              # Sample audio files
│   │   ├── magnitude_spectrum.ipynb
│   │   └── spectrogram.ipynb
│   └── time_domain/            # Time domain analysis
│       ├── audio/              # Sample audio files
│       ├── amplitude_envelope.ipynb
│       └── rms_energy_zcr.ipynb
│
├── projects/                   # Standalone implementations
│   ├── audio_model_visualizer/ # Flask app for wav2vec/whisper comparison
│   └── syncnet/                # SyncNet study notes and presentations
│
└── tutorials/                  # Framework learning
    ├── pytorch_lightning/      # PyTorch Lightning tutorials
    │   ├── README.md
    │   ├── train_a_model.py
    │   ├── train_val_test.py
    │   ├── use_pretrained_model.py
    │   └── control_from_cli/   # CLI argument parsing
    └── hydra/                  # Hydra configuration tutorials
        ├── my_app.py
        └── conf/
```

## Quick Navigation

### Documentation
- [Sound Physics & Perception](docs/audio_fundamentals/sound_physics_and_perception.md)
- [Fourier Transform Intuition](docs/audio_fundamentals/fourier_intuition.md)
- [Aliasing](docs/audio_fundamentals/aliasing.md)
- [Sound Reverb & Room Impulse Response](docs/audio_fundamentals/sound_reverb.md)
- [Audio Codecs Overview](docs/audio_codecs/README.md)
- [Descript Audio Codec (DAC)](docs/audio_codecs/descript_audio_codec.md)

### Notebooks
- [Magnitude Spectrum Analysis](notebooks/spectral_analysis/magnitude_spectrum.ipynb)
- [Spectrogram Analysis](notebooks/spectral_analysis/spectrogram.ipynb)
- [Amplitude Envelope](notebooks/time_domain/amplitude_envelope.ipynb)
- [RMS Energy & Zero Crossing Rate](notebooks/time_domain/rms_energy_zcr.ipynb)

### Projects
- [Audio Model Visualizer](projects/audio_model_visualizer/) - Flask app comparing wav2vec and whisper embeddings
- [SyncNet Notes](projects/syncnet/) - Study notes on audio-visual synchronization

### Tutorials
- [PyTorch Lightning](tutorials/pytorch_lightning/README.md)
- [Hydra Configuration](tutorials/hydra/)

## Setup

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd videoEra

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Running the Audio Model Visualizer

```bash
cd projects/audio_model_visualizer
uv sync
python app.py
```

## Dependencies

Core dependencies include:
- `pytorch` / `torchvision` - Deep learning framework
- `lightning` - PyTorch Lightning for training
- `transformers` - Hugging Face models (wav2vec2, whisper)
- `hydra-core` - Configuration management
- `librosa` - Audio processing
- `jupyter` / `notebook` - Interactive notebooks

## License

This is a personal learning repository.
