# AudioVideoEra

> This repo is actively maintained and keeps getting updated as I go deeper into audio-visual ML research.
When I started working on speech enhancement at Invideo, I realised pretty quickly that you can't build good audio-visual models without going deep into the fundamentals - what sound actually is, how it gets digitised, what happens when you take an STFT, why aliasing destroys information, how codecs compress and reconstruct audio. The gap between "I've used librosa" and "I know what's happening under the hood" is massive, and that gap shows up when you're debugging model outputs, designing experiments, and reasoning about why something failed.

This repo is where I document all of that - from first principles (sound physics, Fourier transforms, sampling theory) all the way up to paper breakdowns (HuBERT, Wav2Vec2, Whisper, SyncNet), spectral analysis notebooks, and standalone implementations. It basically traces the path from "what is a sound wave" to "how does a self-supervised model learn audio-visual synchronisation."

I learn by going deep, writing things down, and building something with it. That's what this repo is. It's not a finished thing - it keeps growing as I go deeper into audio-visual research, pick up new papers, and build new things at work. If you're seeing this, you're looking at a snapshot of where I am right now.

## What's Inside

**`docs/audio_fundamentals/`** - The foundation. Sound physics and perception, Fourier transform intuition (not just the formula, but *why* decomposing into sinusoids works), aliasing from first principles, and room impulse response / reverb. The reverb stuff directly feeds into the synthetic data generation pipeline I built at Invideo for training speech enhancement models.

**`docs/audio_codecs/`** - Documentation on audio codecs, with a focused breakdown of the Descript Audio Codec (DAC) - the codec architecture I worked with during research. Covers the encoder-decoder pipeline, residual vector quantisation, and how neural codecs differ from traditional ones.

**`docs/papers/`** - The core papers behind modern audio and audio-visual ML: HuBERT, Wav2Vec2, Whisper, and SyncNet. Not just stored PDFs - the SyncNet paper has a [full 21-page technical breakdown on Towards Data Science](https://towardsdatascience.com/syncnet-paper-easily-explained/) with custom diagrams, exact tensor shape walkthroughs, and training analysis with W&B logs.

**`notebooks/`** - Spectral and time-domain analysis. Magnitude spectrum, spectrograms, amplitude envelopes, RMS energy, zero crossing rate - all with real audio samples. These are the notebooks I used to build intuition before touching any model code.

**`projects/`**
- **Audio Model Visualiser** - Flask app for comparing Wav2Vec2 and Whisper embeddings side by side. Useful for seeing what different self-supervised audio models actually learn.
- **SyncNet** - Study notes, presentations, and implementation details from my deep dive into audio-visual synchronisation.

**`tutorials/`** - PyTorch Lightning and Hydra configuration tutorials. The training infra I use for experiments - Lightning for the training loop, Hydra for config management.

**`sound/DAC/`** - DAC experiments and notes.

**`top_ai_QnA/`** - Curated Q&A on AI/ML topics.

## Repository Structure

```
videoEra/
├── docs/
│   ├── audio_fundamentals/
│   │   ├── sound_physics_and_perception.md
│   │   ├── fourier_intuition.md
│   │   ├── aliasing.md
│   │   └── sound_reverb.md
│   ├── audio_codecs/
│   │   ├── README.md
│   │   └── descript_audio_codec.md
│   └── papers/
│       ├── hubert.pdf
│       ├── wav2vec2.pdf
│       ├── whisper.pdf
│       └── syncnet.pdf
│
├── notebooks/
│   ├── spectral_analysis/
│   │   ├── magnitude_spectrum.ipynb
│   │   └── spectrogram.ipynb
│   └── time_domain/
│       ├── amplitude_envelope.ipynb
│       └── rms_energy_zcr.ipynb
│
├── projects/
│   ├── audio_model_visualizer/
│   └── syncnet/
│
├── sound/DAC/
├── top_ai_QnA/opus_4.6/
│
└── tutorials/
    ├── pytorch_lightning/
    └── hydra/
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
- [Audio Model Visualiser](projects/audio_model_visualizer) - Flask app comparing Wav2Vec2 and Whisper embeddings
- [SyncNet Notes](projects/syncnet) - Study notes on audio-visual synchronisation

### Tutorials
- [PyTorch Lightning](tutorials/pytorch_lightning/README.md)
- [Hydra Configuration](tutorials/hydra)

## Setup

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

```bash
git clone https://github.com/r4plh/videoEra.git
cd videoEra

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Running the Audio Model Visualiser

```bash
cd projects/audio_model_visualizer
uv sync
python app.py
```

## Related Writing

- [SyncNet Paper Easily Explained](https://towardsdatascience.com/syncnet-paper-easily-explained/) - Full technical breakdown on Towards Data Science
- [The Grey Bar - Sound, Data & Childhood Curiosity](https://r4plh.github.io/blog/grey-bar.html) - On sound waves, compression, and the invisible machinery behind streaming
- [Understanding the Foundational Distortion of Digital Audio](https://r4plh.github.io/blog/aliasing.html) - Aliasing from first principles
- [76 Pages of Handwritten Audio Preprocessing Notes](https://drive.google.com/file/d/1kyGTvSwHMBcZZvMj2ZvdD7BpfGreprFp/view?usp=sharing) - Sound physics → sampling → Fourier transforms → STFT → filter banks → mel spectrograms
