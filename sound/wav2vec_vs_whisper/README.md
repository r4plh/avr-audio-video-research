# Audio Model Embedding Visualizer

This is a web app that lets visualize and compare how different audio AI models (Wav2Vec2, Whisper, and DAC) understand speech. 

All notebook code which was used to verify and integrate code is in /src.

## Working flow

You pick some models, some words (like "yes", "no", "cat", "dog"), and the app:
1. Runs those words through the models
2. Grabs the internal representations (embeddings) from different layers
3. Reduces them to 2D or 3D so you can actually see them
4. Shows you interactive plots where you can see how well the models separate different words

It's useful for understanding which layers work best, comparing model architectures, or just exploring how these models "think" about speech.

## Quick Start

**1. Install stuff:**
```bash
pip install -r requirements.txt or uv setup
```

**2. Make sure you have the dataset:**
The Speech Commands dataset should be at `/data/aman/speech_commands/speech_commands_v0.02/`

If it's somewhere else, update `DATASET_PATH` in `app.py` (line 30).

**3. Run it:**
```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## How to use it

The UI walks you through 4 steps:

### Step 1: Pick your models
- Check the boxes for which models you want to compare
- App have Wav2Vec2 (Facebook's model), Whisper (OpenAI's), and DAC (audio codec)
- Click "Load Model Info" when ready

### Step 2: Pick which layers
- After loading, options for each model will open up
- **For Wav2Vec2/Whisper**: Check boxes for encoder layers (0, 1, 2...) or CNN features
- **For DAC**: Pick an extraction strategy from the dropdown


### Step 3: Pick your words
- Select which words you want to analyze (e.g., "zero", "one", "yes", "no")
- Pro tip: Start with 2-3 words and 10-20 samples to test things out
- You can always run it again with more later

### Step 4: Visualization settings
- **PCA vs t-SNE**: Both are ways to squish high-dimensional data down to 2D/3D
- **2D vs 3D**: 3D plots are cooler to rotate, 2D are easier to read
- **Pooling method**: How to summarize the sequence (usually just leave it on "Mean")

### Step 5: Go
Click "Extract Embeddings", wait a bit (could be a few minutes), then "Generate Plots"

## What models are available?

**Wav2Vec2** (Facebook's speech recognition model):
- wav2vec2-base (95M params) - good starting point
- wav2vec2-large-960h (317M params) - bigger and better
- Several variants available in the dropdown

**Whisper** (OpenAI's multilingual speech model):
- tiny, base, small, medium, large (39M to 1.55B params)
- We also have the English-only versions (.en)

**DAC** (Descript Audio Codec):
- 16kHz, 24kHz, 44kHz versions
- This one's different - it's a compression model, not speech recognition

## File structure

```
app.py                      # Main Flask app - start here
utils/
  ├── model_handler.py      # Loads and caches models
  ├── embedding_extractor.py # Extracts embeddings from audio
  └── visualizer.py         # Creates the plots
templates/index.html        # The web UI
static/
  ├── js/main.js           # Frontend logic
  └── css/style.css        # Makes it pretty
cache/                      # Embeddings get saved here
```

## Tips & Tricks

**Start small**: Your first run? Try 1 model, 2 words, 10 samples. Get a feel for it before going big.

**Layer mode matters**:
- "Individual" = separate plot for each layer (good for comparing layers)
- "Concatenate" = combine all layers into one embedding (good for using all info together)

**Memory issues?** These models are huge. If you run out of RAM:
- Pick smaller models (tiny/base instead of large)
- Fewer samples per word
- Fewer layers selected

**Plots look weird?** Try the other reduction method. Sometimes PCA works better, sometimes t-SNE.


## What's actually happening under the hood?

1. `model_handler.py` loads the AI models and keeps them in memory
2. `embedding_extractor.py` feeds your audio through the models and grabs the internal activations
3. Those embeddings get cached so you don't have to re-extract them
4. `visualizer.py` uses PCA or t-SNE to reduce from 768D (or whatever) down to 2D/3D
5. Plotly creates interactive plots you can rotate and zoom


