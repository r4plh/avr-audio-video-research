from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import pickle
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_handler import ModelHandler
from utils.embedding_extractor import EmbeddingExtractor
from utils.visualizer import Visualizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Initialize handlers
model_handler = ModelHandler()
embedding_extractor = EmbeddingExtractor(model_handler)
visualizer = Visualizer()

# Dataset path
DATASET_PATH = "/data/aman/speech_commands/speech_commands_v0.02/"

# Model lists with parameter counts
WAV2VEC_MODELS = [
    {"name": "facebook/wav2vec2-base", "params": "95M"},
    {"name": "facebook/wav2vec2-base-960h", "params": "95M"},
    # Removed facebook/wav2vec2-large as it doesn't have a processor
    {"name": "facebook/wav2vec2-large-960h", "params": "317M"},
    {"name": "facebook/wav2vec2-large-960h-lv60", "params": "317M"},
    {"name": "facebook/wav2vec2-large-960h-lv60-self", "params": "317M"},
]

WHISPER_MODELS = [
    {"name": "openai/whisper-tiny", "params": "39M"},
    {"name": "openai/whisper-base", "params": "74M"},
    {"name": "openai/whisper-small", "params": "244M"},
    {"name": "openai/whisper-medium", "params": "769M"},
    {"name": "openai/whisper-large", "params": "1.55B"},
    {"name": "openai/whisper-large-v2", "params": "1.55B"},
    {"name": "openai/whisper-large-v3", "params": "1.55B"},
    {"name": "openai/whisper-tiny.en", "params": "39M"},
    {"name": "openai/whisper-base.en", "params": "74M"},
    {"name": "openai/whisper-small.en", "params": "244M"},
    {"name": "openai/whisper-medium.en", "params": "769M"},
]

SPEECH_COMMANDS_WORDS = [
    # Digits (10)
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",

    # Core commands (10)
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",

    # Additional commands (4)
    "backward", "forward", "follow", "learn",

    # Auxiliary words (11)
    "bed", "bird", "cat", "dog", "happy",
    "house", "marvin", "sheila", "tree", "wow", "visual"
]


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html',
                         wav2vec_models=WAV2VEC_MODELS,
                         whisper_models=WHISPER_MODELS,
                         speech_words=SPEECH_COMMANDS_WORDS)


@app.route('/get_model_info', methods=['POST'])
def get_model_info():
    """Get layer information for selected models"""
    data = request.json
    selected_models = data.get('models', [])

    model_info = {}

    for model_name in selected_models:
        try:
            info = model_handler.get_model_layer_info(model_name)
            model_info[model_name] = info
        except Exception as e:
            model_info[model_name] = {'error': str(e)}

    return jsonify(model_info)


@app.route('/extract_embeddings', methods=['POST'])
def extract_embeddings():
    """Extract embeddings based on user configuration"""
    data = request.json

    # Parse configuration
    config = {
        'models': data.get('models', []),
        'words': data.get('words', []),
        'samples_per_word': int(data.get('samples_per_word', 50)),
        'layer_configs': data.get('layer_configs', {}),  # {model_name: {layer_type: [layer_nums]}}
        'pooling_method': data.get('pooling_method', 'mean'),
        'pooling_position': int(data.get('pooling_position', 10)),
        'dataset_path': DATASET_PATH
    }

    try:
        # Extract embeddings
        results = embedding_extractor.extract_all(config)

        # Cache the results
        cache_key = embedding_extractor.generate_cache_key(config)
        cache_path = Path('cache') / f"{cache_key}.pkl"
        cache_path.parent.mkdir(exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

        return jsonify({
            'status': 'success',
            'cache_key': cache_key,
            'num_samples': len(results['labels'])
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """Generate visualizations"""
    data = request.json

    cache_key = data.get('cache_key')
    viz_type = data.get('viz_type', 'pca')  # 'pca', 'tsne', or 'both'
    dimensions = data.get('dimensions', '3d')  # '2d', '3d', or 'both'

    try:
        # Load cached embeddings
        cache_path = Path('cache') / f"{cache_key}.pkl"
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)

        print(f"Visualizing with viz_type={viz_type}, dimensions={dimensions}")

        # Generate visualizations
        plots = visualizer.create_plots(results, viz_type, dimensions)

        print(f"Generated {len(plots)} plots: {list(plots.keys())}")
        for plot_name, plot_html in plots.items():
            print(f"  {plot_name}: {len(plot_html)} characters")

        return jsonify({
            'status': 'success',
            'plots': plots
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear embedding cache"""
    try:
        cache_dir = Path('cache')
        if cache_dir.exists():
            for cache_file in cache_dir.glob('*.pkl'):
                cache_file.unlink()

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Create cache directory
    Path('cache').mkdir(exist_ok=True)

    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
