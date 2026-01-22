import torch
import sys
from pathlib import Path
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WhisperModel, WhisperProcessor,
    WhisperForConditionalGeneration
)
from typing import Dict, Any

# Import DAC utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dac_utils import DACProcessor


class ModelHandler:
    """Handles model loading and caching"""

    def __init__(self):
        self.loaded_models = {}
        self.loaded_processors = {}
        self.loaded_generation_models = {}  # For WhisperForConditionalGeneration
        self.loaded_dac_processors = {}  # For DAC models

    def load_model(self, model_name: str):
        """Load model and processor, cache them"""
        # Handle DAC models separately
        if 'dac' in model_name.lower():
            return self.load_dac_model(model_name)

        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_processors[model_name]

        print(f"Loading model: {model_name}")

        try:
            if 'wav2vec' in model_name.lower():
                model = Wav2Vec2Model.from_pretrained(model_name)
                processor = Wav2Vec2Processor.from_pretrained(model_name)
            elif 'whisper' in model_name.lower():
                model = WhisperModel.from_pretrained(model_name)
                processor = WhisperProcessor.from_pretrained(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
        except TypeError as e:
            if "expected str, bytes or os.PathLike object, not NoneType" in str(e):
                raise ValueError(
                    f"Model '{model_name}' doesn't have a processor/tokenizer. "
                    f"This model may be a base pre-training model without fine-tuning. "
                    f"Consider using the fine-tuned version (e.g., facebook/wav2vec2-large-960h instead of facebook/wav2vec2-large)."
                )
            else:
                raise

        model.eval()

        # Cache
        self.loaded_models[model_name] = model
        self.loaded_processors[model_name] = processor

        return model, processor

    def load_dac_model(self, model_name: str):
        """Load DAC model"""
        if model_name in self.loaded_dac_processors:
            return self.loaded_dac_processors[model_name], None

        print(f"Loading DAC model: {model_name}")

        # Extract model type from name (e.g., "DAC-16khz" -> "16khz")
        if '16khz' in model_name.lower():
            model_type = '16khz'
        elif '24khz' in model_name.lower():
            model_type = '24khz'
        elif '44khz' in model_name.lower():
            model_type = '44khz'
        else:
            model_type = '16khz'  # Default

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dac_processor = DACProcessor(model_type=model_type, device=device)

        # Cache
        self.loaded_dac_processors[model_name] = dac_processor

        return dac_processor, None

    def load_generation_model(self, model_name: str):
        """Load Whisper generation model for decoder inference"""
        if model_name in self.loaded_generation_models:
            return self.loaded_generation_models[model_name], self.loaded_processors[model_name]

        print(f"Loading generation model: {model_name}")

        if 'whisper' not in model_name.lower():
            raise ValueError(f"Generation model only supported for Whisper, got: {model_name}")

        # Load processor if not already loaded
        if model_name not in self.loaded_processors:
            processor = WhisperProcessor.from_pretrained(model_name)
            self.loaded_processors[model_name] = processor
        else:
            processor = self.loaded_processors[model_name]

        # Load generation model
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.eval()

        # Cache
        self.loaded_generation_models[model_name] = model

        return model, processor

    def get_model_layer_info(self, model_name: str) -> Dict[str, Any]:
        """Get layer information for a model"""
        model, _ = self.load_model(model_name)

        info = {
            'model_name': model_name,
            'model_type': None,
            'layers': {}
        }

        if 'dac' in model_name.lower():
            info['model_type'] = 'dac'

            # Determine number of codebooks based on model variant
            if '16khz' in model_name.lower():
                n_codebooks = 12
            else:  # 24khz and 44khz use 9 codebooks
                n_codebooks = 9

            info['layers'] = {
                'extraction_strategy': {
                    'available': True,
                    'strategies': [
                        'indices_mean',
                        'indices_max',
                        'embeddings_avg',
                        'embeddings_concat',
                        'latent_z',
                        'projections_concat',
                        'temporal_slice'
                    ],
                    'description': 'DAC extraction strategies (select one)',
                    'dimensions': {
                        'indices_mean': f'{n_codebooks}D',
                        'indices_max': f'{n_codebooks}D',
                        'embeddings_avg': '8D',
                        'embeddings_concat': f'{n_codebooks * 8}D',
                        'latent_z': '1024D',
                        'projections_concat': f'{n_codebooks * 1024:,}D',
                        'temporal_slice': f'{n_codebooks * 1024:,}D'
                    }
                }
            }

        elif 'wav2vec' in model_name.lower():
            info['model_type'] = 'wav2vec2'
            info['layers'] = {
                'cnn': {
                    'available': True,
                    'description': 'CNN feature extractor output (input to transformer)'
                },
                'encoder': {
                    'available': True,
                    'num_layers': len(model.encoder.layers),
                    'layer_indices': list(range(len(model.encoder.layers))),
                    'hidden_size': model.config.hidden_size,
                    'description': f'Transformer encoder layers (0-{len(model.encoder.layers)-1})'
                }
            }

        elif 'whisper' in model_name.lower():
            info['model_type'] = 'whisper'
            info['layers'] = {
                'cnn': {
                    'available': True,
                    'description': 'Conv layers output (input to transformer)'
                },
                'encoder': {
                    'available': True,
                    'num_layers': len(model.encoder.layers),
                    'layer_indices': list(range(len(model.encoder.layers))),
                    'hidden_size': model.config.d_model,
                    'description': f'Transformer encoder layers (0-{len(model.encoder.layers)-1})'
                },
                'decoder': {
                    'available': True,
                    'num_layers': len(model.decoder.layers),
                    'layer_indices': list(range(len(model.decoder.layers))),
                    'hidden_size': model.config.d_model,
                    'description': f'Transformer decoder layers (0-{len(model.decoder.layers)-1})'
                }
            }

        return info

    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.loaded_processors[model_name]
            torch.cuda.empty_cache()
        if model_name in self.loaded_dac_processors:
            del self.loaded_dac_processors[model_name]
            torch.cuda.empty_cache()

    def unload_all(self):
        """Unload all models"""
        self.loaded_models.clear()
        self.loaded_processors.clear()
        self.loaded_dac_processors.clear()
        torch.cuda.empty_cache()
