import torch
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    WhisperModel, WhisperProcessor
)
from typing import Dict, Any


class ModelHandler:
    """Handles model loading and caching"""

    def __init__(self):
        self.loaded_models = {}
        self.loaded_processors = {}

    def load_model(self, model_name: str):
        """Load model and processor, cache them"""
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

    def get_model_layer_info(self, model_name: str) -> Dict[str, Any]:
        """Get layer information for a model"""
        model, _ = self.load_model(model_name)

        info = {
            'model_name': model_name,
            'model_type': None,
            'layers': {}
        }

        if 'wav2vec' in model_name.lower():
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

    def unload_all(self):
        """Unload all models"""
        self.loaded_models.clear()
        self.loaded_processors.clear()
        torch.cuda.empty_cache()
