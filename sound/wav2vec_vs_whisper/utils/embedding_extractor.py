import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
import sys
from typing import Dict, List, Any
import hashlib
import json


class EmbeddingExtractor:
    """Extract embeddings from various layers of audio models"""

    def __init__(self, model_handler):
        self.model_handler = model_handler

    def generate_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate a unique cache key based on configuration"""
        # Create a deterministic string from config
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def pool_embeddings(self, hidden_states: torch.Tensor, method: str, position: int = 10) -> np.ndarray:
        """
        Pool hidden states to single vector
        Args:
            hidden_states: (batch, seq_len, hidden_dim) tensor
            method: 'mean', 'max', or 'position'
            position: position index if method='position'
        """
        if method == 'mean':
            return hidden_states.mean(dim=1).squeeze().cpu().numpy()
        elif method == 'max':
            return hidden_states.max(dim=1)[0].squeeze().cpu().numpy()
        elif method == 'position':
            if hidden_states.shape[1] > position:
                return hidden_states[:, position, :].squeeze().cpu().numpy()
            else:
                # Fallback to mean if position is out of bounds
                return hidden_states.mean(dim=1).squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def extract_wav2vec_embeddings(self, audio_path: str, model_name: str,
                                   layer_config: Dict[str, List[int]],
                                   pooling_method: str, pooling_position: int) -> Dict[str, np.ndarray]:
        """Extract embeddings from Wav2Vec2 model"""
        model, processor = self.model_handler.load_model(model_name)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        embeddings = {}

        with torch.no_grad():
            # CNN features (pre-transformer)
            if 'cnn' in layer_config and layer_config['cnn']:
                extract_features = model.feature_extractor(inputs.input_values)
                features = extract_features.transpose(1, 2)  # (batch, time, features)
                embeddings['cnn'] = self.pool_embeddings(features, pooling_method, pooling_position)

            # Encoder layers
            if 'encoder' in layer_config and layer_config['encoder']:
                outputs = model(inputs.input_values, output_hidden_states=True)

                # output_hidden_states includes: embedding layer + all transformer layers
                # Index 0 is after embedding, indices 1-N are transformer layer outputs
                for layer_idx in layer_config['encoder']:
                    # layer_idx corresponds to transformer layer index (0-based)
                    # hidden_states[0] = embedding output
                    # hidden_states[1] = transformer layer 0 output
                    # hidden_states[2] = transformer layer 1 output, etc.
                    hidden_state_idx = layer_idx + 1  # +1 because index 0 is embedding
                    if hidden_state_idx < len(outputs.hidden_states):
                        layer_output = outputs.hidden_states[hidden_state_idx]
                        embeddings[f'encoder_layer_{layer_idx}'] = self.pool_embeddings(
                            layer_output, pooling_method, pooling_position
                        )

        return embeddings

    def extract_whisper_embeddings(self, audio_path: str, model_name: str,
                                   layer_config: Dict[str, List[int]],
                                   pooling_method: str, pooling_position: int,
                                   text_label: str = None) -> Dict[str, np.ndarray]:
        """Extract embeddings from Whisper model"""
        model, processor = self.model_handler.load_model(model_name)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        embeddings = {}

        with torch.no_grad():
            # CNN features (pre-transformer)
            if 'cnn' in layer_config and layer_config['cnn']:
                mel = audio_inputs.input_features
                conv1 = model.encoder.conv1(mel)
                conv2 = model.encoder.conv2(conv1)
                features = conv2.permute(0, 2, 1)  # (batch, time, features)
                embeddings['cnn'] = self.pool_embeddings(features, pooling_method, pooling_position)

            # Encoder layers
            if 'encoder' in layer_config and layer_config['encoder']:
                encoder_outputs = model.encoder(
                    audio_inputs.input_features,
                    output_hidden_states=True
                )

                # hidden_states[0] = embedding output
                # hidden_states[1] = layer 0 output, etc.
                for layer_idx in layer_config['encoder']:
                    hidden_state_idx = layer_idx + 1
                    if hidden_state_idx < len(encoder_outputs.hidden_states):
                        layer_output = encoder_outputs.hidden_states[hidden_state_idx]
                        embeddings[f'encoder_layer_{layer_idx}'] = self.pool_embeddings(
                            layer_output, pooling_method, pooling_position
                        )

            # Decoder layers (requires text input)
            if 'decoder' in layer_config and layer_config['decoder'] and text_label:
                # Tokenize the text label
                text_inputs = processor.tokenizer(text_label, return_tensors="pt")
                decoder_input_ids = text_inputs.input_ids

                # Run decoder with encoder outputs
                decoder_outputs = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    output_hidden_states=True
                )

                # hidden_states[0] = embedding output
                # hidden_states[1] = layer 0 output, etc.
                for layer_idx in layer_config['decoder']:
                    hidden_state_idx = layer_idx + 1
                    if hidden_state_idx < len(decoder_outputs.hidden_states):
                        layer_output = decoder_outputs.hidden_states[hidden_state_idx]
                        embeddings[f'decoder_layer_{layer_idx}'] = self.pool_embeddings(
                            layer_output, pooling_method, pooling_position
                        )

        return embeddings

    def extract_all(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all embeddings based on configuration

        Returns:
            {
                'embeddings': {model_name: {embedding_type: [vectors]}},
                'labels': [word labels],
                'config': config
            }
        """
        dataset_path = config['dataset_path']
        words = config['words']
        samples_per_word = config['samples_per_word']
        models = config['models']
        layer_configs = config['layer_configs']
        pooling_method = config['pooling_method']
        pooling_position = config['pooling_position']

        # Collect file paths
        file_paths = []
        file_labels = []

        for word in words:
            word_dir = os.path.join(dataset_path, word)
            if not os.path.exists(word_dir):
                print(f"Warning: Directory not found for word '{word}': {word_dir}")
                continue

            files = [f for f in os.listdir(word_dir) if f.endswith('.wav')][:samples_per_word]

            for file in files:
                path = os.path.join(word_dir, file)
                file_paths.append(path)
                file_labels.append(word)

        print(f"Total files to process: {len(file_paths)}")

        # Initialize storage for embeddings
        all_embeddings = {model_name: {} for model_name in models}

        # Extract embeddings for each model
        for model_name in models:
            print(f"\nProcessing model: {model_name}")

            model_layer_config = layer_configs.get(model_name, {})

            # Determine model type
            if 'wav2vec' in model_name.lower():
                model_type = 'wav2vec2'
            elif 'whisper' in model_name.lower():
                model_type = 'whisper'
            else:
                print(f"Unknown model type: {model_name}")
                continue

            # Process each file
            # Disable tqdm if not in terminal (e.g., web app context)
            disable_progress = not sys.stdout.isatty()

            for file_path, label in tqdm(zip(file_paths, file_labels),
                                        total=len(file_paths),
                                        desc=f"Extracting {model_name}",
                                        disable=disable_progress):
                try:
                    if model_type == 'wav2vec2':
                        embeddings = self.extract_wav2vec_embeddings(
                            file_path, model_name, model_layer_config,
                            pooling_method, pooling_position
                        )
                    else:  # whisper
                        embeddings = self.extract_whisper_embeddings(
                            file_path, model_name, model_layer_config,
                            pooling_method, pooling_position,
                            text_label=label  # Use word label as text input
                        )

                    # Store embeddings
                    for emb_type, emb_vector in embeddings.items():
                        if emb_type not in all_embeddings[model_name]:
                            all_embeddings[model_name][emb_type] = []
                        all_embeddings[model_name][emb_type].append(emb_vector)

                except BrokenPipeError:
                    print(f"Warning: Broken pipe error for {file_path}, skipping...")
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        # Convert lists to numpy arrays
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        for model_name in all_embeddings:
            print(f"\nModel: {model_name}")
            for emb_type in all_embeddings[model_name]:
                all_embeddings[model_name][emb_type] = np.array(
                    all_embeddings[model_name][emb_type]
                )
                shape = all_embeddings[model_name][emb_type].shape
                print(f"  {emb_type}: {shape}")

        print(f"\nTotal samples: {len(file_labels)}")
        print(f"Unique labels: {set(file_labels)}")
        print("="*60 + "\n")

        return {
            'embeddings': all_embeddings,
            'labels': file_labels,
            'config': config
        }
