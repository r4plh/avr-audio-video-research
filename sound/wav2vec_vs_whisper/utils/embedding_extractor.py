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
        # Load base model for encoder/CNN
        model, processor = self.model_handler.load_model(model_name)

        # Load generation model if decoder layers are requested
        gen_model = None
        if 'decoder' in layer_config and layer_config['decoder']:
            gen_model, _ = self.model_handler.load_generation_model(model_name)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        embeddings = {}

        with torch.no_grad():
            # We need encoder outputs for decoder, so compute them if decoder layers are requested
            encoder_outputs = None
            if ('encoder' in layer_config and layer_config['encoder']) or \
               ('decoder' in layer_config and layer_config['decoder']):
                encoder_outputs = model.encoder(
                    audio_inputs.input_features,
                    output_hidden_states=True
                )

            # CNN features (pre-transformer)
            if 'cnn' in layer_config and layer_config['cnn']:
                mel = audio_inputs.input_features
                conv1 = model.encoder.conv1(mel)
                conv2 = model.encoder.conv2(conv1)
                features = conv2.permute(0, 2, 1)  # (batch, time, features)
                embeddings['cnn'] = self.pool_embeddings(features, pooling_method, pooling_position)

            # Encoder layers
            if 'encoder' in layer_config and layer_config['encoder'] and encoder_outputs:
                # hidden_states[0] = embedding output
                # hidden_states[1] = layer 0 output, etc.
                for layer_idx in layer_config['encoder']:
                    hidden_state_idx = layer_idx + 1
                    if hidden_state_idx < len(encoder_outputs.hidden_states):
                        layer_output = encoder_outputs.hidden_states[hidden_state_idx]
                        embeddings[f'encoder_layer_{layer_idx}'] = self.pool_embeddings(
                            layer_output, pooling_method, pooling_position
                        )

            # Decoder layers (use actual generation to transcribe audio)
            if 'decoder' in layer_config and layer_config['decoder'] and encoder_outputs and gen_model:
                try:
                    # Generate transcription from audio (no teacher forcing)
                    # This uses greedy decoding to predict the actual word

                    # Prepare generation arguments
                    generated = gen_model.generate(
                        audio_inputs.input_features,
                        max_new_tokens=10,  # Speech Commands are short words
                        num_beams=1,  # Greedy decoding
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        language="en",
                        task="transcribe"
                    )

                    print(f"DEBUG: Generated object type: {type(generated)}")
                    print(f"DEBUG: Generated attributes: {dir(generated)}")
                    if hasattr(generated, 'decoder_hidden_states'):
                        print(f"DEBUG: decoder_hidden_states type: {type(generated.decoder_hidden_states)}")
                        if generated.decoder_hidden_states:
                            print(f"DEBUG: Number of generation steps: {len(generated.decoder_hidden_states)}")
                            if len(generated.decoder_hidden_states) > 0:
                                print(f"DEBUG: Number of layers in step 0: {len(generated.decoder_hidden_states[0])}")
                    else:
                        print("DEBUG: No decoder_hidden_states attribute!")

                    # generated.decoder_hidden_states is a tuple of tuples
                    # Structure: (step_0_layers, step_1_layers, ...)
                    # Each step_i_layers is a tuple of hidden states for each layer at generation step i

                    # We want the FINAL generation step's hidden states (when word is fully formed)
                    if hasattr(generated, 'decoder_hidden_states') and generated.decoder_hidden_states and len(generated.decoder_hidden_states) > 0:
                        # Get the last generation step
                        last_step_hidden_states = generated.decoder_hidden_states[-1]

                        # last_step_hidden_states is a tuple: (layer_0, layer_1, ..., layer_N)
                        # Each layer_i has shape [batch, 1, hidden_dim] (1 token generated at this step)

                        for layer_idx in layer_config['decoder']:
                            # hidden_states index: 0 = embedding, 1 = layer 0, 2 = layer 1, etc.
                            hidden_state_idx = layer_idx + 1
                            if hidden_state_idx < len(last_step_hidden_states):
                                layer_output = last_step_hidden_states[hidden_state_idx]
                                print(f"DEBUG: Layer {layer_idx} output shape: {layer_output.shape}")
                                # layer_output shape: [batch, 1, hidden_dim]
                                embeddings[f'decoder_layer_{layer_idx}'] = self.pool_embeddings(
                                    layer_output, pooling_method, pooling_position
                                )
                    else:
                        print(f"WARNING: Could not extract decoder hidden states for {audio_path}")

                except Exception as e:
                    print(f"ERROR during decoder generation for {audio_path}: {e}")
                    import traceback
                    traceback.print_exc()

        return embeddings

    def extract_dac_embeddings(self, audio_path: str, model_name: str,
                               dac_strategy: str = 'indices_mean',
                               time_index: int = 25) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from DAC model using various strategies

        Args:
            audio_path: Path to audio file
            model_name: DAC model name
            dac_strategy: One of the 7 extraction strategies
            time_index: Time index for temporal_slice strategy

        Returns:
            Dictionary with embedding_type: vector
        """
        dac_processor, _ = self.model_handler.load_model(model_name)

        embeddings = {}

        # Encode audio once
        encoded = dac_processor.encode_audio(audio_path)
        codes = encoded['codes'][0]  # [n_codebooks, time]

        if dac_strategy == 'indices_mean':
            # Strategy 1: Discrete indices with mean pooling
            vector = codes.float().mean(dim=1).detach().cpu().numpy()
            embeddings['indices_mean'] = vector

        elif dac_strategy == 'indices_max':
            # Strategy 2: Discrete indices with max pooling
            vector = codes.float().max(dim=1)[0].detach().cpu().numpy()
            embeddings['indices_max'] = vector

        elif dac_strategy == 'embeddings_avg':
            # Strategy 3: 8D codebook embeddings, averaged across codebooks + time
            codebook_embs = dac_processor.get_codebook_embeddings(codes.unsqueeze(0))
            vector = codebook_embs.mean(dim=[1, 2]).squeeze(0).detach().cpu().numpy()
            embeddings['embeddings_avg'] = vector

        elif dac_strategy == 'embeddings_concat':
            # Strategy 4: 8D codebook embeddings, time-averaged, concat codebooks (96D)
            codebook_embs = dac_processor.get_codebook_embeddings(codes.unsqueeze(0))  # [1, N, T, 8]
            time_pooled = codebook_embs.mean(dim=2)  # [1, N, 8]
            vector = time_pooled.reshape(1, -1).squeeze(0).detach().cpu().numpy()  # [96]
            embeddings['embeddings_concat'] = vector

        elif dac_strategy == 'latent_z':
            # Strategy 5: Latent representation z (1024D)
            z = encoded['z'][0]  # [1024, time]
            vector = z.mean(dim=1).detach().cpu().numpy()  # [1024]
            embeddings['latent_z'] = vector

        elif dac_strategy == 'projections_concat':
            # Strategy 6: Concatenated projections (12,288D)
            codes_batch = codes.unsqueeze(0).to(dac_processor.device)  # [1, N, T]
            N = codes_batch.shape[1]

            codebook_projections = []
            for i in range(N):
                quantizer = dac_processor.model.quantizer.quantizers[i]
                indices = codes_batch[:, i:i+1, :]  # [1, 1, T]
                z_e = quantizer.embed_code(indices.squeeze(1))  # [1, T, 8]
                z_q = quantizer.out_proj(z_e.transpose(1, 2))  # [1, 1024, T]
                z_q_pooled = z_q.mean(dim=2)  # [1, 1024]
                codebook_projections.append(z_q_pooled)

            concatenated = torch.cat(codebook_projections, dim=1)  # [1, 12288]
            vector = concatenated.squeeze(0).detach().cpu().numpy()
            embeddings['projections_concat'] = vector

        elif dac_strategy == 'temporal_slice':
            # Strategy 7: Temporal slice at specific time index (12,288D)
            codes_batch = codes.unsqueeze(0).to(dac_processor.device)  # [1, N, T]
            N = codes_batch.shape[1]
            T = codes_batch.shape[2]

            # Validate time index
            if time_index >= T:
                time_index = T // 2  # Use middle if out of bounds

            codebook_projections = []
            for i in range(N):
                quantizer = dac_processor.model.quantizer.quantizers[i]
                indices = codes_batch[:, i:i+1, :]  # [1, 1, T]
                z_e = quantizer.embed_code(indices.squeeze(1))  # [1, T, 8]
                z_q = quantizer.out_proj(z_e.transpose(1, 2))  # [1, 1024, T]
                z_q_slice = z_q[:, :, time_index]  # [1, 1024]
                codebook_projections.append(z_q_slice)

            concatenated = torch.cat(codebook_projections, dim=1)  # [1, 12288]
            vector = concatenated.squeeze(0).detach().cpu().numpy()
            embeddings['temporal_slice'] = vector

        else:
            raise ValueError(f"Unknown DAC strategy: {dac_strategy}")

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
        layer_mode = config.get('layer_mode', 'individual')  # Default to individual

        # Collect file paths
        file_paths = []
        file_labels = []

        for word in words:
            word_dir = os.path.join(dataset_path, word)
            if not os.path.exists(word_dir):
                print(f"Warning: Directory not found for word '{word}': {word_dir}")
                continue

            files = [f for f in os.listdir(word_dir) if f.endswith(('.wav', '.mp3'))][:samples_per_word]

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
            if 'dac' in model_name.lower():
                model_type = 'dac'
            elif 'wav2vec' in model_name.lower():
                model_type = 'wav2vec2'
            elif 'whisper' in model_name.lower():
                model_type = 'whisper'
            else:
                print(f"Unknown model type: {model_name}")
                continue

            # For DAC, get the strategy from layer_config
            dac_strategy = 'indices_mean'  # default
            dac_time_index = 25  # default
            if model_type == 'dac' and 'extraction_strategy' in model_layer_config:
                strategies = model_layer_config.get('extraction_strategy', [])
                if strategies and len(strategies) > 0:
                    dac_strategy = strategies[0]
                # Get time index if specified
                dac_time_index = model_layer_config.get('time_index', 25)

            # Process each file
            # Disable tqdm if not in terminal (e.g., web app context)
            disable_progress = not sys.stdout.isatty()

            for file_path, label in tqdm(zip(file_paths, file_labels),
                                        total=len(file_paths),
                                        desc=f"Extracting {model_name}",
                                        disable=disable_progress):
                try:
                    if model_type == 'dac':
                        embeddings = self.extract_dac_embeddings(
                            file_path, model_name, dac_strategy, dac_time_index
                        )
                    elif model_type == 'wav2vec2':
                        embeddings = self.extract_wav2vec_embeddings(
                            file_path, model_name, model_layer_config,
                            pooling_method, pooling_position
                        )
                    else:  # whisper
                        embeddings = self.extract_whisper_embeddings(
                            file_path, model_name, model_layer_config,
                            pooling_method, pooling_position
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

        # If concatenate mode, combine all layers per model
        if layer_mode == 'concatenate':
            concatenated_embeddings = {}

            for model_name in all_embeddings:
                print(f"\nModel: {model_name}")

                # Convert to arrays first
                for emb_type in all_embeddings[model_name]:
                    all_embeddings[model_name][emb_type] = np.array(
                        all_embeddings[model_name][emb_type]
                    )

                # Concatenate all layer embeddings
                if len(all_embeddings[model_name]) > 0:
                    # Get all embeddings for this model
                    layer_embeddings = []
                    layer_names = []

                    for emb_type in sorted(all_embeddings[model_name].keys()):
                        layer_embeddings.append(all_embeddings[model_name][emb_type])
                        layer_names.append(emb_type)

                    # Concatenate along feature dimension (axis=1)
                    concatenated = np.concatenate(layer_embeddings, axis=1)

                    # Store concatenated result
                    concatenated_embeddings[model_name] = {
                        'concatenated': concatenated,
                        'layers_included': layer_names,
                        'original_dims': [emb.shape[1] for emb in layer_embeddings],
                        'total_dim': concatenated.shape[1]
                    }

                    print(f"  Concatenated {len(layer_names)} layers:")
                    print(f"    Layers: {', '.join(layer_names)}")
                    print(f"    Dimensions: {' + '.join(map(str, [emb.shape[1] for emb in layer_embeddings]))} = {concatenated.shape[1]}")
                    print(f"    Final shape: {concatenated.shape}")

            all_embeddings = concatenated_embeddings

        else:  # Individual mode (existing behavior)
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
        print(f"Layer mode: {layer_mode}")
        print("="*60 + "\n")

        return {
            'embeddings': all_embeddings,
            'labels': file_labels,
            'config': config,
            'layer_mode': layer_mode
        }
