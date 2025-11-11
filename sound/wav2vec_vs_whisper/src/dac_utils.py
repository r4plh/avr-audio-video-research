"""
Utilities for DAC (Descript Audio Codec) processing and visualization
"""

import sys
import os
from pathlib import Path

# Add DAC to path
dac_path = Path(__file__).parent.parent / "descript-audio-codec"
if str(dac_path) not in sys.path:
    sys.path.insert(0, str(dac_path))

import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import dac
from audiotools import AudioSignal


class DACProcessor:
    """
    Handles DAC model loading and embedding extraction
    """

    def __init__(self, model_type: str = "16khz", device: str = "cuda"):
        """
        Initialize DAC processor

        Args:
            model_type: One of "44khz", "24khz", or "16khz"
            device: "cuda" or "cpu"
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained DAC model"""
        print(f"Loading DAC model ({self.model_type})...")
        model_path = dac.utils.download(model_type=self.model_type)
        self.model = dac.DAC.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully!")
        print(f"  - Sample rate: {self.model.sample_rate}Hz")
        print(f"  - Codebooks: {self.model.n_codebooks}")
        print(f"  - Codebook size: {self.model.codebook_size}")
        print(f"  - Codebook dim: {self.model.codebook_dim}")

    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> AudioSignal:
        """
        Load audio file and prepare for DAC encoding

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (None = use model's native SR)

        Returns:
            AudioSignal object
        """
        if target_sr is None:
            target_sr = self.model.sample_rate

        # Method 1: Load directly from file path (simplest)
        signal = AudioSignal(audio_path)

        # Resample if needed
        if signal.sample_rate != target_sr:
            signal.resample(target_sr)

        return signal

    def encode_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Encode audio to DAC codes and embeddings

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with:
                - codes: [1, n_codebooks, time] discrete indices
                - latents: [1, n_codebooks*codebook_dim, time] continuous latents
                - z: [1, latent_dim, time] quantized representation
                - audio_length: Original audio length in samples
        """
        # Load audio
        signal = self.load_audio(audio_path)
        signal = signal.to(self.device)

        with torch.no_grad():
            # Preprocess
            x = self.model.preprocess(signal.audio_data, signal.sample_rate)

            # Encode: returns (z, codes, latents, commitment_loss, codebook_loss)
            z, codes, latents, _, _ = self.model.encode(x)

        return {
            'codes': codes.cpu(),  # [B, N, T] - N codebooks
            'latents': latents.cpu(),  # [B, N*D, T] - projected latents
            'z': z.cpu(),  # [B, latent_dim, T] - quantized representation
            'audio_length': x.shape[-1]
        }

    def get_codebook_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete codes to continuous codebook embeddings

        Args:
            codes: [B, N, T] discrete code indices

        Returns:
            embeddings: [B, N, T, D] continuous embeddings
                where D is codebook_dim (typically 8)
        """
        # Move codes to same device as model
        codes = codes.to(self.device)

        B, N, T = codes.shape

        # Get embeddings for each codebook
        embeddings_list = []

        for i in range(N):
            # Get indices for this codebook
            indices = codes[:, i, :]  # [B, T]

            # Get embeddings from this quantizer's codebook
            quantizer = self.model.quantizer.quantizers[i]
            emb = quantizer.embed_code(indices)  # [B, T, D]

            embeddings_list.append(emb)

        # Stack along codebook dimension: [B, N, T, D]
        embeddings = torch.stack(embeddings_list, dim=1)

        return embeddings

    def extract_pooled_embedding(
        self,
        audio_path: str,
        pooling_method: str = 'mean',
        use_codebook_embeddings: bool = True
    ) -> np.ndarray:
        """
        Extract single vector embedding from audio

        Args:
            audio_path: Path to audio file
            pooling_method: 'mean' or 'flatten'
            use_codebook_embeddings: If True, use continuous codebook embeddings
                                    If False, use discrete codes as features

        Returns:
            Single vector embedding as numpy array
        """
        # Encode audio
        encoded = self.encode_audio(audio_path)
        codes = encoded['codes']  # [1, N, T]

        if use_codebook_embeddings:
            # Get continuous embeddings [1, N, T, D]
            embeddings = self.get_codebook_embeddings(codes)

            if pooling_method == 'mean':
                # Average across time and codebooks: [1, D]
                vector = embeddings.mean(dim=[1, 2]).squeeze(0).detach().cpu().numpy()
            elif pooling_method == 'flatten':
                # Flatten all dimensions: [N*T*D]
                vector = embeddings.squeeze(0).reshape(-1).detach().cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
        else:
            # Use discrete codes directly
            if pooling_method == 'mean':
                # Average across time: [1, N]
                vector = codes.float().mean(dim=2).squeeze(0).detach().cpu().numpy()
            elif pooling_method == 'flatten':
                # Flatten: [N*T]
                vector = codes.squeeze(0).reshape(-1).detach().cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

        return vector


class SpeechCommandsLoader:
    """
    Load audio files from Speech Commands dataset
    """

    def __init__(self, dataset_path: str = "/data/aman/speech_commands/speech_commands_v0.02/"):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    def load_word_samples(
        self,
        words: List[str],
        samples_per_word: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Load audio file paths and labels for specified words

        Args:
            words: List of words to load
            samples_per_word: Number of samples per word

        Returns:
            Tuple of (file_paths, labels)
        """
        file_paths = []
        labels = []

        for word in words:
            word_dir = self.dataset_path / word

            if not word_dir.exists():
                print(f"Warning: Directory not found for word '{word}': {word_dir}")
                continue

            # Get wav files
            wav_files = sorted(list(word_dir.glob("*.wav")))[:samples_per_word]

            for wav_file in wav_files:
                file_paths.append(str(wav_file))
                labels.append(word)

        print(f"Loaded {len(file_paths)} audio files from {len(words)} words")
        return file_paths, labels


def extract_dac_embeddings_batch(
    file_paths: List[str],
    labels: List[str],
    dac_processor: DACProcessor,
    pooling_method: str = 'mean',
    use_codebook_embeddings: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract DAC embeddings for a batch of audio files

    Args:
        file_paths: List of audio file paths
        labels: List of corresponding labels
        dac_processor: DACProcessor instance
        pooling_method: How to pool embeddings ('mean' or 'flatten')
        use_codebook_embeddings: Use continuous embeddings vs discrete codes

    Returns:
        Tuple of (embeddings array, labels)
    """
    embeddings_list = []
    valid_labels = []

    for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc="Extracting DAC embeddings"):
        try:
            embedding = dac_processor.extract_pooled_embedding(
                file_path,
                pooling_method=pooling_method,
                use_codebook_embeddings=use_codebook_embeddings
            )
            embeddings_list.append(embedding)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    embeddings = np.array(embeddings_list)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    return embeddings, valid_labels
