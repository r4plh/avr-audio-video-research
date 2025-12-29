from transformers import Wav2VecModel, Wav2Vec2Processor
from transformers import WhisperModel, WhisperProcessor
import os
import librosa
import numpy as np
import torch

wa2vec_model = Wav2VecModel.from_pretrained("facebook/wav2vec2-base-960h")
wa2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

test_words = ['yes', 'no', 'up', 'down', 'left']
dataset_path = "/data/aman/speech_commands/speech_commands_v0.02/"

def extract_embeddings(audio_path, model_type="wav2vec"):

    audio, sr = librosa.load(audio_path, sr=16000)

    if model_type == "wav2vec":
        inputs = wa2vec_processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = wa2vec_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    elif model_type == 'whisper':
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = whisper_model.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

wav2vec_embeddings = []
whisper_embeddings = []
labels = []

# Sample 50 files per word (to start)
for word in test_words:
    word_dir = os.path.join(dataset_path, word)
    files = os.listdir(word_dir)[:50]  # Just 50 samples per word initially
    
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(word_dir, file)
            
            # Get embeddings from both models
            wav2vec_emb = extract_embeddings(path, 'wav2vec')
            whisper_emb = extract_embeddings(path, 'whisper')
            
            wav2vec_embeddings.append(wav2vec_emb)
            whisper_embeddings.append(whisper_emb)
            labels.append(word)