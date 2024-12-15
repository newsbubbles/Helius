import torchaudio
import torch

def load_audio_segment(path, target_sr=48000):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio, target_sr

def save_audio(path, audio, sr=48000):
    torchaudio.save(path, audio, sr)
