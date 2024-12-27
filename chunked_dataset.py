# chunked_dataset.py

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class ChunkedAudioDataset(Dataset):
    """
    Loads a list of audio files and returns random short chunks from them.
    This is useful for training a real-time-friendly VAE or prior model.
    """

    def __init__(self, index_file, chunk_size, sample_rate=48000, in_memory=False):
        """
        Args:
          index_file (str): Path to a text file listing all audio files.
          chunk_size (int): Number of samples per chunk (e.g., 2048 samples ~ 42ms at 48kHz).
          sample_rate (int): Target sample rate for training.
          in_memory (bool): If True, load all audio into RAM at init. 
                            If False, load + chunk on-the-fly (slower but uses less memory).
        """
        super().__init__()
        self.files = [ln.strip() for ln in open(index_file) if ln.strip()]
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.in_memory = in_memory

        self.audios = []  # Will hold (waveform, sr) if in_memory=True

        if self.in_memory:
            print("Loading all audio into memory...")
            for path in self.files:
                audio, sr = torchaudio.load(path)
                audio = audio.float()
                if audio.shape[0] > 1:
                    # Mix stereo to mono
                    audio = torch.mean(audio, dim=0, keepdim=True)
                if sr != sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, sample_rate)
                self.audios.append((audio, sample_rate))
            print("Finished loading audio into RAM.")

    def __len__(self):
        # We can sample from any file, so let's just return a large nominal number
        # or len(self.files) if you want exactly 1 chunk per file per epoch.
        # But for chunk-based training, it's common to treat the dataset as large indefinite.
        return len(self.files)  

    def __getitem__(self, idx):
        """
        Returns a random chunk of length self.chunk_size from the idx-th file (if not in_memory)
        or from a random file if you prefer. 
        We'll randomize properly so each chunk is from a random file offset.
        """
        # If you want a truly random file each time, you could ignore idx and do random choice:
        # But we'll do a consistent approach: pick the file by idx, random offset inside it.

        if self.in_memory:
            # Use the idx-th audio in memory
            audio, sr = self.audios[idx]
        else:
            # Load on-the-fly
            path = self.files[idx]
            audio, sr = torchaudio.load(path)
            audio = audio.float()
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        num_samples = audio.shape[-1]
        if num_samples <= self.chunk_size:
            # If the file is smaller than chunk_size, either pad or just return the whole thing
            # We'll do simple "no pad, just return"
            return audio
        else:
            # Pick a random offset
            max_offset = num_samples - self.chunk_size
            start = random.randint(0, max_offset)
            end = start + self.chunk_size
            chunk = audio[:, start:end]
            return chunk
