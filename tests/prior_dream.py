#!/usr/bin/env python3
"""
prior_dream.py

Generates audio by:
1. Loading a trained VAE (encoder + decoder) and a trained Prior (RNN).
2. Optionally picking a random file from the dataset to get a "warm-start" latent frame.
3. Autoregressively generating latents for a specified number of steps.
4. Decoding the generated latents into audio.
5. Saving the resulting "dream" waveform to disk.

Usage Example:
  python prior_dream.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch100.pt \
    --prior_ckpt checkpoints/prior_final.pt \
    --data_index data/train_index.txt \
    --sample_rate 48000 \
    --steps 100 \
    --out_path my_dream.wav \
    --warm_start
"""

import argparse
import os
import random
import yaml
import torch
import torchaudio

from torch.utils.data import Dataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_prior import LatentPrior

##############################################
# Simple Dataset for optional warm start
##############################################
class AudioDataset(Dataset):
    """
    Loads audio files from an index for optional warm-start latent extraction.
    """
    def __init__(self, index_file, sample_rate=48000):
        self.files = [l.strip() for l in open(index_file) if l.strip()]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        # Convert stereo to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample if needed
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        return audio, path

##############################################
# Helper functions
##############################################
def load_vae_and_prior(config, vae_ckpt, prior_ckpt, device):
    """
    Loads a trained VAE (encoder+decoder) and a trained prior (RNN).
    """
    latent_dim = int(config['model']['latent_dim'])

    # Load VAE
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    vae_state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(vae_state['encoder'])
    decoder.load_state_dict(vae_state['decoder'])
    encoder.eval()
    decoder.eval()

    # Load Prior
    prior = LatentPrior(latent_dim=latent_dim).to(device)
    prior_state = torch.load(prior_ckpt, map_location=device)
    prior.load_state_dict(prior_state)
    prior.eval()

    return encoder, decoder, prior

def warm_start_latent(encoder, audio_dataset, device):
    """
    Picks a random file from the dataset, encodes it to latents,
    and returns the final latent frame to use as a starting seed.

    Returns shape [1, 1, latent_dim].
    """
    if len(audio_dataset) == 0:
        raise ValueError("No audio files in dataset; can't warm start.")
    idx = random.randint(0, len(audio_dataset)-1)
    audio, path = audio_dataset[idx]
    audio = audio.unsqueeze(0).to(device)  # => [B=1, 1, T]
    with torch.no_grad():
        mu, logvar = encoder(audio)
        # We'll just take mu
        # shape: [1, latent_dim, T'] or [B=1,latent_dim,T'] if multiple frames
        # If it's a short segment, might have T'=1 or more. We'll pick the last frame:
        if mu.dim() == 3:
            # let's pick the last frame in time dimension
            start_z = mu[:, :, -1:]  # => [1, latent_dim, 1]
            # But prior.generate expects [1,1,latent_dim]
            start_z = start_z.transpose(1,2)  # => [1,1,latent_dim]
        else:
            # If the model outputs 2D shape [B, latent_dim], just reshape
            start_z = mu.unsqueeze(1)  # => [1,1,latent_dim]
    print(f"Warm start latent from file={path}, shape={start_z.shape}")
    return start_z

def decode_latent_sequence(decoder, z_seq):
    """
    Decodes a sequence of latents: shape [1, T, latent_dim]
    into a single waveform by concatenating the outputs for each frame.
    This approach is simplistic: each latent -> a small wave chunk, then concat.

    If your decoder expects (batch,latent_dim,time), we do shape manipulations accordingly.
    """
    device = next(decoder.parameters()).device
    B, T, D = z_seq.shape

    # We'll decode frame by frame or all at once if your decoder can handle [B*T,latent_dim,1].
    # For demonstration, let's decode each frame individually, then stitch them.
    # A more advanced approach might decode them as a single input with T frames in latent space.
    chunks = []
    with torch.no_grad():
        for t in range(T):
            z_t = z_seq[:, t, :].unsqueeze(-1)  # => [1, latent_dim, 1]
            audio_chunk = decoder(z_t)  # => [1,1,samples]
            chunks.append(audio_chunk)
    # Concatenate along time dimension
    audio = torch.cat(chunks, dim=-1)  # => [1,1, total_samples_across_T]
    return audio

def main():
    parser = argparse.ArgumentParser(description="Use a trained prior to generate dream audio.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--vae_ckpt", required=True, help="Path to trained VAE checkpoint (encoder+decoder)")
    parser.add_argument("--prior_ckpt", required=True, help="Path to trained prior checkpoint")
    parser.add_argument("--data_index", type=str, default=None,
                        help="Optional index file for warm-start latents (if using --warm_start)")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--steps", type=int, default=100, help="Number of latent frames to generate")
    parser.add_argument("--out_path", type=str, default="prior_dream.wav")
    parser.add_argument("--warm_start", action="store_true", help="Use a real latent from dataset to seed generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load VAE + Prior
    encoder, decoder, prior = load_vae_and_prior(config, args.vae_ckpt, args.prior_ckpt, device)

    # Possibly do warm start
    if args.warm_start:
        if not args.data_index:
            raise ValueError("Must provide --data_index when using --warm_start.")
        dataset = AudioDataset(args.data_index, sample_rate=args.sample_rate)
        start_z = warm_start_latent(encoder, dataset, device)  # shape [1,1,latent_dim]
    else:
        # default to zero or random seed
        latent_dim = int(config['model']['latent_dim'])
        start_z = torch.zeros(1,1,latent_dim, device=device)
        print(f"Starting from zero latent: shape={start_z.shape}")

    # Generate using the prior
    # shape => [1, T+1, latent_dim]
    z_seq = prior.generate(steps=args.steps, start_z=start_z, device=device)
    # Possibly skip the first frame or do whatever you like. We'll keep them all.

    # Reshape to [1, T, latent_dim]
    # prior.generate returns [1, T+1, latent_dim]. Let's keep everything for demonstration.
    B, T, D = z_seq.shape
    print(f"Generated latent seq: shape={z_seq.shape}")
    
    # Decode into a single waveform
    audio = decode_latent_sequence(decoder, z_seq)  # => [1,1,total_samples]

    # Flatten to [channels, samples] for torchaudio.save
    audio = audio.squeeze(0)  # => [1, total_samples]
    audio = audio.cpu()

    # Save
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torchaudio.save(args.out_path, audio, args.sample_rate)
    print(f"Saved dream audio to {args.out_path}, shape={tuple(audio.shape)}.")


if __name__ == "__main__":
    main()
