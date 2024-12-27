#!/usr/bin/env python3

"""
umap_plot.py

Generates a UMAP 2D chart from the latents of a trained VAE.
Requires: pip install umap-learn matplotlib
"""

import os
import argparse
import torch
import torchaudio
import yaml
import umap
import matplotlib.pyplot as plt
import numpy as np

from models.encoder import Encoder
from models.decoder import Decoder

def load_vae(config, vae_ckpt, device):
    latent_dim = int(config['model']['latent_dim'])
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    encoder.eval()
    decoder.eval()
    return encoder, decoder

def collect_latents(encoder, index_file, device, sample_rate=48000, max_files=500):
    """
    Iterates over up to `max_files` from index_file, encodes them, collects mu.
    Returns a list of [latent_dim] vectors on CPU.
    """
    latents = []
    files = [l.strip() for l in open(index_file) if l.strip()]
    if len(files) > max_files:
        files = files[:max_files]
    print(f"Collecting latents from {len(files)} files...")

    with torch.no_grad():
        for i, path in enumerate(files):
            audio, sr = torchaudio.load(path)
            audio = audio.float()
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != sample_rate:
                audio = torchaudio.functional.resample(audio, sr, sample_rate)
            audio = audio.unsqueeze(0).to(device)  # [1, 1, T]

            mu, logvar = encoder(audio)
            # Suppose mu is [B, latent_dim, T'] or [B, latent_dim]
            # We'll just flatten or take the average over T' dimension if it exists
            if mu.dim() == 3:
                # shape = [1, latent_dim, T'], take mean over T'
                mu_agg = mu.mean(dim=2).squeeze(0)  # => [latent_dim]
            else:
                mu_agg = mu.squeeze(0)  # => [latent_dim]
            
            latents.append(mu_agg.cpu().numpy())
    return np.array(latents)  # shape [N, latent_dim]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--vae_ckpt", required=True, help="Path to VAE checkpoint")
    parser.add_argument("--index_file", required=True, help="List of audio files for collecting latents")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--max_files", type=int, default=500, help="Limit how many files to embed")
    parser.add_argument("--out_png", type=str, default="umap_latents.png", help="Where to save the chart")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load VAE
    encoder, decoder = load_vae(config, args.vae_ckpt, device)

    # Collect latents
    latents = collect_latents(encoder, args.index_file, device, args.sample_rate, args.max_files)
    print(f"Collected latents shape: {latents.shape}")

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    latents_2d = reducer.fit_transform(latents)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], s=10, alpha=0.7)
    plt.title("UMAP of VAE Latent Distribution")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.savefig(args.out_png)
    print(f"UMAP chart saved to {args.out_png}")

if __name__ == "__main__":
    main()


"""
python umap_plot.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch100.pt \
    --index_file data/train_index.txt \
    --sample_rate 48000 \
    --max_files 500 \
    --out_png latent_umap.png
"""