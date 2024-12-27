# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import os
import yaml
import random

from chunked_dataset import ChunkedAudioDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.utils import sample_latent
from models.latent_prior import LatentPrior

########################
# VAE Training
########################
def train_vae(config):
    # 1) Create a chunked dataset
    chunk_size = int(config['training']['chunk_size_vae'])  # e.g. ~1920 samples for ~40ms
    dataset = ChunkedAudioDataset(
        index_file=config['data']['train_index'],
        chunk_size=chunk_size,
        sample_rate=config['inference']['sample_rate'],
        in_memory=False  # or True if you have enough RAM
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config['training']['batch_size']),  # e.g. 8
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = int(config['model']['latent_dim'])
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(config['training']['lr'])
    )

    def vae_loss(x, x_hat, mu, logvar):
        recon_loss = nn.L1Loss()(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + float(config['training']['kl_weight']) * kl_loss, recon_loss, kl_loss

    os.makedirs(config['training']['ckpt_dir'], exist_ok=True)
    epochs = int(config['training']['epochs'])

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        for batch in loader:
            # batch: [B, 1, chunk_size_samples]
            batch = batch.to(device)
            mu, logvar = encoder(batch)       # shape depends on your encoder
            z = sample_latent(mu, logvar)     # [B, latent_dim, T'] or something
            x_hat = decoder(z)                # [B, 1, chunk_size_samples] presumably

            loss, recon, kl = vae_loss(batch, x_hat, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[VAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
        }, os.path.join(config['training']['ckpt_dir'], f"vae_epoch{epoch+1}.pt"))

########################
# Prior Training
########################
class LatentChunkedDataset(torch.utils.data.Dataset):
    """
    For prior training:
    We create longer chunks (e.g., 200-300ms) to capture short local context.
    Then for each chunk, we encode it to latents. That yields a time series of latents
    for next-step prediction.
    """
    def __init__(self, index_file, vae_ckpt, config):
        super().__init__()
        self.files = [ln.strip() for ln in open(index_file) if ln.strip()]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained VAE
        latent_dim = int(config['model']['latent_dim'])
        self.encoder = Encoder(latent_dim=latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim=latent_dim).to(self.device)
        state = torch.load(vae_ckpt, map_location=self.device)
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.encoder.eval()
        self.decoder.eval()

        # This chunk size is for prior training, e.g. 300ms
        self.chunk_size = int(config['training']['chunk_size_prior'])
        self.sample_rate = int(config['inference']['sample_rate'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # If audio is shorter than chunk_size, we just return it as is.
        # Or random offset if > chunk_size
        length = audio.shape[-1]
        if length > self.chunk_size:
            start = random.randint(0, length - self.chunk_size)
            audio = audio[:, start:start+self.chunk_size]

        audio = audio.unsqueeze(0).to(self.device)  # [1,1,chunk_size]
        with torch.no_grad():
            mu, logvar = self.encoder(audio)  # shape: [1, latent_dim, T'] or [1, latent_dim] 
            # If the encoder downsamples in time, T' might be >1 for chunk_size. 
            z_seq = mu.transpose(1,2) if mu.dim() == 3 else mu.unsqueeze(2).transpose(1,2)
            # => [1, T', latent_dim]

        return z_seq.squeeze(0)  # => [T', latent_dim]

def collate_latent_chunks(batch):
    """
    batch is a list of latent sequences [T', latent_dim], possibly different lengths.
    We can just return them as a list. For now, we do batch_size=1 anyway.
    """
    return batch

def train_prior(config, vae_ckpt):
    dataset = LatentChunkedDataset(
        index_file=config['data']['train_index'],
        vae_ckpt=vae_ckpt,
        config=config
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # each chunk is one item
        shuffle=True,
        num_workers=0,
        collate_fn=collate_latent_chunks
    )

    device = dataset.device
    latent_dim = int(config['model']['latent_dim'])
    prior = LatentPrior(latent_dim=latent_dim, hidden_size=128).to(device)
    optimizer = optim.Adam(prior.parameters(), lr=float(config['training']['prior_lr']))
    criterion = nn.MSELoss()

    epochs = int(config['training']['prior_epochs'])

    for epoch in range(epochs):
        prior.train()
        total_loss = 0.0
        count = 0
        for batch_data in loader:
            # batch_data is a list of single items, each [T', latent_dim]
            z_seq = batch_data[0].to(device)  # [T', latent_dim]
            if z_seq.size(0) < 2:
                continue
            z_seq = z_seq.unsqueeze(0)  # => [1, T', latent_dim]

            input_seq = z_seq[:, :-1, :]
            target_seq = z_seq[:, 1:, :]

            pred = prior(input_seq)  # => [1, T'-1, latent_dim]
            loss = criterion(pred, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"[Prior] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save final prior
    os.makedirs(config['training']['ckpt_dir'], exist_ok=True)
    torch.save(prior.state_dict(), os.path.join(config['training']['ckpt_dir'], "prior_final.pt"))

###################################
# Main
###################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_prior_only", action="store_true")
    parser.add_argument("--vae_ckpt", type=str, default=None,
                        help="Path to VAE checkpoint for prior training")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Train VAE if not skipping
    if not args.train_prior_only:
        train_vae(config)

    # 2. Train prior if vae_ckpt is given
    if args.vae_ckpt is not None:
        train_prior(config, args.vae_ckpt)
