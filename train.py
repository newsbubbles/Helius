import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import os
import yaml
from models.encoder import Encoder
from models.decoder import Decoder
from models.utils import sample_latent
from models.latent_prior import LatentPrior

class AudioDataset(Dataset):
    """
    Used for VAE training:
    Each item is a single audio waveform [1, T] from data/train_index.txt
    """
    def __init__(self, index_file):
        self.files = [l.strip() for l in open(index_file) if l.strip()]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        # Convert stereo to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio  # shape: [1, T]

def train_vae(config):
    dataset = AudioDataset(config['data']['train_index'])
    loader = DataLoader(
        dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=float(config['training']['lr']))

    def vae_loss(x, x_hat, mu, logvar):
        recon_loss = nn.L1Loss()(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + float(config['training']['kl_weight']) * kl_loss, recon_loss, kl_loss

    os.makedirs(config['training']['ckpt_dir'], exist_ok=True)

    for epoch in range(int(config['training']['epochs'])):
        encoder.train()
        decoder.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            mu, logvar = encoder(batch)
            z = sample_latent(mu, logvar)
            x_hat = decoder(z)
            loss, recon, kl = vae_loss(batch, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{int(config['training']['epochs'])}, Loss: {total_loss/len(loader):.4f}")

        # Save checkpoint
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
        }, os.path.join(config['training']['ckpt_dir'], f"vae_epoch{epoch+1}.pt"))

##############################
# LatentDataset for Prior
##############################
class LatentDataset(Dataset):
    """
    For prior training:
    Each item is a latent sequence [1, T, latent_dim] derived from a single file.
    We run the file through the trained encoder on-the-fly.
    """
    def __init__(self, index_file, vae_ckpt, config):
        self.files = [l.strip() for l in open(index_file) if l.strip()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained VAE
        self.encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
        self.decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)

        state = torch.load(vae_ckpt, map_location=device)
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.encoder.eval()
        self.decoder.eval()

        self.device = device
        self.latent_dim = int(config['model']['latent_dim'])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # shape: [1, T]
        audio = audio.unsqueeze(0).to(self.device)  # now [1, 1, T]
        
        with torch.no_grad():
            mu, logvar = self.encoder(audio)
            z_seq = mu.transpose(1, 2)  # [B=1, T, latent_dim]
        
        return z_seq.squeeze(0)  # return shape: [T, latent_dim]

def collate_latents(batch):
    """
    We receive a list of latent sequences, each shape [T, latent_dim].
    For simpler prior training, let's process them one at a time (batch_size=1).
    But if you want to handle multiple at once, you'll need to handle variable sequence lengths.
    """
    # We'll just return them as-is: a list of [T, latent_dim].
    # The DataLoader will pass them individually if batch_size=1.
    return batch

def train_prior(config, vae_ckpt, train_index):
    """
    Instead of merging everything into one giant tensor,
    we treat each file's latent sequence as a separate training example.
    We'll iterate over them using a DataLoader with batch_size=1.
    """
    # Create a dataset of latent sequences
    latent_dataset = LatentDataset(train_index, vae_ckpt, config)

    loader = DataLoader(
        latent_dataset,
        batch_size=1,            # one latent sequence at a time
        shuffle=True,            # shuffle file order
        num_workers=0,           # or more if you like
        collate_fn=collate_latents
    )

    device = latent_dataset.device
    prior = LatentPrior(latent_dim=int(config['model']['latent_dim']), hidden_size=128).to(device)
    optimizer = optim.Adam(prior.parameters(), lr=float(config['training']['prior_lr']))
    criterion = nn.MSELoss()

    for epoch in range(int(config['training']['prior_epochs'])):
        prior.train()
        total_loss = 0
        count = 0

        for batch_seqs in loader:
            # batch_seqs is a list of single sequences (since batch_size=1, we have just 1 item in the list)
            # each item is shape: [T, latent_dim]
            z_seq = batch_seqs[0].to(device)  # shape [T, latent_dim]
            
            if z_seq.size(0) < 2:
                # Not enough frames for next-step prediction
                continue

            # Make a sequence [1, T, latent_dim] so RNN sees (batch, time, features)
            z_seq = z_seq.unsqueeze(0)  # [1, T, latent_dim]

            input_seq = z_seq[:, :-1, :]  # [1, T-1, latent_dim]
            target_seq = z_seq[:, 1:, :]  # [1, T-1, latent_dim]

            pred = prior(input_seq)  # [1, T-1, latent_dim]
            loss = criterion(pred, target_seq)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / (count if count > 0 else 1)
        print(f"Prior Epoch {epoch+1}/{int(config['training']['prior_epochs'])}, Avg Loss: {avg_loss:.4f}")

    # Finally, save the prior
    os.makedirs(config['training']['ckpt_dir'], exist_ok=True)
    torch.save(prior.state_dict(), os.path.join(config['training']['ckpt_dir'], "prior_final.pt"))

######################################
# main
######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_prior_only", action="store_true")
    parser.add_argument("--vae_ckpt", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not args.train_prior_only:
        train_vae(config)

    if args.vae_ckpt is not None:
        train_prior(config, args.vae_ckpt, config['data']['train_index'])
