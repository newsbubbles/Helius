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
    def __init__(self, index_file):
        self.files = [l.strip() for l in open(index_file) if l.strip()]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio  # [1, T]

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

# train.py (relevant section)
def train_prior(config, vae_ckpt, train_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()

    files = [l.strip() for l in open(train_index) if l.strip()]
    latents_list = []
    with torch.no_grad():
        for f in files:
            audio, sr = torchaudio.load(f)
            audio = audio.float()
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            # Add batch dimension: now [1, 1, T]
            audio = audio.unsqueeze(0).to(device)

            mu, logvar = encoder(audio)
            z = mu  # deterministic latent
            z_seq = z.transpose(1, 2).contiguous()  # [1, T, latent_dim]
            latents_list.append(z_seq)
    
    latents = torch.cat(latents_list, dim=1).contiguous()  # [1, Total_T, latent_dim]
    input_seq = latents[:, :-1, :].contiguous()  # [1, Total_T-1, latent_dim]
    target_seq = latents[:, 1:, :].contiguous()  # [1, Total_T-1, latent_dim]

    prior = LatentPrior(latent_dim=int(config['model']['latent_dim']), hidden_size=128).to(device)
    # Ensure GRU parameters are flattened
    prior.rnn.flatten_parameters()
    
    optimizer = optim.Adam(prior.parameters(), lr=float(config['training']['prior_lr']))
    criterion = nn.MSELoss()

    for epoch in range(int(config['training']['prior_epochs'])):
        prior.train()
        pred = prior(input_seq)  # [1, T-1, latent_dim]
        loss = criterion(pred, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Prior Epoch {epoch+1}/{int(config['training']['prior_epochs'])}, Loss: {loss.item():.4f}")

    torch.save(prior.state_dict(), os.path.join(config['training']['ckpt_dir'], "prior_final.pt"))

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
