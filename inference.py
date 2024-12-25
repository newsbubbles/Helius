import torch
import torchaudio
import argparse
import yaml
import os
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_prior import LatentPrior

@torch.no_grad()
def generate_audio(config, vae_ckpt, prior_ckpt, length_seconds=10, sample_rate=48000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=config['model']['latent_dim']).to(device)
    decoder = Decoder(latent_dim=config['model']['latent_dim']).to(device)
    prior = LatentPrior(latent_dim=config['model']['latent_dim']).to(device)

    vae_state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(vae_state['encoder'])
    decoder.load_state_dict(vae_state['decoder'])

    prior_state = torch.load(prior_ckpt, map_location=device)
    prior.load_state_dict(prior_state)

    # Assume each latent frame corresponds to a smaller chunk of audio.
    # Here we must define a relationship between latent steps and audio length.
    # Let's assume the encoder downsamples by a factor of 8 each time (3 conv layers stride=2^3=8).
    # If input was T samples, output latent has length T/8. For simplicity, assume 512 samples per latent step.
    # length_seconds * sample_rate / 512 ~ number of latent steps
    steps = int((length_seconds * sample_rate) / (8*32))  # Adjust based on actual model downsampling.
    if steps <= 0:
        steps = 100  # fallback

    z_seq = prior.generate(steps=steps, device=device) # [1, T+1, latent_dim]
    z_seq = z_seq.squeeze(0).transpose(0,1).unsqueeze(0) # [1, latent_dim, T]
    audio = decoder(z_seq) # [1, 1, T_samples]
    audio = audio.cpu()
    
    # Give it a lil' squeeze to get that batch dim out! yeet!
    audio = audio.squeeze(0)
    
    os.makedirs("generated", exist_ok=True)
    out_path = "generated/generated.wav"
    torchaudio.save(out_path, audio, sample_rate)
    print(f"Generated audio saved at {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--length_seconds", type=float, default=10)
    parser.add_argument("--sample_rate", type=int, default=48000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    generate_audio(config, args.vae_ckpt, args.prior_ckpt, args.length_seconds, args.sample_rate)
