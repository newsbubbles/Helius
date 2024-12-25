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
    """
    Generates ~length_seconds of audio by:
    1. Loading a trained VAE + prior
    2. Computing how many latent steps needed based on an 8x upsampling (3 conv layers, each stride=2)
    3. Generating latent codes from the prior
    4. Decoding the latent codes
    5. Trimming to exactly length_seconds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE components
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    prior = LatentPrior(latent_dim=int(config['model']['latent_dim'])).to(device)

    vae_state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(vae_state['encoder'])
    decoder.load_state_dict(vae_state['decoder'])

    prior_state = torch.load(prior_ckpt, map_location=device)
    prior.load_state_dict(prior_state)

    ############################################
    # Calculate how many latent steps we need
    ############################################
    total_samples = int(length_seconds * sample_rate)
    # If 3 layers of stride=2 => total upsampling ~8x
    # We'll do a slight offset for partial kernel overlap, but 8 is a good approximation.
    # If your architecture differs, adjust accordingly.
    upsample_factor = 8
    steps = total_samples // upsample_factor
    if steps < 1:
        steps = 1  # fallback, generate at least something

    # Generate latent sequence from the prior
    # shape: [1, steps+1, latent_dim]
    z_seq = prior.generate(steps=steps, device=device)

    # Reshape from [1, T, latent_dim] => [1, latent_dim, T]
    z_seq = z_seq.squeeze(0).transpose(0,1).unsqueeze(0)  # [1, latent_dim, steps]

    # Decode to waveform: [1, 1, ~ (steps * upsample_factor)]
    audio = decoder(z_seq)

    # Trim or pad to exactly total_samples if you want precisely length_seconds
    # shape is [1, 1, num_samples]
    current_len = audio.shape[-1]
    if current_len < total_samples:
        # zero-pad at the end
        pad_amount = total_samples - current_len
        padding = torch.zeros((1, 1, pad_amount), dtype=audio.dtype, device=audio.device)
        audio = torch.cat([audio, padding], dim=-1)
    elif current_len > total_samples:
        audio = audio[..., :total_samples]

    # Squeeze out the batch dim => [channels, num_samples]
    audio = audio.squeeze(0)  # => [1, num_samples]
    audio = audio.cpu()
    
    # Ensure we have shape [channels, samples]
    # If you're sure it's always mono, it's already [1, num_samples].
    # Otherwise, if you had multiple channels, they'd be in dimension 0.

    os.makedirs("generated", exist_ok=True)
    out_path = "generated/generated.wav"
    torchaudio.save(out_path, audio, sample_rate)
    print(f"Generated audio saved at {out_path} (length = {audio.shape[-1]} samples)")

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
