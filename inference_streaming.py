import torch
import torchaudio
import argparse
import yaml
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_prior import LatentPrior

@torch.no_grad()
def streaming_inference(config, vae_ckpt, prior_ckpt, input_wav, output_wav, chunk_size=4800):
    """
    Demonstration of a streaming inference pipeline:
    - Read input audio in chunks
    - Encode chunk into latent
    - Optionally pass through a latent prior or modify latent
    - Decode latent back to audio in real-time
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=config['model']['latent_dim']).to(device)
    decoder = Decoder(latent_dim=config['model']['latent_dim']).to(device)
    prior = LatentPrior(latent_dim=config['model']['latent_dim']).to(device)

    vae_state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(vae_state['encoder'])
    decoder.load_state_dict(vae_state['decoder'])

    prior_state = torch.load(prior_ckpt, map_location=device)
    prior.load_state_dict(prior_state)

    audio, sr = torchaudio.load(input_wav)
    if sr != config['inference']['sample_rate']:
        audio = torchaudio.functional.resample(audio, sr, config['inference']['sample_rate'])
        sr = config['inference']['sample_rate']

    # Assume chunk_size samples per chunk
    output_audio = []
    T = audio.shape[-1]
    start = 0

    # For streaming, you might maintain a running RNN state or do incremental encoding.
    # Here we simply process each chunk independently as a demonstration.
    while start < T:
        end = min(start + chunk_size, T)
        chunk = audio[:, start:end].to(device)
        start = end

        # Encode
        mu, logvar = encoder(chunk.unsqueeze(0)) # [1, latent_dim, T']
        # For simplicity, take mu as latent
        z = mu

        # Optionally, run through prior somehow. For streaming, we might skip temporal modeling
        # or use a simple step. Let's just decode directly for now.
        # If we had a mechanism: z_seq = prior(...) # Not well-defined for streaming in this simplistic code

        # Decode
        reconstructed = decoder(z) # [1, 1, T_samples_reconstructed]
        output_audio.append(reconstructed.squeeze(0).cpu())

    output_audio = torch.cat(output_audio, dim=-1) # [1, total_samples]
    torchaudio.save(output_wav, output_audio, sr)
    print(f"Streaming inference output saved to: {output_wav}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="generated/streaming_out.wav")
    parser.add_argument("--chunk_size", type=int, default=4800)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    streaming_inference(config, args.vae_ckpt, args.prior_ckpt, args.input_wav, args.output_wav, args.chunk_size)
