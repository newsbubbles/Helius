import os
import argparse
import torch
import torchaudio
import yaml
import random
from torch.utils.data import DataLoader, Dataset

from models.encoder import Encoder
from models.decoder import Decoder
from models.utils import sample_latent

# Optional: if you want an STFT or Mel-based metric, you can define it or use torchaudio transforms.
# For example, a simple multi-scale STFT approach might be introduced or a single STFT-based metric.

def compute_l1_loss(a, b):
    return torch.mean(torch.abs(a - b))

def compute_mse_loss(a, b):
    return torch.mean((a - b) ** 2)

# If you want a simple spectral distance (not multi-scale, just for debugging):
def compute_spectral_loss(a, b, sample_rate=48000):
    """
    Compute a naive spectral difference via STFT magnitude. 
    a, b: [B, 1, T] waveforms on the same device.
    Returns average L1 difference of STFT magnitudes (a small proxy).
    """
    # Hyperparams for STFT - you can tweak
    n_fft = 1024
    hop_length = 256
    stft_a = torch.stft(a.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=a.device), return_complex=True)
    stft_b = torch.stft(b.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=b.device), return_complex=True)
    mag_a = torch.abs(stft_a)
    mag_b = torch.abs(stft_b)
    return torch.mean(torch.abs(mag_a - mag_b))

class SingleAudioDataset(Dataset):
    """
    For "dataset" mode, we pass an index of .wav files (like train_index).
    We'll simply load them on the fly, returning [1, T] waveforms.
    """
    def __init__(self, index_file):
        self.files = [l.strip() for l in open(index_file) if l.strip()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        audio = audio.float()
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # convert to mono
        return audio, path

def reconstruct_audio(audio, encoder, decoder, sample_latent_fn=True):
    """
    audio: [1, T] on device
    If sample_latent_fn=True, uses mu, logvar -> sample_latent. 
    Otherwise, use just mu for deterministic reconstruction.
    """
    with torch.no_grad():
        mu, logvar = encoder(audio.unsqueeze(0))  # [B=1, latent_dim*2, T'] => mu/logvar
        if sample_latent_fn:
            z = sample_latent(mu, logvar)
        else:
            z = mu
        rec = decoder(z)
    return rec

def save_audio(tensor, sample_rate, path):
    """
    tensor: [B=1, 1, T] or [1, T]
    """
    # Squeeze batch dim if needed
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # shape: [1, T]
    # Ensure on CPU
    tensor = tensor.cpu()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, tensor, sample_rate)

def test_single_file_audio(config, vae_ckpt, audio_path, output_dir, sample_rate=48000, sample_latent_fn=True):
    """
    Test reconstruction on a single audio file. Saves the original (optionally) and the reconstruction,
    and prints reconstruction metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio = audio.float()
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)  # mono
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.to(device)

    # Reconstruct
    rec = reconstruct_audio(audio, encoder, decoder, sample_latent_fn=sample_latent_fn)

    # Compute some metrics
    l1 = compute_l1_loss(rec, audio.unsqueeze(0))
    mse = compute_mse_loss(rec, audio.unsqueeze(0))
    spectral = compute_spectral_loss(rec, audio.unsqueeze(0), sample_rate=sample_rate)
    print(f"Metrics for {audio_path}: L1={l1.item():.4f}, MSE={mse.item():.6f}, Spectral={spectral.item():.4f}")

    # Save recon
    output_rec_path = os.path.join(output_dir, "reconstructed.wav")
    save_audio(rec, sample_rate, output_rec_path)
    print(f"Reconstruction saved to: {output_rec_path}")

def test_dataset_audio(config, vae_ckpt, index_file, output_dir, sample_rate=48000, num_examples=3, sample_latent_fn=True):
    """
    Test reconstruction on a small subset of a dataset index. 
    Randomly samples 'num_examples' files from index_file, reconstructs each, 
    and saves them + prints metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()

    dataset = SingleAudioDataset(index_file)
    if len(dataset) == 0:
        print("No audio files found in index_file.")
        return
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:num_examples]

    for i, idx in enumerate(selected_indices):
        audio, path = dataset[idx]
        sr = 48000  # or read from actual file if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.to(device)
        
        rec = reconstruct_audio(audio, encoder, decoder, sample_latent_fn=sample_latent_fn)

        l1 = compute_l1_loss(rec, audio.unsqueeze(0))
        mse = compute_mse_loss(rec, audio.unsqueeze(0))
        spectral = compute_spectral_loss(rec, audio.unsqueeze(0), sample_rate=sample_rate)

        print(f"[{i+1}] File: {path}")
        print(f"    L1: {l1.item():.4f}, MSE: {mse.item():.6f}, Spectral: {spectral.item():.4f}")

        # Save reconstruction
        file_name = f"reconstructed_{i+1}.wav"
        output_path = os.path.join(output_dir, file_name)
        save_audio(rec, sample_rate, output_path)
        print(f"    Saved reconstruction to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="VAE Reconstruction Debugging & Testing")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--vae_ckpt", required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--mode", choices=["single", "dataset"], default="single", help="Which test mode to run")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to single audio file for reconstruction")
    parser.add_argument("--index_file", type=str, default=None, help="Index file for multiple audio test")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Directory to save reconstructions")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate for resampling and saving")
    parser.add_argument("--num_examples", type=int, default=3, help="How many audio files to reconstruct in dataset mode")
    parser.add_argument("--no_sample", action="store_true", help="Use mu only (deterministic), skip random sampling from logvar")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Decide if we sample from latent or not
    sample_latent_fn = not args.no_sample

    if args.mode == "single":
        if not args.audio_path:
            raise ValueError("Must provide --audio_path for single mode.")
        test_single_file_audio(
            config, 
            args.vae_ckpt, 
            args.audio_path, 
            args.output_dir, 
            sample_rate=args.sample_rate,
            sample_latent_fn=sample_latent_fn
        )
    else:  # dataset mode
        if not args.index_file:
            raise ValueError("Must provide --index_file for dataset mode.")
        test_dataset_audio(
            config, 
            args.vae_ckpt, 
            args.index_file, 
            args.output_dir, 
            sample_rate=args.sample_rate,
            num_examples=args.num_examples,
            sample_latent_fn=sample_latent_fn
        )

if __name__ == "__main__":
    main()

"""
# Usage Examples

## Single File
```
python test.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch10.pt \
    --mode single \
    --audio_path data/example.wav \
    --output_dir vae_recon_debug \
    --sample_rate 48000
```

## Multi-file From Index
```
python test.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch10.pt \
    --mode dataset \
    --index_file data/train_index.txt \
    --output_dir vae_recon_debug \
    --num_examples 5 \
    --sample_rate 48000
```

## Non-logvar 
```
python test.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch10.pt \
    --mode single \
    --audio_path data/example.wav \
    --output_dir vae_recon_debug \
    --no_sample
```
"""