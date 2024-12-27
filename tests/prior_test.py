import torch
import argparse
import yaml
import os
import random
import torchaudio

from torch.utils.data import Dataset

from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_prior import LatentPrior

#############################
#   Utility / Helper Fns
#############################

def load_vae_and_prior(config, vae_ckpt, prior_ckpt, device):
    """
    Loads the trained VAE (encoder+decoder) and the prior model.
    """
    # VAE
    encoder = Encoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    decoder = Decoder(latent_dim=int(config['model']['latent_dim'])).to(device)
    vae_state = torch.load(vae_ckpt, map_location=device)
    encoder.load_state_dict(vae_state['encoder'])
    decoder.load_state_dict(vae_state['decoder'])
    encoder.eval()
    decoder.eval()

    # Prior
    prior = LatentPrior(latent_dim=int(config['model']['latent_dim'])).to(device)
    prior_state = torch.load(prior_ckpt, map_location=device)
    prior.load_state_dict(prior_state)
    prior.eval()

    return encoder, decoder, prior

def latent_mse(pred, target):
    """
    Simple MSE in latent space.
    pred, target: [B, T, latent_dim]
    """
    return torch.mean((pred - target) ** 2)

def decode_latents(decoder, latents):
    """
    latents: [B, T, latent_dim] => decode => [B, 1, num_samples]
    """
    # The decoder expects [B, latent_dim, T']
    # So transpose 1,2
    latents = latents.transpose(1,2)  # => [B, latent_dim, T]
    with torch.no_grad():
        audio = decoder(latents)
    return audio

def save_wav(tensor, sample_rate, path):
    """
    tensor: [B, 1, T] or [1, T], on CPU or GPU
    We'll make sure it's [channels, samples] on CPU for torchaudio.
    """
    tensor = tensor.squeeze(0).detach().cpu()  # => [1, T]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, tensor, sample_rate)

#############################
#   Single-File Test
#############################

def test_prior_on_file(
    config,
    vae_ckpt,
    prior_ckpt,
    audio_path,
    sample_rate=48000,
    output_dir="prior_test_outputs"
):
    """
    1. Load the file, pass it through encoder => latents
    2. Split latents in half:
       - prior "sees" first half
       - prior must predict second half autoregressively
    3. Compare predicted latents to actual latents
    4. Optionally decode both to audio and save
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, prior = load_vae_and_prior(config, vae_ckpt, prior_ckpt, device)

    # --- Load audio file ---
    audio, sr = torchaudio.load(audio_path)
    audio = audio.float()
    # convert to mono if needed
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.unsqueeze(0).to(device)  # => [B=1, 1, T]

    # --- Encode to latent ---
    with torch.no_grad():
        mu, logvar = encoder(audio)
        # We'll just use mu for deterministic check
        z_seq = mu.transpose(1,2)  # => [1, T_lat, latent_dim]

    T_lat = z_seq.shape[1]
    if T_lat < 4:
        print(f"Latent sequence too short (only {T_lat} frames). Not enough to test.")
        return

    # --- Split latents in half ---
    half = T_lat // 2
    obs_seq = z_seq[:, :half, :]   # prior sees these frames
    true_future = z_seq[:, half:, :]  # the actual "future"

    # --- Autoregressive prediction ---
    # We'll feed obs_seq to the prior to generate frames up to total length = half + length_of_future
    steps_to_generate = T_lat - half

    # We'll do something like:
    # 1) Start with obs_seq
    # 2) Recurrently generate next latent frames
    # *OR* we can feed the entire obs_seq to prior in a single forward, but since it's an RNN expecting next-step,
    #    we might do it step-by-step. We'll do a manual loop for clarity.
    #    If your prior has a convenience method like .generate(), you can use that, but let's be explicit here.

    pred_future = []
    hidden = None

    # For GRU: we can feed one frame at a time
    current_input = obs_seq[:, -1:, :]  # the last frame from obs_seq is the "seed"
    for _ in range(steps_to_generate):
        # forward pass for one step
        # prior expects [B, T, latent_dim], let's pass [B,1,latent_dim]
        out, hidden = prior.rnn(current_input, hidden)  # => out shape: [B=1, T=1, hidden_size]
        pred_next = prior.fc(out)  # => [B=1, T=1, latent_dim]
        pred_future.append(pred_next)
        current_input = pred_next  # feed the newly generated frame as next input

    pred_future = torch.cat(pred_future, dim=1)  # => [1, steps_to_generate, latent_dim]

    # Combine predicted future with the observed seq for reference if we want
    pred_entire_seq = torch.cat([obs_seq, pred_future], dim=1)  # => [1, T_lat, latent_dim]

    # Compute MSE in latent space for the future portion only
    latent_mse_val = latent_mse(pred_future, true_future).item()
    print(f"Latent MSE for the future portion = {latent_mse_val:.6f}")

    # --- Decode predicted future & actual future for comparison ---
    pred_future_audio = decode_latents(decoder, pred_future)  # => [1, 1, ~samples]
    true_future_audio = decode_latents(decoder, true_future)

    # Save them
    out_pred = os.path.join(output_dir, "predicted_future.wav")
    out_true = os.path.join(output_dir, "true_future.wav")
    save_wav(pred_future_audio, sample_rate, out_pred)
    save_wav(true_future_audio, sample_rate, out_true)
    print(f"Saved predicted future audio to {out_pred}")
    print(f"Saved actual future audio to {out_true}")

    # If you want, decode entire predicted sequence
    pred_entire_audio = decode_latents(decoder, pred_entire_seq)
    out_pred_entire = os.path.join(output_dir, "predicted_entire_seq.wav")
    save_wav(pred_entire_audio, sample_rate, out_pred_entire)
    print(f"Saved predicted entire seq audio to {out_pred_entire}")

    # Optional: decode entire real sequence for direct comparison
    entire_real_seq = z_seq  # the entire ground truth
    entire_real_audio = decode_latents(decoder, entire_real_seq)
    out_entire_real = os.path.join(output_dir, "true_entire_seq.wav")
    save_wav(entire_real_audio, sample_rate, out_entire_real)
    print(f"Saved ground truth entire seq audio to {out_entire_real}")

def test_prior_on_dataset(
    config,
    vae_ckpt,
    prior_ckpt,
    index_file,
    output_dir="prior_test_outputs",
    sample_rate=48000,
    num_files=3
):
    """
    Does the same as test_prior_on_file but for multiple files from an index,
    picking random subset (num_files). For each file, we do:
     - Encode -> z_seq
     - Observed half vs. predicted half
     - MSE in latent space
     - Save some example audio
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, prior = load_vae_and_prior(config, vae_ckpt, prior_ckpt, device)

    with open(index_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    random.shuffle(lines)
    selected = lines[:num_files]

    for i, audio_path in enumerate(selected, start=1):
        print(f"\n=== [File {i}/{num_files}] {audio_path} ===")
        file_output_dir = os.path.join(output_dir, f"file_{i}")
        os.makedirs(file_output_dir, exist_ok=True)

        audio, sr = torchaudio.load(audio_path)
        audio = audio.float()
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = audio.unsqueeze(0).to(device)

        with torch.no_grad():
            mu, logvar = encoder(audio)
            z_seq = mu.transpose(1,2)  # [1, T_lat, latent_dim]

        T_lat = z_seq.shape[1]
        if T_lat < 4:
            print(f"Latent seq too short (only {T_lat} frames). Skipping.")
            continue

        half = T_lat // 2
        obs_seq = z_seq[:, :half, :]
        true_future = z_seq[:, half:, :]

        # Autoregressive prediction
        steps_to_generate = T_lat - half
        pred_future = []
        hidden = None
        current_input = obs_seq[:, -1:, :]
        for _ in range(steps_to_generate):
            out, hidden = prior.rnn(current_input, hidden)
            pred_next = prior.fc(out)
            pred_future.append(pred_next)
            current_input = pred_next
        pred_future = torch.cat(pred_future, dim=1)  # [1, steps_to_generate, latent_dim]

        # MSE
        mse_val = latent_mse(pred_future, true_future).item()
        print(f"Latent MSE (future portion) = {mse_val:.6f}")

        # Decode predicted future vs. true
        pred_future_audio = decode_latents(decoder, pred_future)
        true_future_audio = decode_latents(decoder, true_future)

        save_wav(pred_future_audio, sample_rate, os.path.join(file_output_dir, "predicted_future.wav"))
        save_wav(true_future_audio, sample_rate, os.path.join(file_output_dir, "true_future.wav"))

        print(f"Saved predicted future & true future audio to {file_output_dir}")

#######################################
#              MAIN
#######################################
def main():
    parser = argparse.ArgumentParser(description="Test Prior by Next-Step Prediction in Latent Space")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--audio_path", type=str, default=None, help="Single-file test mode")
    parser.add_argument("--index_file", type=str, default=None, help="Dataset test mode")
    parser.add_argument("--output_dir", type=str, default="prior_test_outputs")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--num_files", type=int, default=3, help="How many files to sample from index_file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.audio_path is not None:
        # single-file test
        test_prior_on_file(
            config,
            args.vae_ckpt,
            args.prior_ckpt,
            args.audio_path,
            sample_rate=args.sample_rate,
            output_dir=args.output_dir
        )
    elif args.index_file is not None:
        # multi-file test
        test_prior_on_dataset(
            config,
            args.vae_ckpt,
            args.prior_ckpt,
            args.index_file,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            num_files=args.num_files
        )
    else:
        raise ValueError("Provide either --audio_path (single file) or --index_file (dataset)")

if __name__ == "__main__":
    main()

"""
## Single file prior test
```
python prior_test.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch100.pt \
    --prior_ckpt checkpoints/prior_final.pt \
    --audio_path data_test/test_11.wav \
    --output_dir prior_test_debug \
    --sample_rate 48000
```
## Dataset prior
```
python prior_test.py \
    --config configs/config.yaml \
    --vae_ckpt checkpoints/vae_epoch100.pt \
    --prior_ckpt checkpoints/prior_final.pt \
    --index_file data/train_index.txt \
    --output_dir prior_test_debug \
    --sample_rate 48000 \
    --num_files 5
```
"""