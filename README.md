# Helius Project

Helius is a project designed to train and run a state-of-the-art unsupervised real-time audio generation system. It leverages an encoder-decoder (VAE-like) architecture to learn a latent representation of audio and a latent prior model (e.g., an RNN) to generate or transform audio signals in real-time.

## Project Overview

The system is composed of three major parts:

1. **VAE (Encoder-Decoder) Model**:  
   - **Encoder**: Converts raw audio waveforms into a compressed latent representation.
   - **Decoder**: Reconstructs waveforms from latent representations.
   
   This stage is trained in an unsupervised manner from large collections of raw audio (MP3s converted to WAV segments). The model learns to compress and reconstruct a wide variety of timbres and acoustic structures.

2. **Latent Prior**:  
   After training the VAE, we learn a model (e.g., an RNN or Transformer) that predicts future latent vectors given past ones. This allows:
   - **Dream Mode**: Generate new audio “from scratch” by sampling from the prior in latent space.
   - **Audio-to-Audio Transformation**: Given a stream of input audio, we can modify, style-transfer, or extrapolate it in a latent-driven manner.

3. **Real-Time Inference**:  
   We provide scripts for both offline inference (generating arbitrary-length audio from a trained model) and a streaming pipeline that processes incoming audio frames and outputs transformed or "dreamed" audio in real-time.

## Architecture Details

- **Encoder**: A convolutional neural network that downsamples audio into latent codes. Each latent frame represents a small chunk of audio. The encoder outputs both a mean and a log-variance (in a VAE setting), enabling a continuous latent space with smooth interpolation.
  
- **Decoder**: A transposed convolutional or neural vocoder-based network that upsamples latent codes back to the raw audio domain. Trained together with the encoder, it learns to produce high-quality waveform reconstructions from compressed representations.

- **Latent Prior**: A recurrent neural network (GRU) that models sequences in latent space. By learning temporal dependencies, it can predict coherent latent sequences over time, thus generating endless musical or sonic textures when run autoregressively.

### Caveats and Considerations

- **Quality-Fidelity Tradeoff**: The latent compression may lead to audio quality degradation compared to the original signal. High-fidelity results might require more complex architectures (e.g., using a stronger decoder or a more powerful vocoder).
  
- **Unsupervised Nature**: With no labels, the model doesn’t learn specific musical structures or semantic concepts. It just captures statistical regularities. The generated output might be abstract or noisy depending on the training data distribution.
  
- **Computational Requirements**: Training on large audio datasets can be computationally intensive. A GPU (or multiple) is strongly recommended.

- **Data Preparation**: The model relies on having a large corpus of audio preprocessed into WAV segments. The quality and diversity of your training data heavily influence the quality and style of the generated audio.

## Setup

1. **Prerequisites**:
   - Python 3.9+ recommended
   - GPU with CUDA support for training (strongly recommended)
   
2. **Dependencies**:
   After setting up the project files (as per the `setup` script), install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Convert your MP3 files to WAV and create segments for training:
   ```bash
   python data_prep.py --input_dir /path/to/mp3s --output_dir data --segment_length 2.0 --sample_rate 48000
   ```
   This generates `data/train_index.txt` listing all WAV segments.  
   *Tip:* The `segment_length` and `sample_rate` should match your training configuration.

## Training

1. **Train the VAE**:
   In `configs/config.yaml`, ensure `data/train_index` points to your prepared `train_index.txt`.  
   
   Run:
   ```bash
   python train.py --config configs/config.yaml
   ```
   This will:
   - Train the encoder-decoder network for the configured number of epochs.
   - Save checkpoints in `checkpoints/`.

2. **Train the Latent Prior**:
   After VAE training, pick a suitable checkpoint (e.g., `checkpoints/vae_epoch10.pt`) and run:
   ```bash
   python train.py --config configs/config.yaml --train_prior_only --vae_ckpt checkpoints/vae_epoch10.pt
   ```
   This extracts latent sequences from your dataset using the trained encoder and trains the latent prior model.  
   
   After completion, `checkpoints/prior_final.pt` will be created.

## Inference

1. **Offline Generation ("Dream Mode")**:
   Once you have a trained VAE and prior model, generate audio by:
   ```bash
   python inference.py --config configs/config.yaml --vae_ckpt checkpoints/vae_epoch10.pt --prior_ckpt checkpoints/prior_final.pt --length_seconds 10
   ```
   This will generate 10 seconds of audio from the model’s latent prior and save it in `generated/generated.wav`.

2. **Streaming Inference (Audio-to-Audio Transformation)**:
   If you have an input WAV file and want to transform it in a streaming manner:
   ```bash
   python inference_streaming.py --config configs/config.yaml --vae_ckpt checkpoints/vae_epoch10.pt --prior_ckpt checkpoints/prior_final.pt --input_wav input.wav --output_wav generated/streaming_out.wav
   ```
   This simulates a pipeline where audio is processed in chunks. In a real-time scenario, you’d continuously feed new audio chunks and output transformed audio on-the-fly.

## Testing

- **Test Generation**:
  If you have run training and have checkpoints, you can run:
  ```bash
  python tests/test_generation.py
  ```
  This script attempts to generate a short audio sample (5 seconds) using the current VAE and Prior checkpoints.  
  Ensure `vae_epoch10.pt` and `prior_final.pt` exist in the `checkpoints/` directory, or adjust the script accordingly.

## Example Workflows

1. **Full Workflow**:
   - Prepare data:
     ```bash
     python data_prep.py --input_dir /mydata/mp3s --output_dir data
     ```
   - Train VAE:
     ```bash
     python train.py --config configs/config.yaml
     ```
   - Train Prior:
     ```bash
     python train.py --config configs/config.yaml --train_prior_only --vae_ckpt checkpoints/vae_epoch10.pt
     ```
   - Generate:
     ```bash
     python inference.py --config configs/config.yaml --vae_ckpt checkpoints/vae_epoch10.pt --prior_ckpt checkpoints/prior_final.pt --length_seconds 10
     ```
   
2. **Just Dream Some Audio**:
   If you’ve already got trained models:
   ```bash
   python inference.py --config configs/config.yaml --vae_ckpt checkpoints/vae_epoch10.pt --prior_ckpt checkpoints/prior_final.pt --length_seconds 30
   ```

3. **Apply Transform to Existing Audio**:
   ```bash
   python inference_streaming.py --config configs/config.yaml --vae_ckpt checkpoints/vae_epoch10.pt --prior_ckpt checkpoints/prior_final.pt --input_wav my_input.wav --output_wav transformed_output.wav
   ```

## Future Directions

- **Advanced Architectures**:  
  Enhance the decoder with a neural vocoder for higher-quality audio or use a transformer-based latent prior for more complex long-term structure.
  
- **Conditional Generation**:  
  Integrate metadata from MP3 tags or additional style embeddings to guide generation toward specific genres, instruments, or moods.

- **Online Learning**:  
  Adapt the model continuously as new data arrives, enabling incremental improvements over time.

Helius provides a flexible foundation. Experiment with architecture adjustments, hyperparameters, and different datasets to achieve desired audio quality and characteristics.