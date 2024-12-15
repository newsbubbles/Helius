import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, output_channels=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, 9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_channels, 9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [B, latent_dim, T']
        return self.deconv(z)
