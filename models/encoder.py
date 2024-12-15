import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):
        super().__init__()
        # A simple convolutional encoder (1D)
        # In a real system, consider more sophisticated architectures.
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, 9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim*2, 9, stride=2, padding=4) # outputs [mu, logvar]
        )

    def forward(self, x):
        # x: [B, 1, T]
        h = self.conv(x)
        B, C, T = h.shape
        half = C // 2
        mu, logvar = h[:, :half, :], h[:, half:, :]
        return mu, logvar
