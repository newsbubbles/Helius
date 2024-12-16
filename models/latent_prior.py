import torch
import torch.nn as nn

class LatentPrior(nn.Module):
    def __init__(self, latent_dim=64, hidden_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.rnn = nn.GRU(latent_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, z_seq):
        # z_seq: [B, T, latent_dim]
        out, _ = self.rnn(z_seq.contiguous())
        # Typically we want to predict the next frame given the current frame
        # This returns predictions at each step. For next-step prediction:
        return self.fc(out)

    def generate(self, steps=100, start_z=None, device='cpu'):
        # Autoregressive generation
        self.eval()
        with torch.no_grad():
            if start_z is None:
                start_z = torch.zeros(1, 1, self.latent_dim).to(device)
            h = None
            out_seq = [start_z]
            for _ in range(steps):
                # Pass the last frame as input
                out, h = self.rnn(out_seq[-1], h)  # out: [1,1,hidden_size]
                pred = self.fc(out) # [1,1,latent_dim]
                out_seq.append(pred)
            return torch.cat(out_seq, dim=1) # [1, T+1, latent_dim]
