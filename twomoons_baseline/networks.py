import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, num_train_steps: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_train_steps = num_train_steps
        # Use Diffusers' sinusoidal projection + small MLP used in many diffusion models
        self.time_proj = Timesteps(embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_mlp = TimestepEmbedding(embed_dim, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) in [0,1]
        timesteps = (t.clamp(0, 1) * (self.num_train_steps - 1)).view(-1)
        t_proj = self.time_proj(timesteps)
        return self.time_mlp(t_proj)


class ScoreNet(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, time_embed_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # Initialize last layer to near-zero to stabilize early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)
