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


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and time embedding injection.

    Each block applies two linear layers with SiLU activations and pre-norm.
    The time embedding is projected to the hidden size and added as a bias.
    """

    def __init__(self, hidden_dim: int, time_embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Zero-init the last layer to encourage identity at start
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_proj(t_emb)

        y = self.norm1(h)
        y = y + temb
        y = self.act1(y)
        y = self.fc1(y)
        y = self.dropout(y)

        y = self.norm2(y)
        y = self.act2(y)
        y = self.fc2(y)
        return h + y


class ScoreNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_res_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, time_embed_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_act = nn.SiLU()
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Initialize last layer to near-zero to stabilize early training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.final_act(self.final_norm(h))
        return self.output_proj(h)
