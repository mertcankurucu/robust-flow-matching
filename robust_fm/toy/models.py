"""MLP vector fields used by the core 2D experiments."""

import torch
from torch import nn


class VectorField(nn.Module):
    """MLP used by the latest core sweep (standard initialization)."""

    def __init__(self, dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t], dim=1)
        out = self.net(xt)
        return out


class ConditionalVectorField(nn.Module):
    """Neural network to parameterize the conditional vector field v(x, t, y)"""

    def __init__(self, dim=2, n_classes=2, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.label_embed = nn.Embedding(n_classes, 32)
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        input_dim = dim + 64 + 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t, y):
        """
        Args:
            x: (batch, dim) - spatial coordinates
            t: (batch, 1) or (batch,) - time
            y: (batch,) - class labels (integers)
        Returns:
            v: (batch, dim) - velocity vector
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)
        y_emb = self.label_embed(y)
        inp = torch.cat([x, t_emb, y_emb], dim=1)
        v = self.net(inp)
        return v
