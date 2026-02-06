import torch
import torch.nn as nn


class ReadoutFunction(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim, 1)
        self.transform = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, batch_idx):
        gate = torch.sigmoid(self.gate(h))
        transformed = self.transform(h)

        gated = gate * transformed

        pooled = torch.zeros(
            batch_idx.max() + 1,
            h.size(-1),
            device=h.device
        )
        pooled.index_add_(0, batch_idx, gated)

        return self.out(pooled)
