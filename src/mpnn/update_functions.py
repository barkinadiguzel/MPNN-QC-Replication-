import torch
import torch.nn as nn


class UpdateFunction(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, m, dst):
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        return self.gru(agg, h)
