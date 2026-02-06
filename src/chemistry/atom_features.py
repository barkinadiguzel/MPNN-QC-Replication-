import torch
import torch.nn as nn


class AtomFeatures(nn.Module):
    def __init__(self, hidden_dim, max_atomic_num=100):
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_num, hidden_dim)

    def forward(self, atomic_numbers):
        return self.embedding(atomic_numbers)
