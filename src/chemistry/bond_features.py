import torch
import torch.nn as nn


class BondFeatures(nn.Module):
    def __init__(self, edge_dim, num_bond_types=10):
        super().__init__()
        self.embedding = nn.Embedding(num_bond_types, edge_dim)

    def forward(self, bond_types):
        return self.embedding(bond_types)
