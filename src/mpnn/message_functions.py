import torch
import torch.nn as nn


class EdgeNetwork(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        )

    def forward(self, edge_attr):
        W = self.net(edge_attr)
        return W.view(-1, self.hidden_dim, self.hidden_dim)


class MessageFunction(nn.Module):

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.edge_net = EdgeNetwork(edge_dim, hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index

        W = self.edge_net(edge_attr)
        h_src = h[src].unsqueeze(-1)

        m = torch.bmm(W, h_src).squeeze(-1)
        return m, dst
