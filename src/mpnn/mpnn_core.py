import torch.nn as nn

from .message_functions import MessageFunction
from .update_functions import UpdateFunction
from .readout_functions import ReadoutFunction


class MPNN(nn.Module):

    def __init__(self, hidden_dim, edge_dim, out_dim, T=3):
        super().__init__()

        self.T = T

        self.message_fn = MessageFunction(hidden_dim, edge_dim)
        self.update_fn = UpdateFunction(hidden_dim)
        self.readout_fn = ReadoutFunction(hidden_dim, out_dim)

    def forward(self, h, edge_index, edge_attr, batch_idx):

        for _ in range(self.T):
            m, dst = self.message_fn(h, edge_index, edge_attr)
            h = self.update_fn(h, m, dst)

        return self.readout_fn(h, batch_idx)
