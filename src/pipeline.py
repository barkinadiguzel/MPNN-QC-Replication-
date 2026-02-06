from mpnn.mpnn_core import MPNN
from chemistry.atom_features import AtomFeatures
from chemistry.bond_features import BondFeatures
from chemistry.graph_builder import build_graph


class Pipeline:

    def __init__(self, hidden_dim=128, edge_dim=32, out_dim=1, T=3):

        self.model = MPNN(hidden_dim, edge_dim, out_dim, T)
        self.atom_embed = AtomFeatures(hidden_dim)
        self.bond_embed = BondFeatures(edge_dim)

    def forward(self, smiles):

        atomic_numbers, edge_index, bond_types, batch_idx = build_graph(smiles)

        h = self.atom_embed(atomic_numbers)
        edge_attr = self.bond_embed(bond_types)

        return self.model(h, edge_index, edge_attr, batch_idx)
