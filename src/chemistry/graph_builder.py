from rdkit import Chem
import torch


def build_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetAtomicNum())

    atomic_numbers = torch.tensor(atoms, dtype=torch.long)

    src = []
    dst = []
    bond_types = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        btype = int(bond.GetBondTypeAsDouble())

        src += [i, j]
        dst += [j, i]
        bond_types += [btype, btype]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    bond_types = torch.tensor(bond_types, dtype=torch.long)

    batch_idx = torch.zeros(len(atoms), dtype=torch.long)

    return atomic_numbers, edge_index, bond_types, batch_idx
