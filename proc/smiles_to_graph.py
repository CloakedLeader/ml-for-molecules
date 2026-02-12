from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond
import torch
from torch import Tensor
from torch_geometric.data import Data
import pandas as pd


COMMON_ATOMS = ['H','C','N','O','F','P','S','Cl','Br','I']

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot(value, choices):
    return [int(value == c) for c in choices]


def atom_features(atom: Atom) -> Tensor:
    """Creates a pytorch tensor that is the node-features for a specific atom.

    Args:
        atom (Atom): The atom(node) in the molecule to get the data for.

    Returns:
        Tensor: Has dimension 23 and contains  things like atomic number, degree, formal charge, chirality
            and whether something is aromatic. These are returned as floats in the tensor.
    """

    symbol = atom.GetSymbol()
    features = []

    # One-hot atomic symbol (10 dim)
    features += one_hot(symbol, COMMON_ATOMS)
    
    # Atomic number (scaled) for uncommon atoms (1 dim)
    features.append(atom.GetAtomicNum() / 100.0)

    # Degree (1 dim)
    features.append(atom.GetTotalDegree())

    # Formal Charge (1 dim)
    features.append(atom.GetFormalCharge())

    # Hydridization (5 dim)
    features += one_hot(atom.GetHybridization, HYBRIDIZATION_TYPES)

    # Aromaticity (1 dim)
    features.append(int(atom.GetIsAromatic()))

    # Number of Hydrogens (1 dim)
    features.append(atom.GetTotalNumHs())

    # In Ring (1 dim)
    features.append(int(atom.IsInRing()))

    # Chirality (2 dim)
    chiral_tag = atom.GetChiralTag()
    features.append(int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW))
    features.append(int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW))

    # Atomic Mass (1 dim) (scaled)
    features.append(atom.GetMass() * 0.01)

    return torch.tensor(features, dtype=torch.float)

def bond_features(bond: Bond) -> Tensor:
    """Creates a pytorch tensor that is the edge features for a specific bond.

    Args:
        bond (Bond): The bond(edge) in the molecule to get the data for.

    Returns:
        Tensor: Has dimension 4 and is a one-hot encoding which tells the model whether
            the bond is single, double, triple or aromatic.
    """
    return torch.tensor([
        int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
        int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
        int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
        int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
    ], dtype=torch.float)

def mol_to_graph(smiles, inh_pow):
    mol: Mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Edges
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Undirected graph → add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])

        bf = bond_features(bond)
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    y = torch.tensor([inh_pow], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def batch_from_csv(csv_path: str) -> list[Data]:
    df = pd.read_csv(csv_path)
    graphs = []
    for _, row in df.iterrows():
        g = mol_to_graph(row["SMILES"], row["Inh Power"])
        if g is not None:
            graphs.append(g)

    return graphs
