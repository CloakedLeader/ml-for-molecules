from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond, rdDistGeom, rdForceFieldHelpers, rdMolAlign, Conformer, rdPartialCharges, rdMolDescriptors, Descriptors, Crippen
import torch
from torch import Tensor
from torch_geometric.data import Data
import pandas as pd
from pandas.core.frame import DataFrame
from typing import Optional, Sequence
import numpy as np

num_conf = 50 

COMMON_ATOMS = ['H','C','N','O','F','P','S','Cl','Br','I']

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

class MoleculeRepresentation:
    def __init__(self, smiles: str, inh_pow: float):
        self.smiles = smiles
        self.molecule = Chem.MolFromSmiles(smiles)
        self.inh_pow = inh_pow

    @staticmethod
    def one_hot(value, choices):
        return [int(value == c) for c in choices]
    
    def atom_features(self, atom: Atom, conf: Conformer) -> Tensor:
        """Creates a pytorch tensor that is the node-features for a specific atom.

        Args:
            atom (Atom): The atom(node) in the molecule to get the data for.

        Returns:
            Tensor: Has dimension 23 and contains  things like atomic number, degree, formal charge, chirality
                and whether something is aromatic. These are returned as floats in the tensor.
        """
        rdPartialCharges.ComputeGasteigerCharges(self.molecule)
        symbol = atom.GetSymbol()
        features = []

        # One-hot atomic symbol (10 dim)
        features += self.one_hot(symbol, COMMON_ATOMS)
        
        # Atomic number (scaled) for uncommon atoms (1 dim)
        features.append(atom.GetAtomicNum() / 100.0)

        # Degree (1 dim)
        features.append(atom.GetTotalDegree() / 4.0)

        # Formal Charge (1 dim)
        features.append(atom.GetFormalCharge() / 5.0)

        # Hydridization (5 dim)
        features += self.one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)

        # Aromaticity (1 dim)
        features.append(int(atom.GetIsAromatic()))

        # Number of Hydrogens (1 dim)
        features.append(atom.GetTotalNumHs() / 4.0)

        # In Ring (1 dim)
        features.append(int(atom.IsInRing()))

        # Chirality (2 dim)
        chiral_tag = atom.GetChiralTag()
        features.append(int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW))
        features.append(int(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW))

        # Atomic Mass (1 dim) (scaled)
        features.append(atom.GetMass() * 0.01)

        # Partial Charge (1 dim)
        # charge = float(atom.GetProp("_GasteigerCharge"))
        # features.append(charge)

        # # Positions (3 dim)
        # pos = conf.GetAtomPosition(atom.GetIdx())
        # features += [pos.x, pos.y, pos.z]

        return torch.tensor(features, dtype=torch.float)

    def bond_features(self, bond: Bond) -> Tensor:
        """Creates a pytorch tensor that is the edge features for a specific bond.

        Args:
            bond (Bond): The bond(edge) in the molecule to get the data for.

        Returns:
            Tensor: Has dimension 4 and is a one-hot encoding which tells the model whether
                the bond is single, double, triple or aromatic.
        """
        return torch.tensor([
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
            ], dtype=torch.float)
    
    @staticmethod
    def pick_central_conformer(molecule: Mol, conf_ids: Sequence[int]) -> int:
        rmsd_matrix = []
        for i in conf_ids:
            row = []
            for j in conf_ids:
                rms = rdMolAlign.GetBestRMS(molecule, molecule, prbId=i, refId=j)
                row.append(rms)
            rmsd_matrix.append(row)
        
        rmsd_matrix = np.array(rmsd_matrix)
        mean_rmsd = rmsd_matrix.mean(axis=1)
        return conf_ids[int(np.argmin(mean_rmsd))]
    
    
    def find_best_conformer(self) -> Optional[Conformer]:
        if self.molecule is None:
            return None
        self.molecule_3d = Chem.AddHs(self.molecule)
        
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 111
        params.numThreads = 0
        params.pruneRmsThresh = 0.1
        conf_ids = rdDistGeom.EmbedMultipleConfs(self.molecule_3d, numConfs=num_conf, params=params)
        if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(self.molecule_3d):
            print("UFF does not support this molecule. Skipping minimization and using approximate geometry.")
            best_id = self.pick_central_conformer(self.molecule_3d, conf_ids)
            return self.molecule_3d.GetConformer(best_id)
        
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.molecule_3d)
        energies = []
        for cid in conf_ids:
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                self.molecule_3d,
                mp,
                confId=cid
            )
            ff.Minimize()
            energies.append((cid, ff.CalcEnergy()))
        


        energies.sort(key=lambda x: x[1])
        best_conf = energies[0][0]
        conf = Conformer(self.molecule_3d.GetConformer(best_conf))
        self.molecule_3d.RemoveAllConformers()
        self.molecule_3d.AddConformer(conf, assignId=True)
        return conf

    def global_features(self):
        mol = self.molecule
        
        return torch.tensor([
            # Descriptors.HeavyMolWt(mol),
            # Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
        ], dtype=torch.float)

    def mol_to_graph(self) -> Optional[Data]:
        """Creates a graph with the necessary embeddings from a given SMILES string.

        Args:
            smiles (str): The molecular structure given in SMILES format.
            inh_pow (float): The inhibition power for the given molecule, this is the target
                property.

        Returns:
            Data: A torch-geometric Data type which is a graph containing all the structural and
                embedding information. If the SMILES string is unreadable or corrupt then None is
                returned.
        """
        if self.molecule is None:
            return None
        
        best_conf = self.find_best_conformer()

        # Node features
        x = torch.stack([self.atom_features(atom, best_conf) for atom in self.molecule.GetAtoms()])

    
    
            
        # Edges
        edge_ind:list = []
        edge_att:list = []

        def add_edge(i, j, attr):
            edge_ind.append([i,j])
            edge_att.append(attr)

        for bond in self.molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bf = self.bond_features(bond)
            if torch.isnan(bf).any():
                print("NaNs in bond features!")
                print("SMILES:", self.smiles)
                return None

            # Undirected graph → add both directions
            add_edge(i, j, bf)
            add_edge(j, i, bf)

        

        # Turns a list of tuples into a tensor and then changes the shape and memory type.
        edge_index = torch.tensor(edge_ind, dtype=torch.long).t().contiguous()
        # Combines a list of tensors into one big tensor of the required shape.
        edge_attr = torch.stack(edge_att)

        y = torch.tensor([self.inh_pow], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def batch_from_csv(dataframe: DataFrame) -> DataFrame:
    """
    Takes a .csv file (database) and turns each row into a graph using the mol_to_graph
    function.

    Args:
        csv_path (str): The path of the .csv file.

    Returns:
        list[Data]: A list of graphs that will then be put into the model for training.
    """

    df = dataframe
    graphs = []
    for _, row in df.iterrows():
        g = MoleculeRepresentation(row["SMILES"], row["Inh Power"])
        graph = g.mol_to_graph()
        graphs.append(graph)
    
    df["graph"] = graphs
    return df
