from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond, rdDistGeom, rdForceFieldHelpers, rdMolAlign, Conformer, rdPartialCharges, rdMolDescriptors, Descriptors, Crippen
import torch
from torch import Tensor
from torch_geometric.data import Data
import pandas as pd
from pandas.core.frame import DataFrame
from typing import Optional, Sequence
import numpy as np

num_conf = 100

COMMON_ATOMS = ['H','C','N','O','F','P','S','Cl','Br','I']

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

class MoleculeRepresentation:
    def __init__(self, smiles: str, inh_pow: float, struct_3d: bool, seed: int | None = None):
        self.smiles = smiles
        self.molecule = Chem.MolFromSmiles(smiles)
        self.inh_pow = inh_pow
        self.three_d = struct_3d
        self.seed = seed

    @staticmethod
    def one_hot(value, choices):
        return [int(value == c) for c in choices]
    
    def atom_features(self, atom: Atom) -> Tensor:
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

        # # Positions (3 dim)
        # pos = conf.GetAtomPosition(atom.GetIdx())
        # features += [pos.x, pos.y, pos.z]

        return torch.tensor(features, dtype=torch.float)

    def atom_features_3d(self, atom: Atom, conf: Conformer, mean: float) -> Tensor:
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
        
        # Positions (3 dim)
        pos = conf.GetAtomPosition(atom.GetIdx())
        features += [pos.x/ mean, pos.y / mean, pos.z / mean]

        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def compute_atom_distances(bond: Bond, conform: Conformer) -> float:
        beg = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        beg_pos = conform.GetAtomPosition(beg)
        beg_pos = np.array([beg_pos.x, beg_pos.y, beg_pos.z])
        end_pos = conform.GetAtomPosition(end)
        end_pos = np.array([end_pos.x, end_pos.y, end_pos.z])
        return float(np.linalg.norm(beg_pos - end_pos))
    
    def bond_features(self, bond: Bond, conform: Conformer | None = None, mean = None, std = None) -> Tensor:
        """Creates a pytorch tensor that is the edge features for a specific bond.

        Args:
            bond (Bond): The bond(edge) in the molecule to get the data for.

        Returns:
            Tensor: Has dimension 4 and is a one-hot encoding which tells the model whether
                the bond is single, double, triple or aromatic.
        """
        if conform:
            dist = self.compute_atom_distances(bond, conform)
        if self.three_d:
            if mean and std:
            
                return torch.tensor([
                    bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
                    bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
                    bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
                    bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                    (dist - mean) / std,
                    ], dtype=torch.float)
            else:
                raise ValueError("Need to add mean and standard deviation to normalise distances!")
        else:
            return torch.tensor([
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
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
    
    
    def find_best_conformer(self) -> Conformer:
        # if self.molecule is None:
        #     return None
        self.molecule_3d = Chem.AddHs(self.molecule)
        
        params = rdDistGeom.ETKDGv3()
        # params.randomSeed = self.seed or 111
        params.numThreads = 0
        params.pruneRmsThresh = 0.01
        conf_ids = rdDistGeom.EmbedMultipleConfs(self.molecule_3d, numConfs=num_conf, params=params)
        if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(self.molecule_3d):
            print("UFF does not support this molecule. Skipping minimization and using approximate geometry.")
            best_id = self.pick_central_conformer(self.molecule_3d, conf_ids)
            # conf = self.molecule_3d.GetConformer(best_id)
            # self.molecule_3d.RemoveAllConformers()
            # self.molecule_3d.AddConformer(conf, assignId=True)
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
        # self.molecule_3d.RemoveAllConformers()
        # self.molecule_3d.AddConformer(conf, assignId=True)
        return conf

    def mol_to_2d_graph(self) -> Optional[Data]:
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
        # if self.molecule_3d.GetNumConformers() != 1:
        #     best_conf = self.find_best_conformer()
        # else:
        #     best_conf = self.molecule_3d.GetConformer()

        # Node features
        x = torch.stack([self.atom_features(atom) for atom in self.molecule.GetAtoms()])
            
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

            # Undirected graph → add both directions
            add_edge(i, j, bf)
            add_edge(j, i, bf)

        # Turns a list of tuples into a tensor and then changes the shape and memory type.
        edge_index = torch.tensor(edge_ind, dtype=torch.long).t().contiguous()
        # Combines a list of tensors into one big tensor of the required shape.
        edge_attr = torch.stack(edge_att)

        y = torch.tensor([self.inh_pow], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    

    def mol_to_3d_graph(self, mean, std, conf: Conformer, dist: bool, coord: bool):
        if self.molecule is None:
            return None
        if conf:
            best_conf = conf
        else:
            best_conf = self.find_best_conformer()

        # Node features
        if coord:
            x = torch.stack([self.atom_features_3d(atom, best_conf,mean) for atom in self.molecule.GetAtoms()])
        else:
            x = torch.stack([self.atom_features(atom) for atom in self.molecule.GetAtoms()])
        
        # Edges
        edge_ind:list = []
        edge_att:list = []

        def add_edge(i, j, attr):
            edge_ind.append([i,j])
            edge_att.append(attr)

        for bond in self.molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if dist:
                bf = self.bond_features(bond, conform=best_conf, mean=mean, std=std)
            else:
                bf = self.bond_features(bond)

            # Undirected graph → add both directions
            add_edge(i, j, bf)
            add_edge(j, i, bf)

        # Turns a list of tuples into a tensor and then changes the shape and memory type.
        edge_index = torch.tensor(edge_ind, dtype=torch.long).t().contiguous()
        # Combines a list of tensors into one big tensor of the required shape.
        edge_attr = torch.stack(edge_att)

        y = torch.tensor([self.inh_pow], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def batch_from_csv(dataframe: DataFrame, struct_3d: bool, include_coord: bool = True, include_dist: bool = True) -> DataFrame:
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
    if struct_3d == False:
        for _, row in df.iterrows():
            g = MoleculeRepresentation(row["SMILES"], row["Inh Power"], struct_3d)
            graph = g.mol_to_2d_graph()
            graphs.append(graph)

    else:
        count = 0
        mean = 0
        M2 = 0
        for _, row in df.iterrows():
            mole = MoleculeRepresentation(row["SMILES"], row["Inh Power"], struct_3d)
            conf = mole.find_best_conformer()

            for bond in mole.molecule.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                pos_i = conf.GetAtomPosition(i)
                pos_j = conf.GetAtomPosition(j)

                dist = pos_i.Distance(pos_j)
                count += 1
                delta = dist - mean
                mean += delta / count
                delta2 = dist - mean
                M2 += delta * delta2
        
        std = (M2 / count) ** 0.5
        for _, row in df.iterrows():
            mole = MoleculeRepresentation(row["SMILES"], row["Inh Power"], struct_3d)
            conf = mole.find_best_conformer()

            graph = mole.mol_to_3d_graph(mean, std, conf, dist=include_dist, coord=include_coord)
            graphs.append(graph)

    df["graph"] = graphs
    return df


def compute_atom_distances(bond: Bond, conform: Conformer) -> float:
        beg = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        beg_pos = conform.GetAtomPosition(beg)
        beg_pos = np.array([beg_pos.x, beg_pos.y, beg_pos.z])
        end_pos = conform.GetAtomPosition(end)
        end_pos = np.array([end_pos.x, end_pos.y, end_pos.z])
        return float(np.linalg.norm(beg_pos - end_pos))
