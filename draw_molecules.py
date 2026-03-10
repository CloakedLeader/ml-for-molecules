from proc.smiles_to_graph import MoleculeRepresentation
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "input.csv"


class MoleculeDrawing:
    def __init__(self, smiles: str, inhib_pow: float):
        self.smiles = smiles
        self.power = inhib_pow
        self.constructor = MoleculeRepresentation(smiles, inhib_pow)

    
    def create_graph(self, name: str = "molecule.svg") -> nx.Graph:
        mol = self.constructor.molecule
        G = nx.Graph()

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            G.add_node(idx, label=atom.GetSymbol())
        
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            G.add_edge(a1, a2, order=bond.GetBondType())

        pos = nx.spring_layout(G, seed=42)
        labels = nx.get_node_attributes(G, "label")

        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, font_size=12)
        plt.savefig(name)

df = pd.read_csv(csv_path)
df = df.sort_values("Inh Power", ascending=False)
for i in range(0, 10):
    
    drawer = MoleculeDrawing(df.iloc[-i]["SMILES"], df.iloc[-i]["Inh Power"])
    drawer.create_graph(f"molecule_{10-i}.svg")


