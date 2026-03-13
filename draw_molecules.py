from proc.smiles_to_graph import MoleculeRepresentation
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as patheffects

csv_path = "input.csv"


class MoleculeDrawing:
    def __init__(self, smiles: str, inhib_pow: float, name: str):
        self.smiles = smiles
        self.power = inhib_pow
        self.name = name
        self.constructor = MoleculeRepresentation(smiles, inhib_pow)

    
    def create_2d_graph(self, name: str = "molecule.svg") -> None:
        mol = Chem.AddHs(self.constructor.molecule)
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
        plt.title(f"2D Plot of {self.name}")
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, font_size=12)
        plt.savefig(name)

    def create_3d_graph(self, name: str = "molecule_3d.png") -> None:
        conf = self.constructor.find_best_conformer()
        mol = self.constructor.molecule_3d

        G = nx.Graph()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            G.add_node(idx,
                       element=atom.GetSymbol(),
                       coord=(pos.x, pos.y, pos.z))
            
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       order=bond.GetBondTypeAsDouble())
            
        atom_colors = {
            "H": "white",
            "C": "black",
            "N": "blue",
            "O": "red",
            "F": "green",
            "P": "orange",
            "S": "yellow",
            "Cl": "green",
            "Br": "brown",
            "I": "purple",
        }

        fig =plt.figure(figsize=(6,6))
        fig.suptitle(f"3D Conformer of {self.name}")
        ax = fig.add_subplot(111, projection="3d")

        for n, data in G.nodes(data=True):
            x, y, z = data["coord"]
            elem = data["element"]
            ax.scatter(x, y, z,
                    color=atom_colors.get(elem, "gray"),
                    s=200,
                    edgecolors="k",
                    depthshade=True
            )
            ax.text(x, y, z, elem,
                    fontsize=10,
                    ha="center", 
                    va="center",
                    color="white",
                    path_effects=[patheffects.withStroke(linewidth=1, foreground="black")]
                )
            
        for u, v in G.edges():
            x1, y1, z1 = G.nodes[u]["coord"]
            x2, y2, z2 = G.nodes[v]["coord"]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", linewidth=3)

        set_axes_equal(ax)
        ax.view_init(elev=20, azim=30)
        plt.tight_layout()
        plt.savefig(name, dpi=300)
        plt.close(fig)

    def create_pdb_file(self, name) -> None:
        self.constructor.find_best_conformer()
        pdb_block = Chem.MolToPDBBlock(self.constructor.molecule_3d)
        with open(name, "w") as f:
            f.write(pdb_block)



def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

df = pd.read_csv(csv_path)
df = df.sort_values("Inh Power", ascending=False)

for index, j in enumerate([df.iloc[2], df.iloc[3]]):

    drawer = MoleculeDrawing(j["SMILES"], j["Inh Power"], j["Inhibitor Name"])
    drawer.create_pdb_file(f"3d_molecule{index}.pdb")





