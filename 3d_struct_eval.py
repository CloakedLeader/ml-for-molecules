from rdkit import Chem
from rdkit.Chem import Mol, Atom, Bond, rdDistGeom, rdForceFieldHelpers, rdMolAlign, Conformer, rdPartialCharges, rdMolDescriptors, Descriptors, Crippen, AllChem
import torch
from torch import Tensor
from torch_geometric.data import Data
import pandas as pd
from pandas.core.frame import DataFrame
from typing import Optional, Sequence
import numpy as np
import requests
from urllib.parse import quote
import gc
from matplotlib import pyplot as plt

from proc.smiles_to_graph import MoleculeRepresentation





def get_energy_dist(molecule: Mol) -> list[float] | None:

    molecule = Chem.AddHs(molecule)
    
    params = rdDistGeom.ETKDGv3()
    # params.randomSeed = self.seed or 111
    params.numThreads = 0
    params.pruneRmsThresh = 0.001
    conf_ids = rdDistGeom.EmbedMultipleConfs(molecule, numConfs=100, params=params)
    print(len(conf_ids))
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule):
        return None
        print("UFF does not support this molecule. Skipping minimization and using approximate geometry.")
        best_id = self.pick_central_conformer(self.molecule_3d, conf_ids)
        # conf = self.molecule_3d.GetConformer(best_id)
        # self.molecule_3d.RemoveAllConformers()
        # self.molecule_3d.AddConformer(conf, assignId=True)
        return self.molecule_3d.GetConformer(best_id)
    
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(molecule)
    energies = []
    for cid in conf_ids:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule,
            mp,
            confId=cid
        )
        ff.Minimize()
        energies.append(ff.CalcEnergy())
    print(len(energies))
    return energies
    plt.hist(energies, bins=20)
    plt.xlabel("Energy (Units)")
    plt.ylabel("Frequency")
    plt.title("Conformer Energy Distribution")
    plt.show()


def plot_energy_dist():
    
    df = pd.read_csv("input.csv")
    random_rows = df.sample(frac=0.4)

    energy_values = []
    for _, row in random_rows.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        energies = get_energy_dist(mol)
        energy_values.append(energies if energies is not None else [])
    
    rel_energies = []
    for i in energy_values:
        m = min(i)
        rel_energies.extend([e - m for e in i])
    
    plt.hist(rel_energies, bins=50)
    plt.xlabel("Energy (Units)")
    plt.ylabel("Frequency")
    plt.title("Conformer Energy Distribution")
    plt.show()

        



def find_rmsd_distances(coverage: float) -> None:

    df = pd.read_csv("input.csv")
    random_rows = df.sample(frac=coverage)

    rmsd_values = []
    for _, row in random_rows.iterrows():
        smiles = row["SMILES"]
        name = row["Inhibitor Name"]
        cas = row["CAS Number"]
        url_cid = None
        url_cas = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/cids/TXT"
        fallback = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/SDF"
        response = requests.get(url_cas)
        if response.status_code == 200:
            cid = response.text.strip().split("\n")[0]
            url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        
        if url_cid:
            response1 = requests.get(url_cid)
        else:
            response1 = requests.get(fallback)

        if response1.status_code == 200:
            # with open(f"{name}.sdf", "w") as f:
            #     f.write(response.text)
            print(f"Downloaded {name}")

            imported_molecule = Chem.MolFromMolBlock(response1.text)
            molecule = MoleculeRepresentation(smiles, 8, True)
            conformer = molecule.find_best_conformer()
            mol3d = Chem.Mol(molecule.molecule)
            mol3d = Chem.AddHs(mol3d)
            mol3d.RemoveAllConformers()
            mol3d.AddConformer(conformer, assignId=True)
            mol3d = Chem.RemoveHs(mol3d)
            
            try:
                rmsd = rdMolAlign.GetBestRMS(imported_molecule, mol3d)
                rmsd_values.append(rmsd)
                print(f"RMSD for {name}: {rmsd:.3f}")
                del molecule, conformer, imported_molecule, mol3d
            except:
                del molecule, conformer, imported_molecule, mol3d
                continue
        else:
            print(f"Falied: {name}")
        gc.collect()


    plt.hist(rmsd_values, bins=10)
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Frequency")
    plt.title("RMSD Distribution")
    plt.show()


def prepare_mol_for_drawing(coverage: float) -> None:

    df = pd.read_csv("input.csv")
    random_rows = df.sample(frac=coverage)

    for _, row in random_rows.iterrows():
        smiles = row["SMILES"]
        name = row["Inhibitor Name"]
        cas = row["CAS Number"]
        url_cid = None
        url_cas = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/cids/TXT"
        fallback = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/SDF"
        response = requests.get(url_cas)
        if response.status_code == 200:
            cid = response.text.strip().split("\n")[0]
            url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
        
        if url_cid:
            response1 = requests.get(url_cid)
        else:
            response1 = requests.get(fallback)

        if response1.status_code == 200:
            # with open(f"{name}.sdf", "w") as f:
            #     f.write(response.text)
            print(f"Downloaded {name}")

            imported_molecule = Chem.MolFromMolBlock(response1.text)
            # imported_molecule = Chem.AddHs(imported_molecule, addCoords=True)
            molecule = MoleculeRepresentation(smiles, 8, True)
            conformer = molecule.find_best_conformer()
            mol3d = Chem.Mol(molecule.molecule)
            mol3d = Chem.AddHs(mol3d, addCoords=True)
            mol3d.RemoveAllConformers()
            mol3d.AddConformer(conformer, assignId=True)
            mol3d = Chem.RemoveHs(mol3d)
            
            try:
                rmsd = rdMolAlign.GetBestRMS(imported_molecule, mol3d)
                if rmsd < 0.5:
                    Chem.MolToMolFile(mol3d, "generated1.sdf")
                    Chem.MolToMolFile(imported_molecule, "correct1.sdf")
                    print(f"Name of Molecule: {name}")
                    return
                else:
                    del molecule, conformer, imported_molecule, mol3d
                    continue
            except:
                del molecule, conformer, imported_molecule, mol3d
                continue
        else:
            print(f"Falied: {name}")
        gc.collect()


find_rmsd_distances(1)
