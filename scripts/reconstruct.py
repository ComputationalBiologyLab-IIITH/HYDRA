import os
import pickle
import shutil

from glob import glob
from fire import Fire
from rdkit import Chem
from tqdm.auto import tqdm

import utils.misc as misc
from utils.common import distribute_array
from utils.reconstruct import reconstruct_peptide


def reconstruct_results(results_file, results_dir, config):
    # load results
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # reconstruct peptides
    print(f"Reconstructing [{results['id']}]...")
    reconstructed = []
    for i, peptide in enumerate(tqdm(results["peptides"])):
        try:
            xyz = peptide[0]
            aas = peptide[1]
            full_peptide = reconstruct_peptide(xyz, aas, results["protein"], config)

            if full_peptide is not None:
                reconstructed.append(full_peptide)

        except:
            print(f"Failed to reconstruct ({i}).")

    # save peptides as PDB
    pdb_id = results["protein"].split("/")[-2]
    out_dir = f"{results_dir}/reconstructed/{pdb_id}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(results["protein"], f"{out_dir}/protein.pdb")
    for i, mol in enumerate(reconstructed):
        Chem.MolToPDBFile(mol, f"{out_dir}/peptide_{i}.pdb", confId=0)

    print(f"Sucessfully reconstructed {len(reconstructed)} peptides.")

    return reconstructed


def main(results_dir, reconstruct_config_file, ndist=1, idx=0):
    # load config and set seed
    print("Loading config...")
    config = misc.load_config(reconstruct_config_file)

    # distribute targets
    targets = distribute_array(glob(f"{results_dir}/*"), ndist, idx)

    # reconstruct current targets
    for target in tqdm(targets):
        reconstruct_results(target, results_dir, config)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
