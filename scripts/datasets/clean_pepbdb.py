import os
import pickle
import argparse

from rdkit import Chem
from glob import glob
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./data/pepbdb")
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--n_atom_thr", type=float, default=200)
    args = parser.parse_args()

    index = []
    skipped = []
    for rpcomplex in tqdm(glob(f"{args.source}/*")):
        complexname = rpcomplex.split("/")[-1]
        try:
            peptide = Chem.MolFromPDBFile(f"{rpcomplex}/peptide.pdb")
            if peptide.GetNumAtoms() < args.n_atom_thr:
                index.append(
                    (f"{complexname}/receptor.pdb", f"{complexname}/peptide.pdb")
                )

        except:
            skipped.append(complexname)

    print(f"Skipped {len(skipped)} complexes.")

    os.makedirs(args.dest, exist_ok=True)
    with open(os.path.join(args.dest, "index.pkl"), "wb") as f:
        pickle.dump(index, f)

    print(
        f"Done processing {len(index)} protein-peptide pairs in total.\nProcessed files in {args.dest}."
    )
