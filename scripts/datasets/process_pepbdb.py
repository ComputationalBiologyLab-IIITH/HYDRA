import os
import pickle
import shutil
import argparse
import multiprocessing as mp

from tqdm.auto import tqdm
from functools import partial

from utils.data import PDBProtein


def load_item(item, path):
    protein_path = os.path.join(path, item[0])
    peptide_path = os.path.join(path, item[1])
    with open(protein_path, "r") as f:
        protein_block = f.read()
    with open(peptide_path, "r") as f:
        peptide_block = f.read()
    return protein_block, peptide_block


def process_item(item, args):
    try:
        protein_block, peptide_block = load_item(item, args.source)
        protein = PDBProtein(protein_block)
        peptide = PDBProtein(peptide_block)

        protein_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(peptide.__dict__, args.radius)
        )

        peptide_fn = item[1]
        pocket_fn = "/".join(peptide_fn.split("/")[:-1]) + f"/pocket{args.radius}.pdb"
        peptide_dest = os.path.join(args.dest, peptide_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(peptide_dest), exist_ok=True)

        shutil.copyfile(
            src=os.path.join(args.source, peptide_fn),
            dst=os.path.join(args.dest, peptide_fn),
        )
        with open(pocket_dest, "w") as f:
            f.write(protein_block_pocket)
        return pocket_fn, peptide_fn, item[0]  # item[0]: original protein filename

    except KeyError:
        print("UNK detected.", item)
        return None, item[1], item[0]

    except:
        print("Exception occurred.", item)
        return None, item[1], item[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./data/pepbdb")
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=False)

    with open(os.path.join(args.source, "index.pkl"), "rb") as f:
        items = pickle.load(f)

    with mp.Pool(args.num_workers) as pool:
        process_func = partial(process_item, args=args)
        results = list(
            tqdm(
                pool.imap(process_func, items),
                total=len(items),
                desc="Processing",
                dynamic_ncols=True,
            )
        )

    # remove failed items
    results = [r for r in results if r[0] is not None]

    with open(os.path.join(args.dest, "index.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"Done. {len(results)} pocket-peptide pairs in total.")
