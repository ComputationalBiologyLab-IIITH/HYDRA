import re
import pandas as pd

from glob import glob
from fire import Fire
from tqdm.auto import tqdm

from utils.metrics import frodock
from utils.common import distribute_array


def evaluate_directory(result_dir):
    peptides = glob(f"{result_dir}/peptide_*.pdb")

    # compute scores
    scores = []
    for peptide in tqdm(peptides):
        try:
            score = frodock.run("protein", f"{peptide.replace('.pdb', '')}", result_dir)
            scores.append((peptide, score))
        except:
            pass

    # write to file
    with open(f"{result_dir}/results-frodock.csv", "w") as f:
        f.write(re.sub(r"\(|\)", "", "\n".join(map(str, scores))))

    print("Done!")


def main(results_dir, ndist=1, idx=0):
    # distribute targets
    targets = distribute_array(glob(f"{results_dir}/*"), ndist, idx)

    for result_dir in tqdm(targets):
        evaluate_directory(result_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
