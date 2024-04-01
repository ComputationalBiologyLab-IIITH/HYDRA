import numpy as np

from glob import glob
from fire import Fire
from tqdm.auto import tqdm

from utils.metrics import tanimoto


def evaluate_directory(result_dir):
    # get peptides
    peptides = glob(f"{result_dir}/peptide_*.pdb")

    # compute average pairwise tanimoto similarity
    scores = []
    for peptide1 in tqdm(peptides):
        for peptide2 in peptides:
            if peptide1 != peptide2:
                score = tanimoto.run(peptide1, peptide2)
                scores.append(score)
    scores = np.array(scores)
    ats = scores.mean()

    # write to file
    with open(f"{result_dir}/results-tanimoto.txt", "w") as f:
        f.write(f"{ats}\n")

    print("Done!")


def main(results_dir):
    for result_dir in tqdm(glob(f"{results_dir}/*/")):
        evaluate_directory(result_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
