import re

from glob import glob
from fire import Fire
from tqdm.auto import tqdm

from utils.metrics import crankpep


def evaluate_directory(result_dir):
    # get peptides
    peptides = glob(f"{result_dir}/peptide*.pdb")

    # compute affinities
    affinities = []
    for peptide in tqdm(peptides):
        # try:
        peptide_id = peptide.split("/")[-1]
        affinity = crankpep.run(
            f"{result_dir}/pocket10.pdb",
            f"{result_dir}/{peptide_id}",
            result_dir,
        )
        affinities.append((peptide_id, affinity))
        # except:
        # pass
    affinities = sorted(affinities)

    # write to file
    with open(f"{result_dir}/results-crankpep.csv", "w") as f:
        f.write(re.sub(r"\(|\)", "", "\n".join(map(str, affinities))))

    print("Done!")


def main(results_dir):
    for result_dir in tqdm(glob(f"{results_dir}/*/")):
        print(result_dir)
        evaluate_directory(result_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
