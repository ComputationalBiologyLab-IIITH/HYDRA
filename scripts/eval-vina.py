import re

from glob import glob
from fire import Fire
from tqdm.auto import tqdm

from utils.metrics import vina


def evaluate_directory(result_dir):
    # get peptides
    peptides = glob(f"{result_dir}/peptide_*.pdb")

    # compute affinities
    affinities = []
    for peptide in tqdm(peptides):
        # try:
        peptide_id = peptide.split("/")[-1]
        affinity = vina.run(
            "./protein.pdb",
            f"./{peptide_id}",
            result_dir,
            local=True,
        )
        affinities.append((peptide_id, affinity))
        # except:
            # pass
    affinities = sorted(affinities)

    # write to file
    with open(f"{result_dir}/results-vina.csv", "w") as f:
        f.write(re.sub(r"\(|\)", "", "\n".join(map(str, affinities))))

    print("Done!")


def main(results_dir):
    for result_dir in tqdm(glob(f"{results_dir}/*/")):
        evaluate_directory(result_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
