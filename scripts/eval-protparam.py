import pandas as pd

from glob import glob
from fire import Fire
from tqdm.auto import tqdm

from utils.metrics import protparam


def evaluate_directory(result_dir):
    results = []
    for seq_file in glob(f"{result_dir}/*"):
        target = seq_file.split("/")[-1].replace(".seq", "")
        scores = protparam.run(seq_file)
        scores["target"] = target
        results.append(scores)
    results_df = pd.DataFrame.from_dict(results)

    # write to file
    results_df.to_csv(f"{result_dir}/results-protparam.csv", index=False)

    print("Done!")


def main(results_dir):
    for result_dir in tqdm(glob(f"{results_dir}/*/")):
        evaluate_directory(result_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
