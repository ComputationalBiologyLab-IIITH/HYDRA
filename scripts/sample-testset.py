import os
import pickle
import numpy as np

from fire import Fire
from tqdm.auto import tqdm
from datetime import datetime

import torch
import pytorch_lightning as pl

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from models.tarp import TARP

from datasets import get_dataset
from datasets.pepbdb import FOLLOW_BATCH

import utils.misc as misc
import utils.transforms as transforms

from utils.common import distribute_array


def sample_peptides_from_dataset(data_idx, dataset, model, trainer, config, out_dir):
    # output file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{timestamp}_{data_idx}.pkl"

    # generate dataset of of a specific pocket
    assert data_idx < len(
        dataset
    ), f"Data index {data_idx} out of range for test set of size {len(dataset)}"
    pred_set = torch.utils.data.Subset(dataset, [data_idx] * config.sample.num_samples)

    pred_loader = DataLoader(
        pred_set,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        batch_size=config.sample.batch_size,
    )

    print(f"Predicting [{data_idx}]...")
    preds = trainer.predict(model, pred_loader)

    # collate predictions
    peptide_map = [[], []]
    for batch in preds:
        for i in range(len(batch[0])):
            peptide_map[0].append(batch[0][i])
            peptide_map[1].append(batch[1][i])
    peptides = list(zip(peptide_map[0], peptide_map[1]))

    results = {
        "id": data_idx,
        "peptides": peptides,
        "protein": f"{config.data.path}/{pred_set[0].protein_filename}",
        "original_peptide": f"{config.data.path}/{pred_set[0].ligand_filename}",
    }

    # save results
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved to {out_file}")

    return results


def main(train_config_file, sample_config_file, out_dir="./outputs", ndist=1, idx=0):
    # load config and set seed
    print("Loading config...")
    train_config = misc.load_config(train_config_file)
    sample_config = misc.load_config(sample_config_file)
    misc.seed_all(sample_config.sample.seed)

    # combine configs
    config = train_config
    for key in sample_config:
        if key not in config:
            config[key] = sample_config[key]
        else:
            for subkey in sample_config[key]:
                config[key][subkey] = sample_config[key][subkey]

    # input transforms
    print("Preparing data...")
    protein_featurizer = transforms.FeaturizeProteinAtom()
    peptide_featurizer = transforms.FeaturizePeptideAtom(
        config.data.transform.ligand_atom_mode
    )

    transform_list = [protein_featurizer, peptide_featurizer]
    if config.data.transform.random_rot:
        transform_list.append(transforms.RandomRotation())

    transform = Compose(transform_list)

    # get (& split) datasets
    _, subsets = get_dataset(
        config=config.data, transform=transform, version=config.data.version
    )
    train_set, val_set = subsets["train"], subsets["test"]
    print(f"Training: {len(train_set)}, Test: {len(val_set)}")

    # load model
    model = TARP.load_from_checkpoint(
        config.model.checkpoint,
        config=config,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=peptide_featurizer.feature_dim,
    )

    # instantiate trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu")

    # distribute indices
    targets = distribute_array(np.arange(len(val_set)), ndist, idx)

    # predict for current indices
    for target in tqdm(targets):
        sample_peptides_from_dataset(target, val_set, model, trainer, config, out_dir)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
