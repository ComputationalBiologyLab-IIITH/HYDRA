import os
import pickle
import numpy as np

from fire import Fire
from datetime import datetime

import torch
import pytorch_lightning as pl

from torch_geometric.loader import DataLoader

from models.tarp import TARP

from datasets.pepbdb import FOLLOW_BATCH
from datasets.common import ProteinPeptideData

from utils import misc
from utils import transforms
from utils.data import PDBProtein
from utils.common import torchify_dict, sanitize


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.transform(self.data)


def main(pdb, train_config_file, sample_config_file, out_dir="./outputs"):
    # set up
    pdb_idx = sanitize(pdb)

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

    pocket_dict = PDBProtein(pdb).to_dict_atom()
    data = ProteinPeptideData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={},
    )
    dataset = ProteinDataset(data, transform=protein_featurizer)
    pred_set = torch.utils.data.Subset(dataset, [0] * config.sample.num_samples)
    pred_loader = DataLoader(
        pred_set,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        batch_size=config.sample.batch_size,
    )

    # load model
    model = TARP.load_from_checkpoint(
        config.model.checkpoint,
        config=config,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=peptide_featurizer.feature_dim,
    )

    # instantiate trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu")

    # output file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{timestamp}_{pdb_idx}.pkl"

    print(f"Predicting [{pdb_idx}]...")
    preds = trainer.predict(model, pred_loader)

    # collate predictions
    peptide_map = [[], []]
    for batch in preds:
        for i in range(len(batch[0])):
            peptide_map[0].append(batch[0][i])
            peptide_map[1].append(batch[1][i])
    peptides = list(zip(peptide_map[0], peptide_map[1]))

    results = {
        "id": pdb_idx,
        "peptides": peptides,
        "protein": pdb,
    }

    # save results
    with open(out_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Done! Saved to {out_file}")


if __name__ == "__main__":
    Fire(main)
