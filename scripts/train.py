from fire import Fire
from utils import misc, transforms

import pytorch_lightning as pl

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning.loggers import WandbLogger

from datasets import get_dataset
from datasets.pepbdb import FOLLOW_BATCH

from models.tarp import TARP


def main(config, num_gpus=1, ckpt=None):
    # load config and set seed
    print("Loading config...")
    config = misc.load_config(config)
    misc.seed_all(config.train.seed)
    pl.seed_everything(config.train.seed)

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
        config=config.data,
        transform=transform,
        version=config.data.version,
    )
    train_set, test_set = subsets["train"], subsets["test"]
    val_set = train_set[:200]
    train_set = train_set[200:]
    print(f"Training: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    # prepare dataloaders
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        follow_batch=FOLLOW_BATCH,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        batch_size=config.train.batch_size,
    )

    # instantiate model
    print("Instantiating model...")
    print(f"Protein feature dim: {protein_featurizer.feature_dim}")
    print(f"Peptide feature dim: {peptide_featurizer.feature_dim}")

    if ckpt is not None:
        model = TARP.load_from_checkpoint(
            ckpt,
            config=config,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=peptide_featurizer.feature_dim,
        )
        print(f"Loaded checkpoint: {ckpt}")

    else:
        model = TARP(
            config,
            protein_featurizer.feature_dim,
            peptide_featurizer.feature_dim,
        )

    # instantiate wandb logger
    wandb_logger = WandbLogger(project=config.wandb.project)

    # instantiate trainer
    print("Instantiating trainer...")
    trainer = pl.Trainer(
        logger=wandb_logger,
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        max_epochs=config.train.max_epochs,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=(config.train.val_freq / len(train_loader)) * num_gpus,
    )

    # update config
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config["num_gpus"] = num_gpus
        wandb_logger.experiment.config["ckpt"] = ckpt
        wandb_logger.experiment.config["protein_featurize_dim"] = (
            protein_featurizer.feature_dim
        )
        wandb_logger.experiment.config["peptide_featurize_dim"] = (
            peptide_featurizer.feature_dim
        )

    # train model
    print("Training...")
    trainer.fit(model, train_loader, val_loader)

    print("Done!")


if __name__ == "__main__":
    Fire(main)
