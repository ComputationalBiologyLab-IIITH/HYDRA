import torch

from .pepbdb import PepBDBDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == "pepbdb":
        dataset = PepBDBDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")

    if "train_split" in config:
        n_train = int(len(dataset) * config.train_split)
        n_test = len(dataset) - n_train
        subsets = torch.utils.data.random_split(dataset, [n_train, n_test])
        return dataset, {"train": subsets[0], "test": subsets[1]}

    return dataset
