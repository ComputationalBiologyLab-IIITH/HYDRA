import numpy as np


def torchify_dict(data):
    import torch

    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def sanitize(filename):
    return filename.replace("/", ".")


def distribute_array(targets, ndist, idx):
    indices = np.argwhere(np.arange(len(targets)) % ndist == idx).flatten()
    targets = [targets[i] for i in indices]
    return targets
