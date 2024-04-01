"""
Adapted from: https://github.com/guanjq/targetdiff

License: MIT License
https://github.com/guanjq/targetdiff/blob/main/LICIENCE
"""

import torch
import torch.nn.functional as F

from utils.evaluation.atom_num_config import CONFIG


def get_space_size(pocket_3d_pos):
    aa_dist = F.pdist(pocket_3d_pos, p=2)
    aa_dist = torch.sort(aa_dist, 0, descending=True).values
    return torch.median(aa_dist[:10])


def _get_bin_idx(space_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size):
    bin_idx = _get_bin_idx(space_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    p = torch.tensor(prob_list)
    idx = p.multinomial(num_samples=1)
    return num_atom_list[idx] // 3
