import numpy as np

import torch
import torch.nn.functional as F

from datasets.common import ProteinPeptideData


MAP_PEPTIDE_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    16: 4,
}

# (atomic number, amino acid) -> index
MAP_PEPTIDE_TYPE_FULL_TO_INDEX = {
    (6, 0): 0,
    (1, 0): 1,
    (7, 0): 2,
    (8, 0): 3,
    (6, 1): 4,
    (1, 1): 5,
    (7, 1): 6,
    (8, 1): 7,
    (16, 1): 8,
    (6, 2): 9,
    (1, 2): 10,
    (7, 2): 11,
    (8, 2): 12,
    (6, 3): 13,
    (1, 3): 14,
    (7, 3): 15,
    (8, 3): 16,
    (6, 4): 17,
    (1, 4): 18,
    (7, 4): 19,
    (8, 4): 20,
    (6, 5): 21,
    (1, 5): 22,
    (7, 5): 23,
    (8, 5): 24,
    (6, 6): 25,
    (1, 6): 26,
    (7, 6): 27,
    (8, 6): 28,
    (6, 7): 29,
    (1, 7): 30,
    (7, 7): 31,
    (8, 7): 32,
    (6, 8): 33,
    (1, 8): 34,
    (7, 8): 35,
    (8, 8): 36,
    (6, 9): 37,
    (1, 9): 38,
    (7, 9): 39,
    (8, 9): 40,
    (6, 10): 41,
    (1, 10): 42,
    (7, 10): 43,
    (8, 10): 44,
    (16, 10): 45,
    (6, 11): 46,
    (1, 11): 47,
    (7, 11): 48,
    (8, 11): 49,
    (6, 12): 50,
    (1, 12): 51,
    (7, 12): 52,
    (8, 12): 53,
    (6, 13): 54,
    (1, 13): 55,
    (7, 13): 56,
    (8, 13): 57,
    (6, 14): 58,
    (1, 14): 59,
    (7, 14): 60,
    (8, 14): 61,
    (6, 15): 62,
    (1, 15): 63,
    (7, 15): 64,
    (8, 15): 65,
    (6, 16): 66,
    (1, 16): 67,
    (7, 16): 68,
    (8, 16): 69,
    (6, 17): 70,
    (1, 17): 71,
    (7, 17): 72,
    (8, 17): 73,
    (6, 18): 74,
    (1, 18): 75,
    (7, 18): 76,
    (8, 18): 77,
    (6, 19): 78,
    (1, 19): 79,
    (7, 19): 80,
    (8, 19): 81,
}

MAP_INDEX_TO_PEPTIDE_TYPE_ONLY = {
    v: k for k, v in MAP_PEPTIDE_TYPE_ONLY_TO_INDEX.items()
}
MAP_INDEX_TO_PEPTIDE_TYPE_FULL = {
    v: k for k, v in MAP_PEPTIDE_TYPE_FULL_TO_INDEX.items()
}


def get_index_peptide(atom_num, amino_acid, mode):
    if mode == "basic":
        return MAP_PEPTIDE_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == "pep_full":
        return MAP_PEPTIDE_TYPE_FULL_TO_INDEX[(int(atom_num), int(amino_acid))]
    else:
        raise ValueError


class FeaturizeProteinAtom(object):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(
            [1, 6, 7, 8, 16, 34]
        )  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinPeptideData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(
            1, -1
        )  # (N_atoms, N_elements)
        amino_acid = F.one_hot(
            data.protein_atom_to_aa_type, num_classes=self.max_num_aa
        )
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizePeptideAtom(object):
    def __init__(self, mode="pep_basic"):
        super().__init__()
        assert mode in ["pep_basic", "pep_full", "pep_residue"]
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == "pep_basic":
            return len(MAP_PEPTIDE_TYPE_ONLY_TO_INDEX)
        elif self.mode == "pep_full":
            return len(MAP_PEPTIDE_TYPE_FULL_TO_INDEX)
        elif self.mode == "pep_residue":
            return 20

    def __call__(self, data: ProteinPeptideData):
        if self.mode == "pep_residue":
            pos = data.ligand_center_of_mass
            x = data.ligand_amino_acid

            data.ligand_pos = pos
            data.ligand_atom_feature_full = x
            data.ligand_element = x

        else:
            element_list = data.ligand_element
            amino_acid_list = data.ligand_atom_to_aa_type

            x = [
                get_index_peptide(e, a, self.mode)
                for e, a in zip(element_list, amino_acid_list)
            ]
            x = torch.tensor(x)
            data.ligand_atom_feature_full = x

        return data


class RandomRotation(object):
    def __call__(self, data: ProteinPeptideData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data
