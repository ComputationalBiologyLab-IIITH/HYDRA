from torch.utils.data import Dataset
from torch_geometric.data import Data


class ProteinPeptideData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinPeptideData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance["protein_" + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance["ligand_" + key] = item

        # instance["num_nodes"] = instance["ligand_amino_acid"].size(0)

        return instance
