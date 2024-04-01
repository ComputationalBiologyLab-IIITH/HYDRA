import os
import lmdb
import pickle

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from utils.data import PDBProtein
from utils.common import torchify_dict

from .common import ProteinPeptideData

FOLLOW_BATCH = (
    "protein_element",
    "ligand_element",
)


class PepBDBDataset(Dataset):
    def __init__(self, raw_path, transform=None, version="residues"):
        super().__init__()
        self.raw_path = raw_path.rstrip("/")
        self.index_path = os.path.join(self.raw_path, "index.pkl")
        self.processed_path = os.path.join(
            os.path.dirname(self.raw_path),
            os.path.basename(self.raw_path) + f"_processed_{version}.lmdb",
        )
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f"{self.processed_path} does not exist, begin processing data")
            self._process()

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, "rb") as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None:
                    continue
                try:
                    data_prefix = self.raw_path

                    pocket_dict = PDBProtein(
                        os.path.join(data_prefix, pocket_fn)
                    ).to_dict_atom()

                    ligand_dict = PDBProtein(
                        os.path.join(data_prefix, ligand_fn)
                    ).to_dict_residue()

                    data = ProteinPeptideData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )

                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn

                    # assert valid data
                    if data.protein_pos.size(0) == 0:
                        raise ValueError("No protein atoms")

                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(key=str(i).encode(), value=pickle.dumps(data))

                except:
                    num_skipped += 1
                    print(f"Skipping ({num_skipped}) {ligand_fn}")
                    continue

        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinPeptideData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
