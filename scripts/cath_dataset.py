"""CATH domains as PST input graphs, labelled with their architecture.

Restored from the first version of this repository (``scripts/protein_domain_dataset.py``), which
followed the example dataset of the PST repository. Changes: the full C.A.T.H superfamily id is
stored instead of its last number only, and residues listed with alternate locations keep a single
C-alpha atom (before, such residues appeared twice in the sequence; 212 of the 6,273 domains had
at least one).
"""

import os

import esm
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from pst.utils import AA_THREE_TO_ONE
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph

from scripts.cath_labels import LABELS


class CathDomainDataset(Dataset):
    """One graph per CATH domain, in the input format of PST.

    Expects ``root/pdb/<cath_id>.pdb`` (one file per domain) and a CSV with the columns ``cath_id``,
    ``class``, ``architecture``, ``topology`` and ``superfamily``. Graphs are cached in ``root/graphs``.

    Every graph has
        x: ESM token ids of the sequence read from the C-alpha atoms, framed by <cls> and <eos>;
        edge_index: pairs of residues whose C-alpha atoms are within ``eps`` Å of each other
            (shifted by one position for the <cls> token);
        idx_mask: True at residue tokens;
        y: architecture label, see ``scripts/cath_labels.py``;
        cath_id, superfamily: CATH domain id and superfamily id ("3.90.1170.50"), as strings.
    """

    def __init__(self, root: str, cath_csv: str, eps: float = 8.0, transform=None, pre_transform=None):
        self.eps = eps
        self.cath = pd.read_csv(cath_csv).set_index("cath_id")
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "pdb")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "graphs")

    @property
    def raw_file_names(self) -> list[str]:
        return sorted(f for f in os.listdir(self.raw_dir) if f.endswith(".pdb"))

    @property
    def processed_file_names(self) -> list[str]:
        return [f.removesuffix(".pdb") + ".pt" for f in self.raw_file_names]

    def graph_from_pdb(self, path: str) -> Data:
        atoms = PandasPdb().read_pdb(path).df["ATOM"]
        ca = atoms[atoms["atom_name"] == "CA"]
        ca = ca.drop_duplicates(subset=["chain_id", "residue_number", "insertion"], keep="first")

        coords = torch.tensor(ca[["x_coord", "y_coord", "z_coord"]].to_numpy(), dtype=torch.float)
        edge_index = radius_graph(coords, r=self.eps, loop=False) + 1  # position 0 is the <cls> token

        sequence = "".join(AA_THREE_TO_ONE.get(res, "X") for res in ca["residue_name"])
        tokens = [self.alphabet.get_idx(aa) for aa in self.alphabet.tokenize(sequence)]
        x = torch.tensor([self.alphabet.cls_idx, *tokens, self.alphabet.eos_idx], dtype=torch.long)
        idx_mask = torch.zeros_like(x, dtype=torch.bool)
        idx_mask[1:-1] = True
        return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)

    def process(self):
        for pdb_name, graph_name in zip(self.raw_file_names, self.processed_file_names):
            cath_id = pdb_name.removesuffix(".pdb")
            row = self.cath.loc[cath_id]
            key = (int(row["class"]), int(row["architecture"]))
            if key not in LABELS:
                raise ValueError(f"{cath_id}: architecture {key} is not one of the ten classes")

            data = self.graph_from_pdb(os.path.join(self.raw_dir, pdb_name))
            data.y = torch.tensor([LABELS[key]])
            data.cath_id = cath_id
            data.superfamily = ".".join(str(int(row[c])) for c in ("class", "architecture", "topology", "superfamily"))
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, graph_name))

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        return torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
