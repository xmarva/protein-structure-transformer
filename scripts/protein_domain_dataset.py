import os
import torch
import esm
import torch_geometric.nn as gnn
from biopandas.pdb import PandasPdb
from torch_geometric.data import Data, Dataset
from pst.utils import AA_THREE_TO_ONE
from pst.utils import ARCHITECTURE_NAMES

class ProteinDomainDataset(Dataset):
    def __init__(
        self,
        root,
        cath_data,
        eps=8.0,
        esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
        num_workers=0,
        transform=None,
        pre_transform=None,
    ):
        self.eps = eps
        self.cath_data = cath_data
        self.esm_alphabet = esm_alphabet
        self.num_workers = num_workers
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        raw_files = os.listdir(os.path.join(self.root, "pdb"))
        if "processed" in raw_files:
            raw_files.remove("processed")
        if "graphs" in raw_files:
            raw_files.remove("graphs")
        return raw_files

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(len(self.raw_file_names))]

    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs")

    def get_graph_from_pdb(self, fname):
        pdb_contents = PandasPdb().read_pdb(fname).df["ATOM"]
        ca = pdb_contents[pdb_contents["atom_name"] == "CA"]
        structure = ca[["x_coord", "y_coord", "z_coord"]]
        structure = structure.to_numpy()
        structure = torch.tensor(structure, dtype=torch.float)
        edge_index = gnn.radius_graph(
            structure, r=self.eps, loop=False, num_workers=self.num_workers
        )
        edge_index += 1  # shift for cls_idx

        x = torch.cat(
            [
                torch.LongTensor([self.esm_alphabet.cls_idx]),
                torch.LongTensor(
                    [
                        self.esm_alphabet.get_idx(res)
                        for res in self.esm_alphabet.tokenize(
                            "".join(
                                ca["residue_name"]
                                .apply(lambda x: AA_THREE_TO_ONE.get(x, 'X'))
                                .tolist()
                            )
                        )
                    ]
                ),
                torch.LongTensor([self.esm_alphabet.eos_idx]),
            ]
        )
        idx_mask = torch.zeros_like(x, dtype=torch.bool)
        idx_mask[1:-1] = True

        return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)

    def process(self):
        processed_files_exist = all(
            os.path.isfile(os.path.join(self.processed_dir, f"data_{i}.pt"))
            for i in range(len(self.raw_file_names))
        )

        if processed_files_exist:
            print("Processed files already exist. Skipping the graph generation.")
            return

        architecture_to_label = {name: idx for idx, name in enumerate(ARCHITECTURE_NAMES.values())}

        idx = 0
        for raw_path in self.raw_paths:
            pdb_id = os.path.basename(raw_path).replace('.pdb', '')
            pdb_path = os.path.join(self.root, "pdb", pdb_id + ".pdb")

            data = self.get_graph_from_pdb(pdb_path)
 
            cath_labels = self.cath_data[self.cath_data['cath_id'] == pdb_id].iloc[0]
            class_label = cath_labels['class']
            architecture_label = cath_labels['architecture']
            superfamily_label = cath_labels['superfamily']

            data.superfamilies = torch.tensor(superfamily_label, dtype=torch.long)

            cath_label_key = (class_label, architecture_label)

            if cath_label_key in ARCHITECTURE_NAMES:
                architecture_name = ARCHITECTURE_NAMES[cath_label_key]
                data.cath_label = torch.tensor(architecture_to_label[architecture_name], dtype=torch.long)
            else:
                raise ValueError(f"No matching entry found in ARCHITECTURE_NAMES for class: {class_label}, architecture: {architecture_label}")

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data