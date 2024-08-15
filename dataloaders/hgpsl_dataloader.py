import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, node_features, edge_indices, labels, superfamilies):
        self.data_list = []
        for nf, ei, label in zip(node_features, edge_indices, labels):
            data = Data(x=torch.tensor(nf, dtype=torch.float32),
                        edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                        y=torch.tensor(label, dtype=torch.long))
            self.data_list.append(data)
        self.superfamilies = torch.tensor(superfamilies, dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.superfamilies[idx]

def pad_features(features, size):
    padded_features = torch.zeros(size, features.size(1))
    padded_features[:features.size(0)] = features
    return padded_features

def pad_edge_index(edge_index, num_nodes):
    edge_index = edge_index.clone()
    edge_index = torch.cat([edge_index, torch.zeros((2, num_nodes - edge_index.max().item() - 1), dtype=torch.long)], dim=1)
    return edge_index

def custom_collate(batch):
    node_features = [item.x for item in batch]
    edge_indices = [item.edge_index for item in batch]
    labels = [item.y for item in batch]
    
    # Determine max sizes for padding
    max_nodes = max([x.size(0) for x in node_features])
    max_edges = max([edge_index.size(1) for edge_index in edge_indices])
    
    # Pad features and edge indices
    padded_node_features = [pad_features(x, max_nodes) for x in node_features]
    padded_edge_indices = [pad_edge_index(edge_index, max_nodes) for edge_index in edge_indices]

    # Create batched data
    batch = Batch.from_data_list([Data(x=x, edge_index=e, y=y) for x, e, y in zip(padded_node_features, padded_edge_indices, labels)])
    return batch

def check_tensor_sizes(tensors):
    sizes = [t.size() for t in tensors]
    print(f"Sizes of tensors: {sizes}")


def prepare_data(node_features, edge_indices, labels, superfamilies, train_idx=None, val_idx=None, test_idx=None, batch_size=32):
    # Create a dataset
    dataset = ProteinDataset(node_features, edge_indices, labels, superfamilies)

    if train_idx is None or val_idx is None or test_idx is None:
        # Get unique superfamilies
        unique_superfamilies = np.unique(superfamilies)

        # Stratified split: split unique superfamilies into train, val, test sets
        train_sf, temp_sf = train_test_split(unique_superfamilies, test_size=0.3, random_state=42)
        val_sf, test_sf = train_test_split(temp_sf, test_size=0.33, random_state=42)  # 0.33 of 0.3 is ~0.1

        # Get indices for each subset
        train_idx = [i for i, sf in enumerate(superfamilies) if sf in train_sf]
        val_idx = [i for i, sf in enumerate(superfamilies) if sf in val_sf]
        test_idx = [i for i, sf in enumerate(superfamilies) if sf in test_sf]

    # Debugging prints: Number of data entries in each subset
    print(f"Number of entries in train set: {len(train_idx)}")
    print(f"Number of entries in validation set: {len(val_idx)}")
    print(f"Number of entries in test set: {len(test_idx)}\n")
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx) if test_idx else None
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate) if test_dataset else None

    # Inspect a batch from each DataLoader
    for batch in train_loader:
        print("Training batch:", batch)
        break

    for batch in val_loader:
        print("Validation batch:", batch)
        break

    if test_loader:
        for batch in test_loader:
            print("Test batch:", batch)
            break

    return train_loader, val_loader, test_loader
