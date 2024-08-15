import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
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

def custom_collate(batch):
    # Extract data and superfamilies
    data_list, superfamilies = zip(*batch)

    # Use PyG Batch to handle variable-sized graphs
    batch = Batch.from_data_list(data_list)
    
    # Convert superfamilies to tensor
    superfamilies_tensor = torch.tensor(superfamilies, dtype=torch.long)
    
    return batch, superfamilies_tensor

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
    
    # Debugging prints: Superfamilies and their counts in each subset
    def print_superfamily_info(indices, split_name):
        indices_sf = np.array(superfamilies)[indices]  # Use NumPy array for indexing
        sf_counts = Counter(indices_sf)  # Count occurrences of each superfamily
        print(f"{split_name} set superfamilies and their counts:")
        for sf, count in sf_counts.items():
            print(f"Superfamily: {sf}, Number of protein domains: {count}")
        print()

    print_superfamily_info(train_idx, "Train")
    print_superfamily_info(val_idx, "Validation")
    print_superfamily_info(test_idx, "Test")
    
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
