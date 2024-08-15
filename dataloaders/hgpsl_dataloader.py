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

def custom_collate(batch):
    # Extract data and superfamilies
    data_list, superfamilies = zip(*batch)

    # Determine maximum number of nodes and features
    max_num_nodes = max(data.x.size(0) for data in data_list)
    num_features = max(data.x.size(1) if data.x.dim() == 2 else 1 for data in data_list)

    # Initialize lists to hold padded data
    padded_node_features = []
    padded_edge_indices = []

    for data in data_list:
        num_nodes = data.x.size(0)

        # Convert 1D node features to 2D
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(1)  # Convert to 2D [num_nodes, 1]

        # Print debug information
        print(f"Original x.shape={data.x.shape}")
        
        # Pad node features
        if num_nodes < max_num_nodes:
            pad_size = max_num_nodes - num_nodes
            padded_features = torch.cat([data.x, torch.zeros(pad_size, data.x.size(1))], dim=0)
        else:
            padded_features = data.x

        # Adjust edge indices for padding
        edge_index = data.edge_index
        if edge_index.size(1) > 0:
            edge_index = edge_index.clone()
            edge_index[0] = edge_index[0].clamp(max=max_num_nodes - 1)
            edge_index[1] = edge_index[1].clamp(max=max_num_nodes - 1)
        
        # Print debug information
        print(f"Edge index shape before adjustment={data.edge_index.shape}")
        print(f"Edge index shape after adjustment={edge_index.shape}")

        # Recreate Data object with padded features and adjusted edge indices
        padded_data = data.__class__(x=padded_features, edge_index=edge_index, **data.__dict__)
        padded_node_features.append(padded_data)
        
        # Debug the shape of each graph in the batch
        print(f"Padded features shape={padded_data.x.shape}")

    # Create a batch object with padded data
    batch = Batch.from_data_list(padded_node_features)
    
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
