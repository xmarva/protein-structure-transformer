import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels, superfamilies):
        self.embeddings = embeddings
        self.labels = labels
        self.superfamilies = superfamilies

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        superfamily = self.superfamilies[idx]
        return embedding, label, superfamily

def prepare_data(embeddings, labels, superfamilies, train_idx=None, val_idx=None, test_idx=None, batch_size=32):
    # Ensure superfamilies is on CPU and converted to numpy array
    superfamilies_cpu = superfamilies.cpu().numpy() if torch.is_tensor(superfamilies) else superfamilies

    # Create a dataset
    dataset = ProteinDataset(embeddings, labels, superfamilies_cpu)

    if train_idx is None or val_idx is None or test_idx is None:
        # Get unique superfamilies
        unique_superfamilies = np.unique(superfamilies_cpu)

        # Stratified split: split unique superfamilies into train, val, test sets
        train_sf, temp_sf = train_test_split(unique_superfamilies, test_size=0.3, random_state=42)
        val_sf, test_sf = train_test_split(temp_sf, test_size=0.33, random_state=42)  # 0.33 of 0.3 is ~0.1

        # Get indices for each subset
        train_idx = [i for i, sf in enumerate(superfamilies_cpu) if sf in train_sf]
        val_idx = [i for i, sf in enumerate(superfamilies_cpu) if sf in val_sf]
        test_idx = [i for i, sf in enumerate(superfamilies_cpu) if sf in test_sf]

    # Debugging prints: Number of data entries in each subset
    print(f"Number of entries in train set: {len(train_idx)}")
    print(f"Number of entries in validation set: {len(val_idx)}")
    print(f"Number of entries in test set: {len(test_idx)}\n")
    
    # Debugging prints: Superfamilies and their counts in each subset
    def print_superfamily_info(indices, split_name):
        # Convert indices to superfamilies and count occurrences
        indices_sf = superfamilies_cpu[indices]
        sf_counts = Counter(indices_sf)
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

    for batch in train_loader:
        print("Training batch:", batch)
        break

    for batch in val_loader:
        print("Validation batch:", batch)
        break



    return train_loader, val_loader, test_loader
