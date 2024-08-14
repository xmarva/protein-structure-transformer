import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label

def prepare_data(embeddings, labels, train_idx=None, val_idx=None, test_idx=None, batch_size=32):
    dataset = ProteinDataset(embeddings, labels)
    
    if train_idx is not None and val_idx is not None:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return train_loader, val_loader
    
    # Split dataset for training, validation, and test
    total_len = len(dataset)
    test_len = int(total_len * 0.1)
    val_len = int(total_len * 0.2)
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
