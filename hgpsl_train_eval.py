import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
from torch_geometric.data import Data, DataLoader

from models.hgp.hgpsl_model import Model  # Adjust this import based on your directory structure
from dataloaders.hgpsl_dataloader import prepare_data  # Adjust this import based on your directory structure

import os
import torch
from torch_geometric.data import Data

def load_data_from_directory(directory):
    node_features_list = []
    edge_indices_list = []
    labels_list = []
    superfamilies_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            file_path = os.path.join(directory, filename)
            print(f"Loading file: {file_path}")  # Debug print to check file paths
            
            try:
                data = torch.load(file_path)
                
                # Debug print to inspect the type of data
                print(f"Type of data loaded: {type(data)}")
                
                if not isinstance(data, Data):
                    raise TypeError(f"Expected 'Data' type but got {type(data)}")

                # Extract node features, edge indices, and labels
                node_features_list.append(data.x)
                edge_indices_list.append(data.edge_index)
                labels_list.append(data.cath_label)
                superfamilies_list.append(data.superfamilies)
            
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
    
    # Concatenate node features and edge indices
    if node_features_list:
        node_features = torch.cat(node_features_list, dim=0)
    else:
        node_features = torch.tensor([])

    if edge_indices_list:
        edge_indices = torch.cat(edge_indices_list, dim=1)
    else:
        edge_indices = torch.tensor([[], []], dtype=torch.long)

    if labels_list:
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = torch.tensor([])

    if superfamilies_list:
        superfamilies = torch.cat(superfamilies_list, dim=0)
    else:
        superfamilies = torch.tensor([])

    return node_features, edge_indices, labels, superfamilies

def main(args):
    # Initialize parameters
    input_dim = args.num_features
    hidden_dim = args.nhid
    output_dim = args.num_classes
    dropout_rate = args.dropout_ratio
    batch_size = args.batch_size
    max_epochs = args.epochs
    n_splits = 5  # Number of folds

    # Determine the device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_directory = '/kaggle/working/medium-bio/data/graphs'

    # Load data and convert to tensors if necessary
    node_features, edge_indices, labels, superfamilies = load_data_from_directory(data_directory)

    if args.mode == 'train':
        # Initialize K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Move superfamilies tensor to CPU and convert to NumPy array
        superfamilies_np = superfamilies.cpu().numpy()

        # Initialize logger
        csv_logger = CSVLogger(save_dir='logs/', name='hgpsl_training_fold')

        # Track validation scores
        validation_scores = []

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.unique(superfamilies_np))):
            print(f"Fold {fold+1}/{n_splits}")

            # Create custom split for train and validation
            train_sf = np.unique(superfamilies_np)[train_idx]
            val_sf = np.unique(superfamilies_np)[val_idx]

            train_idx = [i for i, sf in enumerate(superfamilies_np) if sf in train_sf]
            val_idx = [i for i, sf in enumerate(superfamilies_np) if sf in val_sf]

            # Prepare data loaders for this fold
            train_loader, val_loader, test_loader = prepare_data(node_features, edge_indices, labels, superfamilies, train_idx, val_idx, batch_size=batch_size)

            # Initialize model and move it to the device
            model = Model(args).to(device)

            # Set up checkpointing to save the best model
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'checkpoints/hgpsl/fold_{fold+1}',
                filename='best-checkpoint',
                save_top_k=1,
                mode='min'
            )

            # Set up learning rate monitoring
            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            # Initialize trainer with the logger
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=[checkpoint_callback, lr_monitor],
                logger=csv_logger,
                enable_progress_bar=True
            )

            # Train the model
            trainer.fit(model, train_loader, val_loader)

            # Load the best model checkpoint and validate
            best_model_path = checkpoint_callback.best_model_path
            best_model = Model.load_from_checkpoint(best_model_path, **vars(args)).to(device)

            checkpoint_path = '/kaggle/working/best-checkpoint.ckpt'
            trainer.save_checkpoint(checkpoint_path)

            # Evaluate on validation set
            val_loss = trainer.validate(best_model, val_loader)[0]['val_loss']
            validation_scores.append(val_loss)

            # Evaluate the model on the entire validation data after training
            evaluate_model(best_model, val_loader, device)

        # Print the cross-validation results
        avg_val_loss = np.mean(validation_scores)
        print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")

    elif args.mode == 'test':
        # Ensure indices are properly initialized
        superfamilies_np = superfamilies.cpu().numpy()
        train_idx = list(range(len(superfamilies_np)))  # You may need to adjust this according to your setup
        val_idx = []  # Provide appropriate indices if you have validation data separate from training
        test_idx = list(range(len(superfamilies_np)))  # You may need to adjust this according to your setup

        # Prepare data loaders for testing
        train_loader, val_loader, test_loader = prepare_data(node_features, edge_indices, labels, superfamilies, train_idx, val_idx, batch_size=batch_size)

        # Load the model from the best checkpoint and move it to the device
        best_model = Model.load_from_checkpoint(args.checkpoint_path, **vars(args)).to(device)

        # Evaluate the model on the test data
        evaluate_model(best_model, test_loader, device)

def evaluate_model(model, data_loader, device):
    model.to(device)  # Ensure the model is on the correct device
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            if len(data) == 2:  # Expected case: batch contains inputs and labels
                inputs, labels = data
            elif len(data) == 3:  # Case where batch contains inputs, labels, and additional data
                inputs, labels, _ = data
            else:
                raise ValueError("Unexpected batch format")

            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the correct device

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Optionally, return predictions and labels for further analysis
    return all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test a Hierarchical Graph Pooling with Structure Learning model for protein classification.')
    parser.add_argument('mode', choices=['train', 'test'], help="Mode to run: 'train' or 'test'")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--checkpoint_path', type=str, required=False, help="Path to the model checkpoint for testing")
    parser.add_argument('--num_features', type=int, required=True, help="Number of features in the input data")
    parser.add_argument('--nhid', type=int, required=True, help="Number of hidden units")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes")
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help="Pooling ratio for the pooling layer")
    parser.add_argument('--sample_neighbor', type=bool, default=False, help="Whether to sample neighbors during training")
    parser.add_argument('--structure_learning', type=bool, default=False, help="Whether to use structure learning")
    parser.add_argument('--lamb', type=float, default=1.0, help="Lambda for regularization or structure learning")
    parser.add_argument('--sparse_attention', type=bool, default=False, help="Whether to use sparse attention")
    parser.add_argument('--dropout_ratio', type=float, required=True, help="Dropout ratio")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/testing')
    args = parser.parse_args()

    main(args)