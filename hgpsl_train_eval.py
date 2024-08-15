import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
from dataloaders.hgpsl_dataloader import prepare_data  # Adjust this import based on your directory structure
from models.hgp.hgpsl_model import Model  # Adjust this import based on your directory structure

import os
import torch
import numpy as np
from torch_geometric.data import Data

def load_data_from_directory(data_directory):
    node_features_list = []
    edge_indices_list = []
    labels_list = []
    superfamilies_list = []

    # List all .pt files in the directory
    for file_name in os.listdir(data_directory):
        if file_name.endswith('.pt'):
            file_path = os.path.join(data_directory, file_name)
            
            # Load the data
            data = torch.load(file_path)
            
            # Check if the data is an instance of `torch_geometric.data.Data`
            if isinstance(data, Data):
                # Extract and convert data to numpy arrays
                node_features = data.x.numpy()
                edge_indices = data.edge_index.numpy()
                labels = np.array([data.cath_label])  # Assuming a single label, adjust if needed
                superfamilies = np.array([data.superfamilies])  # Assuming a single superfamily, adjust if needed
                
                # Append to the lists
                node_features_list.append(node_features)
                edge_indices_list.append(edge_indices)
                labels_list.append(labels)
                superfamilies_list.append(superfamilies)
            else:
                print(f"File {file_name} is not of type 'torch_geometric.data.Data'.")

    # Convert lists to numpy arrays if needed for further processing
    # You may also want to handle concatenation or batching depending on your needs
    return node_features_list, edge_indices_list, labels_list, superfamilies_list


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

    data_directory = args.data_path

    # Load data
    node_features, edge_indices, labels, superfamilies = load_data_from_directory(data_directory)

    if args.mode == 'train':
        # Initialize K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Move superfamilies tensor to CPU and convert to NumPy array
        superfamilies_np = torch.tensor(superfamilies).cpu().numpy()

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
            model = Model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)

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
            best_model = Model.load_from_checkpoint(best_model_path).to(device)

            checkpoint_path = '/kaggle/working/best-checkpoint.ckpt'
            trainer.save_checkpoint(checkpoint_path)

            # Evaluate on validation set
            val_loss = trainer.validate(best_model, val_loader)[0]['val_loss']
            validation_scores.append(val_loss)

            # Evaluate the model on the entire validation data after training
            evaluate_model(best_model, val_loader, device)

        # Print the cross-validation results
        print("Cross-validation results:")
        print(f"Mean validation loss: {np.mean(validation_scores):.4f}")
        print(f"Standard deviation of validation loss: {np.std(validation_scores):.4f}")

    elif args.mode == 'test':
        # Load best model checkpoint
        checkpoint_path = '/kaggle/working/best-checkpoint.ckpt'
        model = Model.load_from_checkpoint(checkpoint_path).to(device)

        # Prepare data loaders for test
        _, _, test_loader = prepare_data(node_features, edge_indices, labels, superfamilies, batch_size=batch_size)

        # Evaluate the model on the test set
        evaluate_model(model, test_loader, device)

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for node_features, edge_index, labels, _ in data_loader:
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)

            outputs = model(node_features, edge_index)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
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
