import torch
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold  # Ensure to import KFold

from models.mlp_model import MLPTrainer
from dataloaders.mlp_dataloader import prepare_data

def main(args):
    # Initialize parameters
    input_dim = 1280
    hidden_dim = 256
    output_dim = 10
    dropout_rate = 0.3
    batch_size = 32
    max_epochs = 100
    n_splits = 5  # Number of folds

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and convert to tensors if necessary
    embeddings = torch.tensor(torch.load(f'{args.data_path}/protein_representations.pt')).to(device)  # Convert to tensor and move to device
    labels = torch.tensor(np.load(f'{args.data_path}/labels_cath.npy')).long().to(device)  # Convert to tensor and move to device
    superfamilies = torch.tensor(np.load(f'{args.data_path}/superfamilies.npy')).long().to(device)

    if args.mode == 'train':
        # Initialize K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Move superfamilies tensor to CPU and convert to NumPy array
        superfamilies_np = superfamilies.cpu().numpy()

        # Initialize logger
        csv_logger = CSVLogger(save_dir=f'logs/', name=f'mlp_training_fold')  # Ensure csv_logger is initialized

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
            train_loader, val_loader, test_loader = prepare_data(embeddings, labels, superfamilies, train_idx, val_idx, batch_size=batch_size)

            # Initialize model and move it to the device
            model = MLPTrainer(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

            # Set up checkpointing to save the best model
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'checkpoints/mlp/fold_{fold+1}',
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
                enable_progress_bar=True  # Make sure your version supports this argument
            )

            # Train the model
            trainer.fit(model, train_loader, val_loader)

            # Load the best model checkpoint and validate
            best_model_path = checkpoint_callback.best_model_path
            best_model = MLPTrainer.load_from_checkpoint(best_model_path, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)

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
        # Prepare data loaders for testing
        train_loader, val_loader, test_loader = prepare_data(embeddings, labels, superfamilies, train_idx, val_idx, batch_size=batch_size)

        # Load the model from the best checkpoint and move it to the device
        best_model = MLPTrainer.load_from_checkpoint(args.checkpoint_path, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)
        
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
        for batch in data_loader:
            if len(batch) == 2:  # Expected case: batch contains inputs and labels
                inputs, labels = batch
            elif len(batch) == 3:  # Case where batch contains inputs, labels, and additional data
                inputs, labels, _ = batch
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
    parser = argparse.ArgumentParser(description='Train or test an MLP model for protein classification.')
    parser.add_argument('mode', choices=['train', 'test'], help="Mode to run: 'train' or 'test'")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--checkpoint_path', type=str, required=False, help="Path to the model checkpoint for testing")
    args = parser.parse_args()

    main(args)
